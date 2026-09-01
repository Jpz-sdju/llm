"""TinyNews next-token 训练循环与 TrainConfig。"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from model_input import (
    ToyForCausalLM,
    ids_lists_to_input_ids,
    next_token_cross_entropy,
    texts_to_input_ids,
)
from toyllm import ToyLLM, causal_mask, fmt
from utils import random_crop_text


@dataclass
class TrainConfig:
    device: str = "auto"
    log_dir: Path = Path("log")

    dim: int = 128
    n_layers: int = 16
    seed: int = 42

    corpus_n: int | None = 500

    batch_size: int = 4
    use_crop: bool = False
    train_steps: int = 10000
    lr: float = 1e-3
    log_every: int = 500
    crop_min: int = 128
    crop_max: int = 512
    # False=关；True=每步；list/tuple=指定 step
    detail_steps: bool | list[int] | tuple[int, ...] = False

    ckpt_path: Path = Path("checkpoints/toyllm.pt")
    save_ckpt: bool = True
    load_ckpt: Path | None = None

    interactive_after: bool = True
    gen_tokens: int = 32


def _tensor_rms(t: torch.Tensor | None) -> float | None:
    if t is None:
        return None
    return torch.sqrt((t ** 2).mean()).item()


def _fmt_opt(v: float | None) -> str:
    return "None" if v is None else fmt(v)


def _param_grad_rms(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return _tensor_rms(param.grad)


def _attach_block_backward_hooks(model: ToyLLM) -> list:
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, grad_input, grad_output):
            go = grad_output[0] if grad_output and grad_output[0] is not None else None
            gi = grad_input[0] if grad_input and grad_input[0] is not None else None
            print(f"  ◀ Layer {layer_idx} | Block 边界（反向）")
            print(f"    dL/d(Block输出) RMS = {_fmt_opt(_tensor_rms(go))}  ← 从 Layer {layer_idx + 1} / Loss 传来")
            dest = f"Layer {layer_idx - 1}" if layer_idx > 0 else "Embedding 输出"
            print(f"    dL/d(Block输入) RMS = {_fmt_opt(_tensor_rms(gi))}  → 继续传向 {dest}")

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_full_backward_hook(make_hook(i)))
    return handles


def _print_layer_weight_grads(model: ToyLLM) -> None:
    print(f"\n  各层权重梯度（Layer {model.n_layers - 1} → 0）")
    for i in range(model.n_layers - 1, -1, -1):
        blk = model.blocks[i]
        parts = [
            f"RMSNorm_γ={_fmt_opt(_param_grad_rms(blk.attn.rms_norm.weight))}",
            f"W_q={_fmt_opt(_param_grad_rms(blk.attn.W_q.weight))}",
            f"W_k={_fmt_opt(_param_grad_rms(blk.attn.W_k.weight))}",
            f"W_v={_fmt_opt(_param_grad_rms(blk.attn.W_v.weight))}",
            f"W_o={_fmt_opt(_param_grad_rms(blk.W_o.weight))}",
            f"FFN1={_fmt_opt(_param_grad_rms(blk.ffn[0].weight))}",
            f"FFN2={_fmt_opt(_param_grad_rms(blk.ffn[2].weight))}",
        ]
        print(f"    Layer {i} | " + " | ".join(parts))


def _snapshot_attn(
    model: ToyForCausalLM,
    input_ids: torch.Tensor,
    *,
    dim: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = model.toy.embed_tokens(input_ids)
        h = x
        for blk in model.toy.blocks[:-1]:
            h = blk(h)
        blk = model.toy.blocks[-1]
        attn = blk.attn
        x_in = h
        x_n = attn.rms_norm(x_in)
        q, k = attn.W_q(x_n), attn.W_k(x_n)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (dim ** 0.5)
        mask = ~causal_mask(scores.size(-1), scores.device)
        scores_masked = scores.masked_fill(~mask, float("nan"))
        scores_for_softmax = scores.masked_fill(~mask, float("-inf"))
        a = torch.softmax(scores_for_softmax, dim=-1)
        out = blk(x_in)
        finite = scores_masked[torch.isfinite(scores_masked)]
    return {
        "x_rms": torch.sqrt((out ** 2).mean()).item(),
        "s_std": finite.std().item() if finite.numel() else float("nan"),
        "a_max": a.max().item(),
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _sample_batch_corpus_indices(corpus_len: int, batch_size: int) -> list[int]:
    """从 corpus 无放回抽 batch_size 个下标；语料不足时取全部（不重复）。"""
    k = min(batch_size, corpus_len)
    return random.sample(range(corpus_len), k=k)


def _format_batch_corpus_indices(indices: list[int]) -> str:
    return ", ".join(f"seq[{i}]→corpus第{j + 1}篇" for i, j in enumerate(indices))


def run_train_loop(
    model: ToyForCausalLM,
    *,
    cfg: TrainConfig,
    corpus: list[str],
    tokenizer,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("=" * 75)
    print(f"训练：TinyNews next-token 预测 | {'random crop' if cfg.use_crop else '整篇'}")
    print(
        f"steps={cfg.train_steps}, batch={cfg.batch_size}, lr={cfg.lr}, "
        f"dim={cfg.dim}, layers={cfg.n_layers}, corpus={len(corpus)}, seed={cfg.seed}"
    )
    if cfg.use_crop:
        print(
            f"crop: 每 step 随机切 [{cfg.crop_min}, {cfg.crop_max}] tokens"
            f"（不足 {cfg.crop_min} 则用整篇）"
        )
    else:
        print("crop: 关闭（每 step 用整篇 token 序列）")
    print("=" * 75)
    print("loss = CrossEntropy；位置 i 的 logits 预测 input_ids[i+1]")
    print("判据：loss 能否下降；末层 Attention max 能否离开 ~1/seq_len（僵死=均匀）\n")

    if cfg.batch_size > len(corpus):
        print(
            f"  注意: batch_size={cfg.batch_size} > 语料篇数={len(corpus)}，"
            f"每 step 实际只用 {len(corpus)} 条（无放回、不重复）\n"
        )

    for step in range(cfg.train_steps):
        batch_indices = _sample_batch_corpus_indices(len(corpus), cfg.batch_size)
        batch_texts = [corpus[i] for i in batch_indices]
        print(f"Step {step} | batch 语料下标: {_format_batch_corpus_indices(batch_indices)}")
        if cfg.use_crop:
            batch_ids = [
                random_crop_text(tokenizer, t, min_len=cfg.crop_min, max_len=cfg.crop_max)
                for t in batch_texts
            ]
            input_ids, attention_mask = ids_lists_to_input_ids(
                tokenizer, batch_ids, device=device
            )
        else:
            input_ids, attention_mask = texts_to_input_ids(
                tokenizer, batch_texts, device=device
            )
        ds = cfg.detail_steps
        if ds is True:
            log_detail = True
        elif ds is False or ds is None:
            log_detail = False
        else:
            log_detail = step in ds

        logits = model(input_ids, log_stats=log_detail)
        loss = next_token_cross_entropy(logits, input_ids, attention_mask)

        hooks: list = []
        if log_detail:
            print(f"\n  ▼▼▼ 反向传播（Layer {cfg.n_layers - 1} → 0）▼▼▼")
            hooks = _attach_block_backward_hooks(model.toy)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if log_detail:
            _print_layer_weight_grads(model.toy)
            for h in hooks:
                h.remove()

        optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.train_steps - 1:
            stats = _snapshot_attn(model, input_ids, dim=cfg.dim)
            print(f">>> Step {step}")
            print(
                f"  loss={fmt(loss.item())} | "
                f"末层 S.std={fmt(stats['s_std'])} | A.max={fmt(stats['a_max'])} | X.rms={fmt(stats['x_rms'])}"
            )
            if not torch.isfinite(loss):
                print("  !! loss 非有限值（NaN/Inf），训练已崩")
                break
            print()
            sys.stdout.flush()
            print(f"  … Step {step}/{cfg.train_steps}", file=sys.__stdout__, flush=True)

    _sync_device(device)
    print(f"训练完成:  device={device}  |  steps={cfg.train_steps}\n")
