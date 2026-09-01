"""TinyNews next-token 训练逻辑（由 attention_demo.py 调用）。"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from model_input import (
    ToyLLMWithEmbed,
    ids_lists_to_input_ids,
    interactive_ask,
    load_checkpoint,
    next_token_cross_entropy,
    save_checkpoint,
    texts_to_input_ids,
)
from tokenizer_setup import embedding_vocab_size, encode_split
from toyllm import ToyLLM, causal_mask, fmt
from utils import (
    QWEN_TOKENIZER_ID,
    ensure_tinyhelen_news,
    get_device,
    load_qwen_tokenizer,
    load_tinyhelen_texts,
    random_crop_text,
    redirect_stdout_to_log,
    restore_stdout,
)


@dataclass
class TrainConfig:
    device: str = "auto"
    log_path: Path = Path("log")

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
    detail_step: int = -1

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
    model: ToyLLMWithEmbed,
    input_ids: torch.Tensor,
    *,
    dim: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = model.embed(input_ids)
        h = x
        for blk in model.toyllm.blocks[:-1]:
            h = blk(h)
        blk = model.toyllm.blocks[-1]
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
        "a_std": a.std().item(),
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _print_corpus_preview(
    tokenizer,
    corpus: list[str],
    *,
    device: torch.device,
) -> None:
    preview_text = corpus[0]
    ids, pieces = encode_split(tokenizer, preview_text)
    print("预览第 1 篇 pieces=", pieces[:12], "...")
    preview_ids, _ = texts_to_input_ids(tokenizer, preview_text, device=device)
    print(f"[预览] tokens: {preview_ids.shape[1]}, shape: {tuple(preview_ids.shape)}\n")


def run_train_loop(
    model: ToyLLMWithEmbed,
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

    for step in range(cfg.train_steps):
        batch_texts = random.choices(corpus, k=cfg.batch_size)
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
        detail = step == cfg.detail_step

        if detail:
            print("\n" + "=" * 75)
            print(f"详细追踪 Step {step} | 前向 Layer 0 → {cfg.n_layers - 1}")
            print("=" * 75)

        logits = model(input_ids, log_stats=detail)
        loss = next_token_cross_entropy(logits, input_ids, attention_mask)

        hooks: list = []
        if detail:
            print(f"\n  ▼▼▼ 反向传播（Layer {cfg.n_layers - 1} → 0）▼▼▼")
            hooks = _attach_block_backward_hooks(model.toyllm)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if detail:
            _print_layer_weight_grads(model.toyllm)
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

    print("-" * 75)
    print("读结果：loss 明显下降、A.max 拉开 → Attention 在学；loss 不降 / A.max≈1/seq_len → 仍僵死")
    print("-" * 75)
    _sync_device(device)
    print(f"训练完成:  device={device}  |  steps={cfg.train_steps}\n")


def run(cfg: TrainConfig) -> ToyLLMWithEmbed:
    device = get_device(cfg.device)
    real_stdout, log_fp = redirect_stdout_to_log(cfg.log_path)
    t_run0 = time.perf_counter()
    print(f"=== 运行设备: {device}  (DEVICE={cfg.device}) ===\n")

    tokenizer = load_qwen_tokenizer()
    vocab_size = embedding_vocab_size(tokenizer)
    print(f"\ntokenizer.vocab_size = {tokenizer.vocab_size}")
    print(f"embedding 行数       = {vocab_size}\n")

    if cfg.load_ckpt is not None:
        ckpt_path = Path(cfg.load_ckpt)
        print(f"跳过训练，加载 checkpoint: {ckpt_path.resolve()}")
        model, ckpt_cfg = load_checkpoint(ckpt_path, device=device)
        print(
            f"  dim={ckpt_cfg['dim']}, layers={ckpt_cfg['n_layers']}, "
            f"vocab={ckpt_cfg['vocab_size']}, tokenizer={ckpt_cfg['tokenizer_id']}"
        )
        if ckpt_cfg.get("train_steps") is not None:
            print(f"  训练 step 数（记录）: {ckpt_cfg['train_steps']}")
        elapsed = time.perf_counter() - t_run0
        restore_stdout(
            real_stdout,
            log_fp,
            f"已加载 checkpoint（{elapsed:.2f} s）|  日志: {cfg.log_path}",
            "下面进入交互问答\n" if cfg.interactive_after else "",
        )
    else:
        news_path = ensure_tinyhelen_news()
        all_corpus = load_tinyhelen_texts(news_path)
        corpus = all_corpus if cfg.corpus_n is None else all_corpus[: cfg.corpus_n]
        print(f"TinyNews 语料: {news_path}")
        if cfg.corpus_n is None:
            print(f"  训练篇数: {len(corpus)}  （全部有效篇）\n")
        else:
            print(f"  训练篇数: {len(corpus)}  （固定取 JSONL 前 {cfg.corpus_n} 篇）\n")
        _print_corpus_preview(tokenizer, corpus, device=device)

        torch.manual_seed(cfg.seed)
        model = ToyLLMWithEmbed(vocab_size, dim=cfg.dim, n_layers=cfg.n_layers).to(device)
        run_train_loop(model, cfg=cfg, corpus=corpus, tokenizer=tokenizer, device=device)

        if cfg.save_ckpt:
            saved = save_checkpoint(
                cfg.ckpt_path,
                model,
                tokenizer_id=QWEN_TOKENIZER_ID,
                train_steps=cfg.train_steps,
            )
            print(f"已保存 checkpoint → {saved.resolve()}\n")

        elapsed = time.perf_counter() - t_run0
        restore_stdout(
            real_stdout,
            log_fp,
            f"训练完成: {elapsed:.2f} s  |  日志: {cfg.log_path}",
            "下面进入交互问答\n" if cfg.interactive_after else "",
        )

    if cfg.interactive_after:
        interactive_ask(model, tokenizer, device=device, max_new_tokens=cfg.gen_tokens)

    return model
