"""ToyLLM 唯一入口：TinyNews 多文档 → next-token 预测训练（Pre-RMSNorm）。"""

import random
import sys
import time
from pathlib import Path

import torch

from model_input import (
    ToyLLMWithEmbed,
    ids_lists_to_input_ids,
    interactive_ask,
    next_token_cross_entropy,
    texts_to_input_ids,
)
from tokenizer_setup import embedding_vocab_size, encode_split
from toyllm import ToyLLM, causal_mask, fmt
from utils import (
    ensure_tinyhelen_news,
    get_device,
    load_qwen_tokenizer,
    load_tinyhelen_texts,
    random_crop_text,
    redirect_stdout_to_log,
    restore_stdout,
)

# ── Demo 参数（以后改动改这里即可）──────────────────────────────────────────────────

# 运行环境
DEVICE = "auto"  # "auto" | "cpu" | "cuda" | "xpu"
LOG_PATH = Path(__file__).resolve().parent / "log"  # 训练 stdout 写入路径

# 模型结构
DIM = 128        # 隐藏维度（Embedding / Attention / FFN 宽度）
N_LAYERS = 16    # Pre-RMSNorm Block 堆叠层数
SEED = 42        # 权重初始化随机种子（torch.manual_seed）

# 语料
CORPUS_N = 5000     # 固定取 JSONL 前 N 篇训练；设为 None 则用全部有效篇

# 训练循环
BATCH_SIZE = 4       # 每 step 并行样本数（pad 成一批）；显存够可再加大
USE_CROP = True      # True=每 step 随机切 [CROP_MIN,CROP_MAX]；False=整篇 encode
TRAIN_STEPS = 5000   # 优化 step 数（每 step：抽 BATCH_SIZE 篇 → 1 次 forward/backward）
LR = 1e-3            # Adam 学习率
LOG_EVERY = 500      # 每 N step 打印 loss 与末层 Attention 统计
CROP_MIN = 128       # USE_CROP 时：窗口最短 token 数（整篇不足则不切）
CROP_MAX = 512       # USE_CROP 时：窗口最长 token 数
DETAIL_STEP = -1     # 等于某 step 时打印逐层矩阵与梯度；-1=关闭（极慢）

# 训练后交互
GEN_TOKENS = 32      # 终端问答：贪心续写的新 token 数

# ── 以下为实现逻辑 ──────────────────────────────────────────────────────────

device = get_device(DEVICE)
_real_stdout, _log_fp = redirect_stdout_to_log(LOG_PATH)
print(f"=== 运行设备: {device}  (DEVICE={DEVICE}) ===\n")
t_run0 = time.perf_counter()

tokenizer = load_qwen_tokenizer()
vocab_size = embedding_vocab_size(tokenizer)
print(f"\ntokenizer.vocab_size = {tokenizer.vocab_size}")
print(f"embedding 行数       = {vocab_size}\n")

news_path = ensure_tinyhelen_news()
_all_corpus = load_tinyhelen_texts(news_path)
corpus = _all_corpus if CORPUS_N is None else _all_corpus[:CORPUS_N]
print(f"TinyNews 语料: {news_path}")
if CORPUS_N is None:
    print(f"  训练篇数: {len(corpus)}  （全部有效篇）\n")
else:
    print(f"  训练篇数: {len(corpus)}  （固定取 JSONL 前 {CORPUS_N} 篇）\n")

preview_text = corpus[0]
ids, pieces = encode_split(tokenizer, preview_text)
print("预览第 1 篇 pieces=", pieces[:12], "...")

preview_ids, _ = texts_to_input_ids(tokenizer, preview_text, device=device)
print(f"[预览] tokens: {preview_ids.shape[1]}, shape: {tuple(preview_ids.shape)}\n")


def tensor_rms(t: torch.Tensor | None) -> float | None:
    if t is None:
        return None
    return torch.sqrt((t ** 2).mean()).item()


def fmt_opt(v: float | None) -> str:
    return "None" if v is None else fmt(v)


def attach_block_backward_hooks(model: ToyLLM) -> list:
    """在 backward 经过每个 Block 时打印激活梯度（顺序：Layer n-1 → 0）。"""
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, grad_input, grad_output):
            go = grad_output[0] if grad_output and grad_output[0] is not None else None
            gi = grad_input[0] if grad_input and grad_input[0] is not None else None
            print(f"  ◀ Layer {layer_idx} | Block 边界（反向）")
            print(f"    dL/d(Block输出) RMS = {fmt_opt(tensor_rms(go))}  ← 从 Layer {layer_idx + 1} / Loss 传来")
            dest = f"Layer {layer_idx - 1}" if layer_idx > 0 else "Embedding 输出"
            print(f"    dL/d(Block输入) RMS = {fmt_opt(tensor_rms(gi))}  → 继续传向 {dest}")

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_full_backward_hook(make_hook(i)))
    return handles


def print_layer_weight_grads(model: ToyLLM) -> None:
    """backward 完成后，按反向顺序打印各层权重梯度。"""
    print(f"\n  各层权重梯度（Layer {model.n_layers - 1} → 0）")
    for i in range(model.n_layers - 1, -1, -1):
        blk = model.blocks[i]
        parts = [
            f"RMSNorm_γ={fmt_opt(param_grad_rms(blk.attn.rms_norm.weight))}",
            f"W_q={fmt_opt(param_grad_rms(blk.attn.W_q.weight))}",
            f"W_k={fmt_opt(param_grad_rms(blk.attn.W_k.weight))}",
            f"W_v={fmt_opt(param_grad_rms(blk.attn.W_v.weight))}",
            f"W_o={fmt_opt(param_grad_rms(blk.W_o.weight))}",
            f"FFN1={fmt_opt(param_grad_rms(blk.ffn[0].weight))}",
            f"FFN2={fmt_opt(param_grad_rms(blk.ffn[2].weight))}",
        ]
        print(f"    Layer {i} | " + " | ".join(parts))


def param_grad_rms(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return tensor_rms(param.grad)


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


torch.manual_seed(SEED)
model = ToyLLMWithEmbed(vocab_size, dim=DIM, n_layers=N_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def snapshot_attn(model: ToyLLMWithEmbed, input_ids: torch.Tensor) -> dict[str, float]:
    """取最后一层 Attention 的 Scores/A 统计，用于看是否仍僵死。"""
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
        scores = torch.matmul(q, k.transpose(-1, -2)) / (DIM ** 0.5)
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


print("=" * 75)
print(f"训练：TinyNews next-token 预测 | {'random crop' if USE_CROP else '整篇'}")
print(
    f"steps={TRAIN_STEPS}, batch={BATCH_SIZE}, lr={LR}, dim={DIM}, layers={N_LAYERS}, "
    f"corpus={len(corpus)}, seed={SEED}"
)
if USE_CROP:
    print(f"crop: 每 step 随机切 [{CROP_MIN}, {CROP_MAX}] tokens（不足 {CROP_MIN} 则用整篇）")
else:
    print("crop: 关闭（每 step 用整篇 token 序列）")
print("=" * 75)
print("loss = CrossEntropy；位置 i 的 logits 预测 input_ids[i+1]")
print("判据：loss 能否下降；末层 Attention max 能否离开 ~1/seq_len（僵死=均匀）\n")

for step in range(TRAIN_STEPS):
    batch_texts = random.choices(corpus, k=BATCH_SIZE)
    if USE_CROP:
        batch_ids = [
            random_crop_text(tokenizer, t, min_len=CROP_MIN, max_len=CROP_MAX)
            for t in batch_texts
        ]
        input_ids, attention_mask = ids_lists_to_input_ids(
            tokenizer, batch_ids, device=device
        )
    else:
        input_ids, attention_mask = texts_to_input_ids(
            tokenizer, batch_texts, device=device
        )
    detail = step == DETAIL_STEP

    if detail:
        print("\n" + "=" * 75)
        print(f"详细追踪 Step {step} | 前向 Layer 0 → {N_LAYERS - 1}")
        print("=" * 75)

    logits = model(input_ids, log_stats=detail)
    loss = next_token_cross_entropy(logits, input_ids, attention_mask)

    hooks: list = []
    if detail:
        print(f"\n  ▼▼▼ 反向传播（Layer {N_LAYERS - 1} → 0）▼▼▼")
        hooks = attach_block_backward_hooks(model.toyllm)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if detail:
        print_layer_weight_grads(model.toyllm)
        remove_hooks(hooks)

    optimizer.step()

    if step % LOG_EVERY == 0 or step == TRAIN_STEPS - 1:
        stats = snapshot_attn(model, input_ids)
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
        print(f"  … Step {step}/{TRAIN_STEPS}", file=sys.__stdout__, flush=True)

print("-" * 75)
print("读结果：loss 明显下降、A.max 拉开 → Attention 在学；loss 不降 / A.max≈1/seq_len → 仍僵死")
print("-" * 75)
if device.type == "xpu":
    torch.xpu.synchronize()
elif device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t_run0
print(f"训练完成: {elapsed:.2f} s  |  device={device}  |  steps={TRAIN_STEPS}\n")
restore_stdout(
    _real_stdout,
    _log_fp,
    f"训练完成: {elapsed:.2f} s  |  日志: {LOG_PATH}",
    "下面进入交互问答\n",
)

interactive_ask(
    model,
    tokenizer,
    device=device,
    max_new_tokens=GEN_TOKENS,
)
