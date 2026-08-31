"""ToyLLM 唯一入口：sample 文本 → next-token 预测训练（有/无 Norm 对比）。"""

import time
from pathlib import Path

import torch

from model_input import (
    ToyLLMWithEmbed,
    interactive_ask,
    next_token_cross_entropy,
    texts_to_input_ids,
)
from tokenizer_setup import embedding_vocab_size, encode_split
from toyllm import ToyLLM, causal_mask, fmt
from utils import get_device, load_qwen_tokenizer, redirect_stdout_to_log, restore_stdout

LOG_PATH = Path(__file__).resolve().parent / "log"

# 手动测速：改这里 → "cpu" / "xpu" / "cuda" / "auto"
DEVICE = "auto"
device = get_device(DEVICE)
_real_stdout, _log_fp = redirect_stdout_to_log(LOG_PATH)
print(f"=== 运行设备: {device}  (DEVICE={DEVICE}) ===\n")
t_run0 = time.perf_counter()

# dim = 512
dim = 128
n_layers = 16
# ── 输入文本 → token ──────────────────────────────────────────────────────

tokenizer = load_qwen_tokenizer()
vocab_size = embedding_vocab_size(tokenizer)
print(f"\ntokenizer.vocab_size = {tokenizer.vocab_size}")
print(f"embedding 行数       = {vocab_size}\n")

samples = ["""写在最前：该MOD是基于1000人战场而制作。战场规模低于或高于1000，都可能会出现问题。
我已经将汉化文件发给了作者，如果我未及时更新，各位也可通过原址下载自带中文的最新版。"""]

# samples = ["""写在最前：
# 该MOD是基于1000人战场而制作。战场规模低于或高于1000，都可能会出现问题。
# 我已经将汉化文件发给了作者，如果我未及时更新，各位也可通过原址下载自带中文的最新版。
# 介绍：
# 你是否会困惑？
# 明明周围就有己方军团或部队，可是在交战中他们一点忙都帮不上。
# 如下图：
# 你以77人的部队迎战共151人的敌军。附近有一支325人的军团。
# 正常情况下，那支军团只会在你的战斗结束后出来收拾残局。
# 现在，使用这个MOD，只要你在战斗中坚持一段时间，
# 这支325人的军团就会作为援军出现在你的战场上，
# 正所谓，攻守之势异也！"""]

text = samples[0]
ids, pieces = encode_split(tokenizer, text)
print("pieces=", pieces)

input_ids, attention_mask = texts_to_input_ids(tokenizer, text, device=device)
print(f"[input] tokens: {input_ids.shape[1]}, shape: {tuple(input_ids.shape)}\n")

# ── 训练：next-token 预测，有 Norm vs 无 Norm ─────────────────────────────

train_steps = 3000
lr = 1e-3
log_every = 500
detail_step = -1  # 设为某 step 才打印逐层前后向；-1=关闭（打开会极慢）
gen_tokens = 32   # 交互问答时贪心续写 token 数


def tensor_rms(t: torch.Tensor | None) -> float | None:
    if t is None:
        return None
    return torch.sqrt((t ** 2).mean()).item()


def fmt_opt(v: float | None) -> str:
    return "None" if v is None else fmt(v)


def attach_block_backward_hooks(model: ToyLLM, label: str) -> list:
    """在 backward 经过每个 Block 时打印激活梯度（顺序：Layer n-1 → 0）。"""
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, grad_input, grad_output):
            go = grad_output[0] if grad_output and grad_output[0] is not None else None
            gi = grad_input[0] if grad_input and grad_input[0] is not None else None
            print(f"  ◀ Layer {layer_idx} | {label} | Block 边界（反向）")
            print(f"    dL/d(Block输出) RMS = {fmt_opt(tensor_rms(go))}  ← 从 Layer {layer_idx + 1} / Loss 传来")
            dest = f"Layer {layer_idx - 1}" if layer_idx > 0 else "Embedding 输出"
            print(f"    dL/d(Block输入) RMS = {fmt_opt(tensor_rms(gi))}  → 继续传向 {dest}")

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_full_backward_hook(make_hook(i)))
    return handles


def print_layer_weight_grads(model: ToyLLM, label: str) -> None:
    """backward 完成后，按反向顺序打印各层权重梯度。"""
    print(f"\n  【{label}】各层权重梯度（Layer {model.n_layers - 1} → 0）")
    for i in range(model.n_layers - 1, -1, -1):
        blk = model.blocks[i]
        parts = [
            f"W_q={fmt_opt(param_grad_rms(blk.attn.W_q.weight))}",
            f"W_k={fmt_opt(param_grad_rms(blk.attn.W_k.weight))}",
            f"W_v={fmt_opt(param_grad_rms(blk.attn.W_v.weight))}",
            f"W_o={fmt_opt(param_grad_rms(blk.W_o.weight))}",
            f"FFN1={fmt_opt(param_grad_rms(blk.ffn[0].weight))}",
            f"FFN2={fmt_opt(param_grad_rms(blk.ffn[2].weight))}",
        ]
        if blk.attn.use_norm and blk.attn.rms_norm.weight is not None:
            parts.insert(0, f"RMSNorm_γ={fmt_opt(param_grad_rms(blk.attn.rms_norm.weight))}")
        print(f"    Layer {i} | " + " | ".join(parts))


def param_grad_rms(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return tensor_rms(param.grad)


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


torch.manual_seed(42)
model_norm = ToyLLMWithEmbed(vocab_size, dim=dim, n_layers=n_layers, use_norm=True).to(device)
torch.manual_seed(42)
model_nonorm = ToyLLMWithEmbed(vocab_size, dim=dim, n_layers=n_layers, use_norm=False).to(device)

opt_norm = torch.optim.Adam(model_norm.parameters(), lr=lr)
opt_nonorm = torch.optim.Adam(model_nonorm.parameters(), lr=lr)


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
        x_n = attn.rms_norm(x_in) if attn.use_norm else x_in
        q, k = attn.W_q(x_n), attn.W_k(x_n)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (dim ** 0.5)
        scores = scores.masked_fill(causal_mask(scores.size(-1), scores.device), float("-inf"))
        a = torch.softmax(scores, dim=-1)
        out = blk(x_in)
    return {
        "x_rms": torch.sqrt((out ** 2).mean()).item(),
        "s_std": scores.std().item(),
        "a_max": a.max().item(),
        "a_std": a.std().item(),
    }


print("=" * 75)
print(f"训练对比：sample 文本 next-token 预测 | 有 Norm vs 无 Norm")
print(f"steps={train_steps}, lr={lr}, layers={n_layers}, seq={input_ids.shape[1]}")
print("=" * 75)
print("loss = CrossEntropy；位置 i 的 logits 预测 input_ids[i+1]")
print("判据：loss 能否下降；末层 Attention max 能否离开 ~1/seq_len（僵死=均匀）\n")

for step in range(train_steps):
    detail = step == detail_step

    if detail:
        print("\n" + "=" * 75)
        print(f"详细追踪 Step {step} | 有 Pre-RMSNorm | 前向 Layer 0 → {n_layers - 1}")
        print("=" * 75)

    logits_n = model_norm(input_ids, log_stats=detail)
    loss_n = next_token_cross_entropy(logits_n, input_ids, attention_mask)

    hooks_n: list = []
    if detail:
        print(f"\n  ▼▼▼ 有 Norm | 反向传播（Layer {n_layers - 1} → 0）▼▼▼")
        hooks_n = attach_block_backward_hooks(model_norm.toyllm, "有 Norm")

    opt_norm.zero_grad(set_to_none=True)
    loss_n.backward()

    if detail:
        print_layer_weight_grads(model_norm.toyllm, "有 Norm")
        remove_hooks(hooks_n)

    opt_norm.step()

    if detail:
        print("\n" + "=" * 75)
        print(f"详细追踪 Step {step} | 无 Norm | 前向 Layer 0 → {n_layers - 1}")
        print("=" * 75)

    logits_0 = model_nonorm(input_ids, log_stats=detail)
    loss_0 = next_token_cross_entropy(logits_0, input_ids, attention_mask)

    hooks_0: list = []
    if detail:
        print(f"\n  ▼▼▼ 无 Norm | 反向传播（Layer {n_layers - 1} → 0）▼▼▼")
        hooks_0 = attach_block_backward_hooks(model_nonorm.toyllm, "无 Norm")

    opt_nonorm.zero_grad(set_to_none=True)
    loss_0.backward()

    if detail:
        print_layer_weight_grads(model_nonorm.toyllm, "无 Norm")
        remove_hooks(hooks_0)

    opt_nonorm.step()

    if step % log_every == 0 or step == train_steps - 1:
        sn = snapshot_attn(model_norm, input_ids)
        s0 = snapshot_attn(model_nonorm, input_ids)
        print(f">>> Step {step}")
        print(
            f"  有 Norm  | loss={fmt(loss_n.item())} | "
            f"末层 S.std={fmt(sn['s_std'])} | A.max={fmt(sn['a_max'])} | X.rms={fmt(sn['x_rms'])}"
        )
        print(
            f"  无 Norm  | loss={fmt(loss_0.item())} | "
            f"末层 S.std={fmt(s0['s_std'])} | A.max={fmt(s0['a_max'])} | X.rms={fmt(s0['x_rms'])}"
        )
        if not torch.isfinite(loss_0):
            print("  !! 无 Norm loss 非有限值（NaN/Inf），训练已崩")
            break
        print()

print("-" * 75)
print("读结果：若有 Norm loss 明显下降、A.max 拉开；无 Norm loss 不降 / A.max≈1/seq_len / 出现 NaN → 训练救不活无 Norm")
print("-" * 75)
if device.type == "xpu":
    torch.xpu.synchronize()
elif device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t_run0
print(f"训练完成: {elapsed:.2f} s  |  device={device}  |  steps={train_steps}\n")
restore_stdout(
    _real_stdout,
    _log_fp,
    f"训练完成: {elapsed:.2f} s  |  日志: {LOG_PATH}",
    "下面进入交互问答\n",
)

interactive_ask(
    model_norm,
    model_nonorm,
    tokenizer,
    device=device,
    max_new_tokens=gen_tokens,
)
