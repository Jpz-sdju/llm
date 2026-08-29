"""ToyLLM 唯一入口：文本 encode → Embedding → ToyLLM；以及有/无 Norm 对比。"""

import torch

from model_input import ToyLLMWithEmbed, texts_to_input_ids
from tokenizer_setup import decode, embedding_vocab_size, encode_split, load_qwen_tokenizer
from toyllm import ToyLLM, embed_init_std, fmt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ===\n")

dim = 64
n_layers = 70
batch_size = 4
seq_len = 8


# ═══════════════════════════════════════════════════════════════════════════
# Part 1：文本 → encode → Embedding → ToyLLM
# ═══════════════════════════════════════════════════════════════════════════

tokenizer = load_qwen_tokenizer()
vocab_size = embedding_vocab_size(tokenizer)
print(f"\ntokenizer.vocab_size = {tokenizer.vocab_size}")
print(f"embedding 行数       = {vocab_size}\n")

samples = ["""写在最前：
该MOD是基于1000人战场而制作。战场规模低于或高于1000，都可能会出现问题。
我已经将汉化文件发给了作者，如果我未及时更新，各位也可通过原址下载自带中文的最新版。
介绍：
你是否会困惑？
明明周围就有己方军团或部队，可是在交战中他们一点忙都帮不上。
如下图：
你以77人的部队迎战共151人的敌军。附近有一支325人的军团。
正常情况下，那支军团只会在你的战斗结束后出来收拾残局。
现在，使用这个MOD，只要你在战斗中坚持一段时间，
这支325人的军团就会作为援军出现在你的战场上，
正所谓，攻守之势异也！"""]

text = samples[0]
ids, pieces = encode_split(tokenizer, text)
print("pieces=", pieces)


# 2) 单条 → tensor → 模型
torch.manual_seed(42)
model = ToyLLMWithEmbed(vocab_size, dim=dim, n_layers=n_layers, use_norm=True).to(device)
model.eval()

input_ids, attention_mask = texts_to_input_ids(tokenizer, text, device=device)
print(f"[forward] tokens: {input_ids.shape[1]}, shape: {tuple(input_ids.shape)}")
with torch.no_grad():
    out = model(input_ids)
print(f"  ToyLLM 输出 shape: {tuple(out.shape)}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2：有 Norm vs 无 Norm（随机向量输入，逐层 RMS 诊断）
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"Part 2 | {n_layers} 层堆叠，有 Norm vs 无 Norm，看 X 的 RMS 逐层变化")
print("=" * 75)

torch.manual_seed(42)
model_norm = ToyLLM(dim, n_layers, use_norm=True).to(device)
torch.manual_seed(42)
model_nonorm = ToyLLM(dim, n_layers, use_norm=False).to(device)

for step in [0]:
    torch.manual_seed(123 + step)
    x0 = torch.randn(batch_size, seq_len, dim, device=device) * embed_init_std
    x_rms = torch.sqrt((x0 ** 2).mean()).item()
    print(f"\n>>> Step {step}   两模型共用同一输入 X，RMS = {fmt(x_rms)}  shape = {tuple(x0.shape)}\n")

    print("=" * 75)
    print("【模型 A】有 Pre-RMSNorm")
    print("=" * 75)
    with torch.no_grad():
        model_norm(x0.clone(), log_stats=True)

    print("\n\n" + "=" * 75)
    print("【模型 B】无 Norm")
    print("=" * 75)
    with torch.no_grad():
        model_nonorm(x0.clone(), log_stats=True)

print("\n" + "-" * 75)
