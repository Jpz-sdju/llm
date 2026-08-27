import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ===\n")

dim = 64
seq_len = 8
batch_size = 4
embed_init_std = 0.02  # 模拟 Embedding 初始化后 X 每维的尺度（GPT 类常用值）


# 1. 定义一个简单的自注意力层（不带 QK-Norm）
class ToyAttention(nn.Module):
    def __init__(self, dim, use_norm=True):   # 加了 use_norm，默认开 Norm
        super().__init__()
        self.dim = dim
        self.use_norm = use_norm
        self.rms_norm = nn.RMSNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.W_q.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.W_k.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.W_v.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    def forward(self, x, log_stats=False):
        # Pre-RMSNorm：use_norm=False 时跳过 Norm，直接用原始 x（用来对比无 Norm）
        x_norm = self.rms_norm(x) if self.use_norm else x
        q = self.W_q(x_norm)
        k = self.W_k(x_norm)
        v = self.W_v(x_norm)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        if log_stats:
            tag = "有 Pre-RMSNorm" if self.use_norm else "无 Norm"
            print(f"  ── {tag} ──")
            print(f"  [X (norm | no norm)激活]   Mean: {x_norm.mean().item():.4f} | Std: {x_norm.std().item():.4f} | RMS: {torch.sqrt((x_norm**2).mean()).item():.4f}")
            print(f"  [W_q 权重]     Std: {self.W_q.weight.std().item():.4f} | Max: {self.W_q.weight.abs().max().item():.4f}")
            print(f"  [Q 激活]       Mean: {q.mean().item():.4f} | Std: {q.std().item():.4f} | RMS: {torch.sqrt((q**2).mean()).item():.4f}")
            print(f"  [Scores (S)]  Mean: {scores.mean().item():.4f} | Std: {scores.std().item():.4f} | Max: {scores.max().item():.4f}")
            print(f"  [Softmax 概率] Max: {attn_weights.max().item():.4f} (越接近 1 说明越极化)")
            print("-" * 75)

        return out


# ════════════════════════════════════════════════════════════
# 在 ToyAttention 基础上堆多层
#   Block = ToyAttention + W_o + 残差 + FFN + 残差
#   MiniLM = 堆 N 个 Block
# ════════════════════════════════════════════════════════════
n_layers = 6


class Block(nn.Module):
    def __init__(self, dim, use_norm=True):
        super().__init__()
        self.attn = ToyAttention(dim, use_norm=use_norm)     # 复用你的 ToyAttention
        self.W_o = nn.Linear(dim, dim, bias=False)           # 输出投影
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )
        nn.init.normal_(self.W_o.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    def forward(self, x):
        x = x + self.W_o(self.attn(x))                      # 残差 1
        x = x + self.ffn(x)                                  # 残差 2
        return x


class MiniLM(nn.Module):
    def __init__(self, dim, n_layers, use_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, use_norm) for _ in range(n_layers)])

    def forward(self, x, log_stats=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if log_stats:
                rms = torch.sqrt((x ** 2).mean()).item()
                print(f"  Layer {i}: X RMS = {rms:.4f}  Std = {x.std().item():.4f}")
        return x


print("\n" + "=" * 75)
print(f"{n_layers} 层堆叠，有 Norm vs 无 Norm，看 X 的 RMS 逐层变化")
print("=" * 75)

torch.manual_seed(42)
model_norm   = MiniLM(dim, n_layers, use_norm=True).to(device)
torch.manual_seed(42)
model_nonorm = MiniLM(dim, n_layers, use_norm=False).to(device)

for step in [0, 50, 200, 500]:
    torch.manual_seed(123 + step)
    x0 = torch.randn(batch_size, seq_len, dim, device=device) * embed_init_std

    print(f"\n>>> Step {step}   输入 X 初始 RMS = {torch.sqrt((x0**2).mean()).item():.4f}")
    print("── 有 Pre-RMSNorm ──")
    with torch.no_grad():
        model_norm(x0, log_stats=True)
    print("── 无 Norm（看 X 是否越变越大）──")
    with torch.no_grad():
        model_nonorm(x0, log_stats=True)
    print("-" * 75)
