"""ToyLLM：ToyAttention + Block 堆叠（Attention demo / 训练共用）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

init_std = 0.02        # GPT-2 initializer_range
embed_init_std = 0.02  # Embedding 初始化 std


def fmt(v: float) -> str:
    """小数值多打几位，避免 Scores≈1e-5 显示成 0.0000。"""
    v = float(v)
    a = abs(v)
    if a >= 0.01:
        return f"{v:.6f}"
    if a >= 1e-8:
        return f"{v:.10f}"
    if a == 0.0:
        return "0"
    return f"{v:.4e}"


def init_linear_(module: nn.Module) -> None:
    """GPT-2 / HuggingFace 风格：Linear 权重 N(0, init_std²)，bias 置零。"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_embedding_(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=embed_init_std)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """(L, L) bool；True = 未来位置，softmax 前置为 -inf。位置 i 只能 attend 到 j≤i。"""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


class ToyAttention(nn.Module):
    def __init__(self, dim, use_norm=True):
        super().__init__()
        self.dim = dim
        self.use_norm = use_norm
        self.rms_norm = nn.RMSNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, log_stats=False, layer_idx=None):
        x_norm = self.rms_norm(x) if self.use_norm else x
        q = self.W_q(x_norm)
        k = self.W_k(x_norm)
        v = self.W_v(x_norm)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        scores = scores.masked_fill(causal_mask(scores.size(-1), scores.device), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        if log_stats:
            tag = "有 Pre-RMSNorm" if self.use_norm else "无 Norm"
            x_rms = torch.sqrt((x ** 2).mean()).item()
            print(f"  ┌─ Layer {layer_idx} | {tag} | Attention ─────────────────")
            print(f"  │ [Block 输入 X]  Mean: {fmt(x.mean())} | Std: {fmt(x.std())} | RMS: {fmt(x_rms)}")
            if self.use_norm:
                xn_rms = torch.sqrt((x_norm ** 2).mean()).item()
                print(f"  │ [X_norm]        Mean: {fmt(x_norm.mean())} | Std: {fmt(x_norm.std())} | RMS: {fmt(xn_rms)}")
            print(f"  │ [W_q 权重]      Mean: {fmt(self.W_q.weight.mean())} | Std: {fmt(self.W_q.weight.std())} | Max: {fmt(self.W_q.weight.abs().max())}")
            print(f"  │ [Q 激活]        Mean: {fmt(q.mean())} | Std: {fmt(q.std())} | RMS: {fmt(torch.sqrt((q**2).mean()))}")
            print(f"  │ [W_k 权重]      Mean: {fmt(self.W_k.weight.mean())} | Std: {fmt(self.W_k.weight.std())} | Max: {fmt(self.W_k.weight.abs().max())}")
            print(f"  │ [K 激活]        Mean: {fmt(k.mean())} | Std: {fmt(k.std())} | RMS: {fmt(torch.sqrt((k**2).mean()))}")
            print(f"  │ [Scores S]      Mean: {fmt(scores.mean())} | Std: {fmt(scores.std())} | Min: {fmt(scores.min())} | Max: {fmt(scores.max())}  (S = QK^T/√d, causal mask)")
            print(f"  │ [Attention A]   Mean: {fmt(attn_weights.mean())} | Std: {fmt(attn_weights.std())} | Min: {fmt(attn_weights.min())} | Max: {fmt(attn_weights.max())}  (A = softmax(S), 每行只看 j≤i)")
            print(f"  │ [V 激活]        Mean: {fmt(v.mean())} | Std: {fmt(v.std())} | RMS: {fmt(torch.sqrt((v**2).mean()))}")
            print(f"  │ [输出 O]        Mean: {fmt(out.mean())} | Std: {fmt(out.std())} | RMS: {fmt(torch.sqrt((out**2).mean()))}  (O = AV)")

        return out


class Block(nn.Module):
    def __init__(self, dim, use_norm=True):
        super().__init__()
        self.attn = ToyAttention(dim, use_norm=use_norm)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x, log_stats=False, layer_idx=None):
        x = x + self.W_o(self.attn(x, log_stats=log_stats, layer_idx=layer_idx))
        x = x + self.ffn(x)
        return x


class ToyLLM(nn.Module):
    def __init__(self, dim, n_layers, use_norm=True):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.use_norm = use_norm
        self.blocks = nn.ModuleList([Block(dim, use_norm) for _ in range(n_layers)])
        self.apply(init_linear_)

    def forward(self, x, log_stats=False):
        tag = "有 Pre-RMSNorm" if self.use_norm else "无 Norm"
        for i, blk in enumerate(self.blocks):
            x = blk(x, log_stats=log_stats, layer_idx=i)
            if log_stats:
                rms = torch.sqrt((x ** 2).mean()).item()
                std = x.std().item()
                print(f"  └─ Layer {i} | {tag} | Block 输出 → X RMS = {fmt(rms)}  Std = {fmt(std)}")
                print()
        return x
