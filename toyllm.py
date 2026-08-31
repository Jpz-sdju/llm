"""ToyLLM：Pre-RMSNorm Attention + Block 堆叠。"""

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


def _print_mat(name: str, t: torch.Tensor) -> None:
    """打印矩阵；每个最后一维向量附带 mean/std/rms。"""
    if t.device.type == "xpu":
        torch.xpu.synchronize()
    elif t.device.type == "cuda":
        torch.cuda.synchronize()
    t = t.detach().float().contiguous().cpu().clone()
    rms_all = torch.sqrt((t ** 2).mean()).item()
    absmax = t.abs().max().item()
    print(f"  │ [{name}] shape={tuple(t.shape)} | 全体 RMS={fmt(rms_all)} | absmax={fmt(absmax)}")

    vectors = t.reshape(-1, t.shape[-1])
    for i, vec in enumerate(vectors):
        mean = vec.mean().item()
        std = vec.std(unbiased=False).item()
        rms = torch.sqrt((vec ** 2).mean()).item()
        print(f"  │   vec[{i}] mean={fmt(mean)} | std={fmt(std)} | rms={fmt(rms)}")
        print(f"  │   {vec.tolist()}")
    print()


class ToyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.rms_norm = nn.RMSNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, log_stats=False, layer_idx=None):
        x_norm = self.rms_norm(x)
        q = self.W_q(x_norm)
        k = self.W_k(x_norm)
        v = self.W_v(x_norm)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        scores = scores.masked_fill(causal_mask(scores.size(-1), scores.device), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        if log_stats:
            print(f"  ┌─ Layer {layer_idx} | Pre-RMSNorm Attention ─────────────")
            _print_mat("X (Block 输入)", x)
            _print_mat("X_norm", x_norm)
            _print_mat("W_q", self.W_q.weight)
            _print_mat("Q", q)
            _print_mat("W_k", self.W_k.weight)
            _print_mat("K", k)
            _print_mat("W_v", self.W_v.weight)
            _print_mat("V", v)
            _print_mat("Scores S (= QK^T/√d, 未来为 -inf)", scores)
            _print_mat("Attention A (= softmax(S))", attn_weights)
            _print_mat("O (= A V)", out)

        return out


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = ToyAttention(dim)
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
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([Block(dim) for _ in range(n_layers)])
        self.apply(init_linear_)

    def forward(self, x, log_stats=False):
        if log_stats:
            torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, threshold=10**9)
        for i, blk in enumerate(self.blocks):
            x = blk(x, log_stats=log_stats, layer_idx=i)
            if log_stats:
                print(f"  └─ Layer {i} | Block 输出")
                _print_mat("X (Block 输出)", x)
        return x
