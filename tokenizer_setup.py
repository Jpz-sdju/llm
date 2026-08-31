"""Qwen3 tokenizer：字符串 ↔ token id，供 ToyLLM 训练流水线使用。

使用 Qwen/Qwen3-0.6B 的预训练词表（BPE 与 Qwen2 相同，vocab_size=151643）。
首次运行会从 HuggingFace 下载 tokenizer 文件（不含 Qwen3 神经网络权重）。
国内网络可设镜像：export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import os

# 未显式配置时默认走 hf-mirror（WSL/国内环境）；已有 HF_ENDPOINT 则不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

# HuggingFace 仓库 id：只下载 tokenizer，不加载 0.6B 模型权重
QWEN_TOKENIZER_ID = "Qwen/Qwen3-0.6B"


def load_qwen_tokenizer(model_id: str = QWEN_TOKENIZER_ID):
    """从 HuggingFace Hub（或本地缓存）加载 Qwen3 预训练 tokenizer。"""
    # 优先离线：已缓存时不访问外网（国内常因 ConnectTimeout 挂在 list_repo_templates）
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 批训练时需要 pad；Qwen 默认无 pad，用 eos 占位即可
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def encode_split(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """encode → (ids, pieces) 两个平行数组。"""
    ids = encode(tokenizer, text)
    pieces = [tokenizer.decode([i]) for i in ids]
    return ids, pieces


def embedding_vocab_size(tokenizer) -> int:
    """nn.Embedding 行数：须覆盖全部 token id（含 special tokens，可能 > tokenizer.vocab_size）。"""
    return len(tokenizer)


def demo() -> None:
    print(f"=== Qwen3 Tokenizer demo | {QWEN_TOKENIZER_ID} ===\n")

    tokenizer = load_qwen_tokenizer()
    print(f"vocab_size = {tokenizer.vocab_size}")
    print(f"pad_token  = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"eos_token  = {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})\n")

    samples = [
        "hello world",
        "你好，世界",
        "ToyLLM will use these token ids as Embedding row indices.",
    ]
    for text in samples:
        ids = encode(tokenizer, text)
        back = decode(tokenizer, ids)
        print(f"原文: {text!r}")
        print(f"ids : {ids}")
        print(f"还原: {back!r}")
        print(f"长度: {len(ids)} tokens\n")


if __name__ == "__main__":
    demo()
