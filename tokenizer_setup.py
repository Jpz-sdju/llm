"""Qwen3 tokenizer：字符串 ↔ token id（加载见 utils.load_qwen_tokenizer）。"""

from __future__ import annotations


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
    """nn.Embedding 行数：须覆盖全部 token id（含 special tokens）。"""
    return len(tokenizer)


def demo() -> None:
    from utils import QWEN_TOKENIZER_ID, load_qwen_tokenizer

    print(f"=== Qwen3 Tokenizer demo | {QWEN_TOKENIZER_ID} ===\n")
    tokenizer = load_qwen_tokenizer()
    print(f"vocab_size = {tokenizer.vocab_size}")
    print(f"pad_token  = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"eos_token  = {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})\n")
    for text in ["hello world", "你好，世界"]:
        ids = encode(tokenizer, text)
        print(f"原文: {text!r}")
        print(f"ids : {ids}")
        print(f"还原: {decode(tokenizer, ids)!r}\n")


if __name__ == "__main__":
    demo()
