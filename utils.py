"""通用工具：设备选择、Qwen tokenizer 加载。"""

from __future__ import annotations

import os

import torch

# 未显式配置时默认走 hf-mirror；已有 HF_ENDPOINT 则不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

QWEN_TOKENIZER_ID = "Qwen/Qwen3-0.6B"


def get_device(name: str = "auto") -> torch.device:
    """选择运行设备。name: \"auto\" | \"cpu\" | \"xpu\" | \"cuda\"。"""
    wanted = name.strip().lower()
    if wanted == "auto":
        if torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if wanted == "xpu":
        if not torch.xpu.is_available():
            raise RuntimeError("name=xpu 但 torch.xpu.is_available()=False")
        return torch.device("xpu")
    if wanted == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("name=cuda 但 torch.cuda.is_available()=False")
        return torch.device("cuda")
    if wanted == "cpu":
        return torch.device("cpu")
    raise ValueError(f'name 只能是 "auto"|"cpu"|"xpu"|"cuda"，当前={name!r}')


def load_qwen_tokenizer(model_id: str = QWEN_TOKENIZER_ID):
    """下载/加载 Qwen tokenizer（优先本地缓存，避免国内联网超时）。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
