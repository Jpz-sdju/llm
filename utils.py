"""通用工具：设备选择、Qwen tokenizer 加载、训练 log 重定向、TinyHelen 语料。"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

import torch

# 未显式配置时默认走 hf-mirror；已有 HF_ENDPOINT 则不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

QWEN_TOKENIZER_ID = "Qwen/Qwen3-0.6B"
DEFAULT_LOG_PATH = Path("log")

TINYHELEN_REPO_ID = "fzmnm/TinyHelen-zh"
TINYHELEN_NEWS_FILE = "TinyNews-zh_000.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parent
TINYHELEN_DATA_DIR = PROJECT_ROOT / "data" / "TinyHelen-zh"
TINYHELEN_NEWS_PATH = TINYHELEN_DATA_DIR / TINYHELEN_NEWS_FILE


def redirect_stdout_to_log(log_path: Path | str = DEFAULT_LOG_PATH) -> tuple[TextIO, TextIO]:
    """训练阶段：stdout 只写 log 文件（行缓冲，便于实时看进度）。返回 (原 stdout, log 文件句柄)。"""
    path = Path(log_path)
    print(f"训练中，stdout → {path}（终端静默；进度见 log）", file=sys.__stdout__)
    # buffering=1：按行立刻落盘，避免「训练很久但 log 一直是空」
    log_fp = open(path, "w", encoding="utf-8", buffering=1)
    real_stdout = sys.stdout
    sys.stdout = log_fp  # type: ignore[assignment]
    return real_stdout, log_fp


def restore_stdout(real_stdout: TextIO, log_fp: TextIO, *terminal_lines: str) -> None:
    """恢复 stdout、关闭 log；可选在终端打印几行提示。"""
    try:
        log_fp.flush()
    except Exception:
        pass
    sys.stdout = real_stdout
    log_fp.close()
    for line in terminal_lines:
        print(line)


def _download_tinyhelen_news(local_dir: Path) -> None:
    """等价于 huggingface-cli download ... --include TinyNews-zh_000.jsonl。"""
    local_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args = [
        "download",
        TINYHELEN_REPO_ID,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
        "--include",
        TINYHELEN_NEWS_FILE,
    ]
    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli:
        subprocess.run([cli, *args], check=True, env=env)
        return
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=TINYHELEN_REPO_ID,
        repo_type="dataset",
        filename=TINYHELEN_NEWS_FILE,
        local_dir=str(local_dir),
    )


def ensure_tinyhelen_news(
    path: Path | None = None,
    *,
    data_dir: Path | None = None,
) -> Path:
    """本地无 TinyNews-zh_000.jsonl 时自动下载；有则直接返回路径。"""
    target = Path(path) if path else (data_dir or TINYHELEN_DATA_DIR) / TINYHELEN_NEWS_FILE
    if target.is_file() and target.stat().st_size > 0:
        return target
    print(f"未找到 {target}，正在下载 {TINYHELEN_REPO_ID}/{TINYHELEN_NEWS_FILE} …", file=sys.__stdout__)
    _download_tinyhelen_news(target.parent)
    if not target.is_file():
        raise FileNotFoundError(f"下载后仍不存在: {target}")
    return target


def load_tinyhelen_texts(
    path: Path | str | None = None,
    *,
    min_chars: int = 20,
) -> list[str]:
    """从 TinyNews JSONL 逐行解析，收集每条记录的 text 字段。"""
    jsonl_path = Path(path) if path is not None else ensure_tinyhelen_news()
    texts: list[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if len(text) < min_chars:
                continue
            texts.append(text)
    if not texts:
        raise ValueError(f"{jsonl_path} 中未读到有效 text（min_chars={min_chars}）")
    return texts


def random_crop_ids(
    ids: list[int],
    *,
    min_len: int = 32,
    max_len: int = 512,
    rng: random.Random | None = None,
) -> list[int]:
    """从 token id 序列中随机切一段，长度 ∈ [min_len, min(max_len, len(ids))]。"""
    if min_len > max_len:
        raise ValueError(f"min_len ({min_len}) > max_len ({max_len})")
    n = len(ids)
    if n <= min_len:
        return ids
    r = rng or random
    crop_len = r.randint(min_len, min(max_len, n))
    start = r.randint(0, n - crop_len)
    return ids[start : start + crop_len]


def random_crop_text(
    tokenizer,
    text: str,
    *,
    min_len: int = 32,
    max_len: int = 512,
    rng: random.Random | None = None,
) -> list[int]:
    """encode 后 random_crop_ids；训练时长文随机窗口。"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return random_crop_ids(ids, min_len=min_len, max_len=max_len, rng=rng)


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
