"""总调度：装设备/tokenizer → 训练或加载 ckpt → 可选交互续写。"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model_input import (
    ToyForCausalLM,
    interactive_ask,
    load_checkpoint,
    save_checkpoint,
    texts_to_input_ids,
)
from tokenizer_setup import embedding_vocab_size, encode_split
from train import TrainConfig, run_train_loop
from utils import (
    QWEN_TOKENIZER_ID,
    ensure_tinyhelen_news,
    get_device,
    load_qwen_tokenizer,
    load_tinyhelen_texts,
    make_run_log_path,
    redirect_stdout_to_log,
    restore_stdout,
)


def _platform_label(device: torch.device) -> str:
    if device.type == "xpu":
        return "XPU"
    if device.type == "cuda":
        return "GPU"
    return device.type.upper()


def _print_run_header(cfg: TrainConfig, *, device: torch.device, log_file: Path) -> None:
    """把本次全部配置与运行平台写到 log 开头。"""
    print("=" * 75)
    print("本次运行参数")
    print("=" * 75)
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value!r}")
    print("-" * 75)
    print(f"  platform: {_platform_label(device)}")
    print(f"  python_platform: {platform.platform()}")
    print(f"  log_file: {log_file.resolve()}")
    print("=" * 75)
    print()


def _print_corpus_preview(
    tokenizer,
    corpus: list[str],
    *,
    device: torch.device,
) -> None:
    for i, text in enumerate(corpus):
        _, pieces = encode_split(tokenizer, text)
        print(f"预览第 {i + 1} 篇 pieces=", pieces[:12], "...")
        preview_ids, _ = texts_to_input_ids(tokenizer, text, device=device)
        print(f"[预览] tokens: {preview_ids.shape[1]}, shape: {tuple(preview_ids.shape)}\n")


def run(cfg: TrainConfig) -> ToyForCausalLM:
    """根据 cfg.load_ckpt 选择训练或只加载推理，最后可选交互续写。"""
    device = get_device(cfg.device)
    log_file = make_run_log_path(cfg.log_dir)
    real_stdout, log_fp = redirect_stdout_to_log(log_file)
    t_run0 = time.perf_counter()
    _print_run_header(cfg, device=device, log_file=log_file)

    tokenizer = load_qwen_tokenizer()
    vocab_size = embedding_vocab_size(tokenizer)
    print(f"tokenizer.vocab_size = {tokenizer.vocab_size}")
    print(f"embedding 行数       = {vocab_size}\n")

    if cfg.load_ckpt is not None:
        ckpt_path = Path(cfg.load_ckpt)
        print(f"跳过训练，加载 checkpoint: {ckpt_path.resolve()}")
        model, ckpt_cfg = load_checkpoint(ckpt_path, device=device)
        print(
            f"  dim={ckpt_cfg['dim']}, layers={ckpt_cfg['n_layers']}, "
            f"vocab={ckpt_cfg['vocab_size']}, tokenizer={ckpt_cfg['tokenizer_id']}"
        )
        if ckpt_cfg.get("train_steps") is not None:
            print(f"  训练 step 数（记录）: {ckpt_cfg['train_steps']}")
        elapsed = time.perf_counter() - t_run0
        restore_stdout(
            real_stdout,
            log_fp,
            f"已加载 checkpoint（{elapsed:.2f} s）|  日志: {log_file}",
            "下面进入交互问答\n" if cfg.interactive_after else "",
        )
    else:
        news_path = ensure_tinyhelen_news()
        all_corpus = load_tinyhelen_texts(news_path)
        corpus = all_corpus if cfg.corpus_n is None else all_corpus[: cfg.corpus_n]
        print(f"TinyNews 语料: {news_path}")
        if cfg.corpus_n is None:
            print(f"  训练篇数: {len(corpus)}  （全部有效篇）\n")
        else:
            print(f"  训练篇数: {len(corpus)}  （固定取 JSONL 前 {cfg.corpus_n} 篇）\n")
        _print_corpus_preview(tokenizer, corpus, device=device)

        torch.manual_seed(cfg.seed)
        model = ToyForCausalLM(vocab_size, dim=cfg.dim, n_layers=cfg.n_layers).to(device)
        run_train_loop(model, cfg=cfg, corpus=corpus, tokenizer=tokenizer, device=device)

        if cfg.save_ckpt:
            saved = save_checkpoint(
                cfg.ckpt_path,
                model,
                tokenizer_id=QWEN_TOKENIZER_ID,
                train_steps=cfg.train_steps,
            )
            print(f"已保存 checkpoint → {saved.resolve()}\n")

        elapsed = time.perf_counter() - t_run0
        restore_stdout(
            real_stdout,
            log_fp,
            f"训练完成: {elapsed:.2f} s  |  日志: {log_file}",
            "下面进入交互问答\n" if cfg.interactive_after else "",
        )

    if cfg.interactive_after:
        interactive_ask(model, tokenizer, device=device, max_new_tokens=cfg.gen_tokens)

    return model
