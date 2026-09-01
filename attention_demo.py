"""ToyLLM 唯一入口：参数在此改；训练逻辑在 train.py。"""

from pathlib import Path

from train import TrainConfig, run

# ── Demo 参数（以后改动改这里即可）──────────────────────────────────────────────────

# 运行环境
DEVICE = "auto"  # "auto" | "cpu" | "cuda" | "xpu"
LOG_PATH = Path(__file__).resolve().parent / "log"

# 模型结构
DIM = 128
N_LAYERS = 16
SEED = 42

# 语料
CORPUS_N = 500  # 固定取 JSONL 前 N 篇；None = 全部有效篇

# 训练循环
BATCH_SIZE = 4
USE_CROP = False
TRAIN_STEPS = 10000
LR = 1e-3
LOG_EVERY = 500
CROP_MIN = 128
CROP_MAX = 512
DETAIL_STEP = -1  # 等于某 step 时打印逐层矩阵与梯度；-1=关闭

# Checkpoint
CKPT_PATH = Path("checkpoints/toyllm.pt")
SAVE_CKPT = True
LOAD_CKPT: Path | None = None  # 设路径则跳过训练，直接加载权重

# 训练后交互续写；仅加载已有权重推理：设 LOAD_CKPT  above
INTERACTIVE_AFTER = True
GEN_TOKENS = 32

# ── 以下无需改 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        TrainConfig(
            device=DEVICE,
            log_path=LOG_PATH,
            dim=DIM,
            n_layers=N_LAYERS,
            seed=SEED,
            corpus_n=CORPUS_N,
            batch_size=BATCH_SIZE,
            use_crop=USE_CROP,
            train_steps=TRAIN_STEPS,
            lr=LR,
            log_every=LOG_EVERY,
            crop_min=CROP_MIN,
            crop_max=CROP_MAX,
            detail_step=DETAIL_STEP,
            ckpt_path=CKPT_PATH,
            save_ckpt=SAVE_CKPT,
            load_ckpt=LOAD_CKPT,
            interactive_after=INTERACTIVE_AFTER,
            gen_tokens=GEN_TOKENS,
        )
    )
