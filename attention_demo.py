"""ToyLLM 唯一入口：参数在此改；总调度见 pipeline.py。"""

from pathlib import Path

from pipeline import run
from train import TrainConfig

# ── Demo 参数（以后改动改这里即可）──────────────────────────────────────────────────

# 只改这一行切换配置： "XPU_DEBUG" | "GPU_TRAIN"
PROFILE = "XPU_DEBUG"

LOG_DIR = Path(__file__).resolve().parent / "log"
CKPT_PATH = Path("checkpoints/toyllm.pt")
LOAD_CKPT: Path | None = None  # 设路径则跳过训练，直接加载权重
SEED = 42
INTERACTIVE_AFTER = True
GEN_TOKENS = 32

# 核显：小模型、少步、开 detail，方便看矩阵
XPU_DEBUG = dict(
    device="xpu",
    dim=32,
    n_layers=4,
    corpus_n=1,
    batch_size=1,
    use_crop=False,
    train_steps=2,
    lr=1e-3,
    log_every=1,
    crop_min=128,
    crop_max=512,
    detail_steps=[0],  # False=关 | True=每步 | [0,1]=指定 step
    save_ckpt=False,
)

# 大卡：正经训，关 detail
GPU_TRAIN = dict(
    device="cuda",
    dim=128,
    n_layers=16,
    corpus_n=500,
    batch_size=4,
    use_crop=False,
    train_steps=10000,
    lr=1e-3,
    log_every=500,
    crop_min=128,
    crop_max=512,
    detail_steps=False,
    save_ckpt=True,
)

_PROFILES = {"XPU_DEBUG": XPU_DEBUG, "GPU_TRAIN": GPU_TRAIN}
if PROFILE not in _PROFILES:
    raise ValueError(f"未知 PROFILE={PROFILE!r}，可选: {list(_PROFILES)}")
cfg = _PROFILES[PROFILE]

# ── 以下无需改 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        TrainConfig(
            device=cfg["device"],
            log_dir=LOG_DIR,
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            seed=SEED,
            corpus_n=cfg["corpus_n"],
            batch_size=cfg["batch_size"],
            use_crop=cfg["use_crop"],
            train_steps=cfg["train_steps"],
            lr=cfg["lr"],
            log_every=cfg["log_every"],
            crop_min=cfg["crop_min"],
            crop_max=cfg["crop_max"],
            detail_steps=cfg["detail_steps"],
            ckpt_path=CKPT_PATH,
            save_ckpt=cfg["save_ckpt"],
            load_ckpt=LOAD_CKPT,
            interactive_after=INTERACTIVE_AFTER,
            gen_tokens=GEN_TOKENS,
        )
    )
