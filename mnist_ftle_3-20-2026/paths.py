from __future__ import annotations

from pathlib import Path

from configs import PathConfig, TrainConfig


def ensure_dirs(paths: PathConfig) -> None:
    for d in [
        paths.root,
        paths.ckpt_dir,
        paths.eval_dir,
        paths.plot_dir,
        paths.proj_dir,
        paths.manifests_dir,
        paths.jobs_root,
        paths.summaries_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def fmt_float(x: float) -> str:
    s = f"{x:.6g}".replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def run_stem(cfg: TrainConfig) -> str:
    return (
        f"mnist_w{cfg.width}_d{cfg.depth}_g{fmt_float(cfg.gain)}_"
        f"lr{fmt_float(cfg.base_lr)}_bs{cfg.batch_size}_ep{cfg.max_epochs}_seed{cfg.seed}"
    )


def ckpt_path(paths: PathConfig, cfg: TrainConfig) -> Path:
    return paths.ckpt_dir / f"{run_stem(cfg)}.pt"


def eval_npz_path(paths: PathConfig, cfg: TrainConfig) -> Path:
    return paths.eval_dir / f"{run_stem(cfg)}.npz"


def plot_prefix(paths: PathConfig, cfg: TrainConfig) -> Path:
    return paths.plot_dir / run_stem(cfg)


def projection_path(paths: PathConfig, cfg: TrainConfig) -> Path:
    return paths.proj_dir / f"{run_stem(cfg)}.npz"


def job_dir(paths: PathConfig, dataset: str, job_id: str) -> Path:
    return paths.jobs_root / dataset / job_id
