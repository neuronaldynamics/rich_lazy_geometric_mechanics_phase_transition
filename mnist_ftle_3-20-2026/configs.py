from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrainConfig:
    width: int = 20
    depth: int = 4
    gain: float = 1.0
    base_lr: float = 0.05
    batch_size: int = 8192
    max_epochs: int = 500
    target_test_acc: float = 0.97
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    eps_hi: float = 0.30
    pgd_steps: int = 20
    bisection_iters: int = 10
    eval_subset: int | None = 2000
    ftle_batch_size: int = 64
    margin_batch_size: int = 256
    projection_method: str = "pca"
    projection_points: int = 5000


@dataclass(frozen=True)
class PlotConfig:
    run: bool = True
    bins: int = 20


@dataclass(frozen=True)
class GridConfig:
    widths: Sequence[int] = (20, 50, 100)
    depths: Sequence[int] = (4, 8, 16)
    gains: Sequence[float] = (1.0,)
    base_lrs: Sequence[float] = (0.05,)
    seeds: Sequence[int] = (0, 1, 2)


@dataclass(frozen=True)
class PathConfig:
    root: Path = Path("runs")

    @property
    def ckpt_dir(self) -> Path:
        return self.root / "single" / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.root / "single" / "eval"

    @property
    def plot_dir(self) -> Path:
        return self.root / "single" / "plots"

    @property
    def proj_dir(self) -> Path:
        return self.root / "single" / "projections"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def jobs_root(self) -> Path:
        return self.root / "jobs"

    @property
    def summaries_dir(self) -> Path:
        return self.root / "summaries"
