from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = rankdata(x[mask], method="average")
    ry = rankdata(y[mask], method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def binned_error_curve(ftle: np.ndarray, correct: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(ftle, np.linspace(0, 1, bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    errs = np.full(bins, np.nan)
    for i in range(bins):
        if i < bins - 1:
            mask = (ftle >= edges[i]) & (ftle < edges[i + 1])
        else:
            mask = (ftle >= edges[i]) & (ftle <= edges[i + 1])
        if mask.any():
            errs[i] = 1.0 - correct[mask].mean()
    return centers, errs
