from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import binned_error_curve


def make_plots(eval_npz_path: Path, output_dir: Path, bins: int = 20) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(eval_npz_path)

    ftle = data["ftle"]
    correct = data["correct"].astype(float)
    margin = data["margin"]
    proj = data["x_proj"]

    centers, errs = binned_error_curve(ftle, correct, bins=bins)
    ftle_error_path = output_dir / "ftle_vs_error.png"
    plt.figure(figsize=(6, 4))
    plt.plot(centers, errs, marker="o")
    plt.xlabel("max FTLE")
    plt.ylabel("binned test error")
    plt.tight_layout()
    plt.savefig(ftle_error_path, dpi=180)
    plt.close()

    mask = np.isfinite(margin)
    ftle_margin_path = output_dir / "ftle_vs_margin.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(ftle[mask], margin[mask], s=6)
    plt.xlabel("max FTLE")
    plt.ylabel("adversarial margin")
    plt.tight_layout()
    plt.savefig(ftle_margin_path, dpi=180)
    plt.close()

    proj_path = output_dir / "projection_ftle.png"
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], c=ftle, s=6)
    plt.xlabel("proj-1")
    plt.ylabel("proj-2")
    plt.colorbar(label="max FTLE")
    plt.tight_layout()
    plt.savefig(proj_path, dpi=180)
    plt.close()

    return {
        "ftle_vs_error": str(ftle_error_path),
        "ftle_vs_margin": str(ftle_margin_path),
        "projection_ftle": str(proj_path),
    }
