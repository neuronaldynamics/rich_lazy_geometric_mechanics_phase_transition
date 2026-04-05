from __future__ import annotations

import argparse

from configs import PathConfig, TrainConfig
from paths import ensure_dirs, eval_npz_path, plot_prefix
from plotting import make_plots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = PathConfig()
    ensure_dirs(paths)
    cfg = TrainConfig(width=args.width, depth=args.depth, gain=args.gain, base_lr=args.lr, seed=args.seed)
    prefix = plot_prefix(paths, cfg)
    out_dir = prefix.parent / prefix.name
    outputs = make_plots(eval_npz_path(paths, cfg), out_dir, bins=20)

    print({"output_dir": str(out_dir), "outputs": outputs})


if __name__ == "__main__":
    main()
