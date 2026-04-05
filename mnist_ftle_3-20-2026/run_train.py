from __future__ import annotations

import argparse

from configs import PathConfig, TrainConfig
from paths import ensure_dirs
from train import train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    paths = PathConfig()
    ensure_dirs(paths)
    cfg = TrainConfig(
        width=args.width,
        depth=args.depth,
        gain=args.gain,
        base_lr=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
    )
    out = train_one(paths, cfg)
    print(out)


if __name__ == "__main__":
    main()
