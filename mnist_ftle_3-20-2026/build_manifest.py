from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable

from configs import EvalConfig, PathConfig, PlotConfig, TrainConfig
from paths import ensure_dirs, run_stem
from runtime import append_jsonl, read_yaml_or_json


def _listify(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _expand_jobs(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    experiment_name = config["experiment_name"]
    dataset = config.get("dataset", "mnist")
    sweep = config.get("sweep", {})

    widths = _listify(sweep.get("width"), [TrainConfig.width])
    depths = _listify(sweep.get("depth"), [TrainConfig.depth])
    gains = _listify(sweep.get("gain"), [TrainConfig.gain])
    lrs = _listify(sweep.get("lr"), [TrainConfig.base_lr])
    seeds = _listify(sweep.get("seed"), [TrainConfig.seed])

    train_defaults = TrainConfig()
    eval_defaults = EvalConfig()
    plot_defaults = PlotConfig()

    for width, depth, gain, lr, seed in product(widths, depths, gains, lrs, seeds):
        train_cfg = TrainConfig(
            width=int(width),
            depth=int(depth),
            gain=float(gain),
            base_lr=float(lr),
            seed=int(seed),
            batch_size=int(config.get("train", {}).get("batch_size", train_defaults.batch_size)),
            max_epochs=int(config.get("train", {}).get("max_epochs", train_defaults.max_epochs)),
            target_test_acc=float(config.get("train", {}).get("target_test_acc", train_defaults.target_test_acc)),
        )
        eval_cfg = EvalConfig(
            eps_hi=float(config.get("eval", {}).get("eps_hi", eval_defaults.eps_hi)),
            pgd_steps=int(config.get("eval", {}).get("pgd_steps", eval_defaults.pgd_steps)),
            bisection_iters=int(config.get("eval", {}).get("bisection_iters", eval_defaults.bisection_iters)),
            eval_subset=config.get("eval", {}).get("eval_subset", eval_defaults.eval_subset),
            ftle_batch_size=int(config.get("eval", {}).get("ftle_batch_size", eval_defaults.ftle_batch_size)),
            margin_batch_size=int(config.get("eval", {}).get("margin_batch_size", eval_defaults.margin_batch_size)),
            projection_method=str(config.get("eval", {}).get("projection_method", eval_defaults.projection_method)),
            projection_points=int(config.get("eval", {}).get("projection_points", eval_defaults.projection_points)),
        )
        plot_cfg = PlotConfig(
            run=bool(config.get("plots", {}).get("run", plot_defaults.run)),
            bins=int(config.get("plots", {}).get("bins", plot_defaults.bins)),
        )
        runtime_cfg = {
            "skip_finished": bool(config.get("runtime", {}).get("skip_finished", True)),
            "continue_on_error": bool(config.get("runtime", {}).get("continue_on_error", True)),
        }
        yield {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "job_id": run_stem(train_cfg),
            "train": asdict_clean(train_cfg),
            "eval": asdict_clean(eval_cfg),
            "plots": asdict_clean(plot_cfg),
            "runtime": runtime_cfg,
        }


def asdict_clean(obj) -> Dict[str, Any]:
    return {k: v for k, v in obj.__dict__.items()}


def build_manifest(config_path: Path, paths: PathConfig) -> Path:
    ensure_dirs(paths)
    config = read_yaml_or_json(config_path)
    manifest_path = paths.manifests_dir / f"{config['experiment_name']}_manifest.jsonl"
    append_jsonl(manifest_path, _expand_jobs(config))
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    manifest_path = build_manifest(args.config, PathConfig())
    print({"manifest": str(manifest_path)})


if __name__ == "__main__":
    main()
