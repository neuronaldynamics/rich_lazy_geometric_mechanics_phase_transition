from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from build_manifest import _expand_jobs

from configs import PathConfig
from paths import ensure_dirs
from runtime import read_json, read_yaml_or_json


def _row_from_job(job_path: Path) -> Dict[str, Any]:
    spec = read_json(job_path / "spec.json", {})
    status = read_json(job_path / "status.json", {})
    train_metrics = read_json(job_path / "artifacts" / "train_metrics.json", {})
    eval_metrics = read_json(job_path / "artifacts" / "eval_metrics.json", {})

    train_cfg = spec.get("train", {})
    return {
        "job_id": spec.get("job_id", job_path.name),
        "dataset": spec.get("dataset", "mnist"),
        "width": train_cfg.get("width"),
        "depth": train_cfg.get("depth"),
        "gain": train_cfg.get("gain"),
        "lr": train_cfg.get("base_lr"),
        "batch_size": train_cfg.get("batch_size"),
        "max_epochs": train_cfg.get("max_epochs"),
        "seed": train_cfg.get("seed"),
        "train_state": status.get("train", {}).get("state"),
        "eval_state": status.get("eval", {}).get("state"),
        "plots_state": status.get("plots", {}).get("state"),
        "final_train_acc": train_metrics.get("final_train_acc"),
        "final_test_acc": train_metrics.get("final_test_acc"),
        "best_test_acc": train_metrics.get("best_test_acc"),
        "rho_all": eval_metrics.get("rho_all"),
        "rho_unsat": eval_metrics.get("rho_unsat"),
        "number_correct": eval_metrics.get("number_correct"),
        "number_eval_samples": eval_metrics.get("number_eval_samples"),
        "updated_at": status.get("updated_at"),
    }


def collect_results(
    experiment_name: str,
    paths: PathConfig,
    dataset: str = "mnist",
    job_ids: Iterable[str] | None = None,
) -> Path:
    ensure_dirs(paths)
    jobs_root = paths.jobs_root / dataset
    rows = []
    job_id_set = set(job_ids) if job_ids is not None else None
    if jobs_root.exists():
        for job_path in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            spec = read_json(job_path / "spec.json", {})
            if job_id_set is not None:
                if spec.get("job_id", job_path.name) in job_id_set:
                    rows.append(_row_from_job(job_path))
            elif spec.get("experiment_name") == experiment_name or experiment_name in spec.get("experiment_names", []):
                rows.append(_row_from_job(job_path))

    out_dir = paths.summaries_dir / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"

    fieldnames = [
        "job_id",
        "dataset",
        "width",
        "depth",
        "gain",
        "lr",
        "batch_size",
        "max_epochs",
        "seed",
        "train_state",
        "eval_state",
        "plots_state",
        "final_train_acc",
        "final_test_acc",
        "best_test_acc",
        "rho_all",
        "rho_unsat",
        "number_correct",
        "number_eval_samples",
        "updated_at",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-name", type=str)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()

    if args.experiment_name is None and args.config is None:
        raise ValueError("Provide --experiment-name or --config.")

    experiment_name = args.experiment_name
    dataset = "mnist"
    if args.config is not None:
        config = read_yaml_or_json(args.config)
        experiment_name = config["experiment_name"]
        dataset = config.get("dataset", dataset)
        job_ids = [spec["job_id"] for spec in _expand_jobs(config)]
    else:
        job_ids = None
    assert experiment_name is not None

    out_dir = collect_results(
        experiment_name=experiment_name,
        paths=PathConfig(),
        dataset=dataset,
        job_ids=job_ids,
    )
    print({"summary_dir": str(out_dir)})


if __name__ == "__main__":
    main()
