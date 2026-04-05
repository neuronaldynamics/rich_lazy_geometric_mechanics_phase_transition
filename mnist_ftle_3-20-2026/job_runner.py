from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from analysis import spearman_rho
from configs import EvalConfig, PlotConfig, TrainConfig
from data import load_mnist_tensors
from ftle import compute_ftle_batch
from margin import multiclass_margin_bisection
from plotting import make_plots
from projection import project_to_2d
from runtime import (
    ensure_job_dirs,
    load_status,
    log_stage_header,
    record_failure,
    save_status,
    stage_logger,
    write_spec_once,
)
from train import load_job_model, train_job
from utils import DEVICE, atomic_save_npz, atomic_write_json


def _required_train_files(job_path: Path) -> list[Path]:
    return [
        job_path / "checkpoints" / "latest.pt",
        job_path / "checkpoints" / "best.pt",
        job_path / "artifacts" / "train_metrics.json",
    ]


def _required_eval_files(job_path: Path) -> list[Path]:
    return [
        job_path / "artifacts" / "ftle_margin_data.npz",
        job_path / "artifacts" / "eval_metrics.json",
    ]


def _required_plot_files(job_path: Path) -> list[Path]:
    return [
        job_path / "artifacts" / "plots" / "ftle_vs_error.png",
        job_path / "artifacts" / "plots" / "ftle_vs_margin.png",
        job_path / "artifacts" / "plots" / "projection_ftle.png",
    ]


def _files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _chunk_path(root: Path, start: int, stop: int) -> Path:
    return root / f"chunk_{start:05d}_{stop - 1:05d}.npz"


def _run_train_stage(job_path: Path, spec: Dict[str, Any], status: Dict[str, Any]) -> None:
    stage = "train"
    if status.get(stage, {}).get("state") == "done" and _files_exist(_required_train_files(job_path)):
        print(f"[batch] skip train for {spec['job_id']} (already done)")
        return

    logger = stage_logger(job_path, stage)
    log_stage_header(logger, spec)
    print(f"[batch] start train for {spec['job_id']}")
    status[stage] = {"state": "running"}
    save_status(job_path, status)

    try:
        train_cfg = TrainConfig(**spec["train"])
        metrics = train_job(job_path, train_cfg, logger=logger)
        status[stage] = {
            "state": "done",
            "last_epoch": metrics["last_epoch"],
            "best_test_acc": metrics["best_test_acc"],
            "checkpoint": metrics["best_checkpoint"],
            "metrics_file": "artifacts/train_metrics.json",
        }
        save_status(job_path, status)
        print(f"[batch] done train for {spec['job_id']} best_test_acc={metrics['best_test_acc']:.4f}")
    except Exception as exc:
        logger.exception("train stage failed")
        print(f"[batch] failed train for {spec['job_id']}: {exc}")
        record_failure(job_path, stage, status, exc)
        raise


def _run_eval_stage(job_path: Path, spec: Dict[str, Any], status: Dict[str, Any]) -> None:
    stage = "eval"
    if status.get(stage, {}).get("state") == "done" and _files_exist(_required_eval_files(job_path)):
        print(f"[batch] skip eval for {spec['job_id']} (already done)")
        return

    logger = stage_logger(job_path, stage)
    log_stage_header(logger, spec)
    print(f"[batch] start eval for {spec['job_id']}")
    eval_status = status.setdefault(stage, {})
    eval_status["state"] = "running"
    save_status(job_path, status)

    try:
        train_cfg = TrainConfig(**spec["train"])
        eval_cfg = EvalConfig(**spec["eval"])
        model, payload = load_job_model(job_path, train_cfg, checkpoint_name="best.pt")
        _, test_ds = load_mnist_tensors(normalize=False)

        x_all, y_all = test_ds.tensors
        if eval_cfg.eval_subset is not None:
            x_all = x_all[: eval_cfg.eval_subset]
            y_all = y_all[: eval_cfg.eval_subset]
        total = int(x_all.shape[0])

        artifacts_dir = job_path / "artifacts"
        preds_path = artifacts_dir / "predictions.npz"
        ftle_chunks_dir = artifacts_dir / "ftle_chunks"
        margin_chunks_dir = artifacts_dir / "margin_chunks"
        final_npz_path = artifacts_dir / "ftle_margin_data.npz"
        eval_metrics_path = artifacts_dir / "eval_metrics.json"

        if preds_path.exists():
            preds_data = np.load(preds_path)
            pred_np = preds_data["pred"]
            correct_np = preds_data["correct"]
            logit_margin = preds_data["logit_margin"]
        else:
            with torch.no_grad():
                logits = model(x_all.to(DEVICE))
                pred = logits.argmax(dim=1).cpu()
                correct = pred == y_all
                top2 = logits.topk(2, dim=1).values
                target_logits = logits[torch.arange(logits.shape[0], device=DEVICE), y_all.to(DEVICE)]
                logit_margin = (target_logits - top2[:, 1]).cpu().numpy()
            pred_np = pred.numpy()
            correct_np = correct.numpy()
            atomic_save_npz(
                preds_path,
                pred=pred_np,
                correct=correct_np,
                logit_margin=logit_margin,
                y=y_all.numpy(),
            )

        eval_status["predictions_done"] = True
        save_status(job_path, status)

        ftle_chunks = []
        ftle_processed = 0
        ftle_range = range(0, total, eval_cfg.ftle_batch_size)
        ftle_bar = tqdm(
            ftle_range,
            desc=f"{job_path.name} ftle",
            total=(total + eval_cfg.ftle_batch_size - 1) // eval_cfg.ftle_batch_size,
            leave=True,
        )
        for start in ftle_bar:
            stop = min(start + eval_cfg.ftle_batch_size, total)
            chunk_path = _chunk_path(ftle_chunks_dir, start, stop)
            ftle_chunks.append(chunk_path)
            if not chunk_path.exists():
                xb = x_all[start:stop]
                lam, sig = compute_ftle_batch(model, xb, depth=train_cfg.depth, exact=True)
                atomic_save_npz(chunk_path, start=np.array(start), stop=np.array(stop), ftle=lam, sigma_max=sig)
                logger.info("saved ftle chunk start=%d stop=%d", start, stop)
            ftle_processed = stop
            eval_status["ftle_processed"] = ftle_processed
            eval_status["ftle_total"] = total
            save_status(job_path, status)
            ftle_bar.set_postfix(processed=f"{ftle_processed}/{total}")
        ftle_bar.close()

        ftle_vals = np.concatenate([np.load(path)["ftle"] for path in ftle_chunks]) if ftle_chunks else np.empty(0)
        sigma_vals = np.concatenate([np.load(path)["sigma_max"] for path in ftle_chunks]) if ftle_chunks else np.empty(0)
        eval_status["ftle_done"] = True
        save_status(job_path, status)

        good_idx = np.where(correct_np)[0]
        margin_total = int(good_idx.shape[0])
        margin_processed = 0
        margin_chunks = []
        margin_range = range(0, margin_total, eval_cfg.margin_batch_size)
        margin_bar = tqdm(
            margin_range,
            desc=f"{job_path.name} margin",
            total=(margin_total + eval_cfg.margin_batch_size - 1) // eval_cfg.margin_batch_size if margin_total else 0,
            leave=True,
        )
        for start in margin_bar:
            stop = min(start + eval_cfg.margin_batch_size, margin_total)
            chunk_path = _chunk_path(margin_chunks_dir, start, stop)
            margin_chunks.append(chunk_path)
            if not chunk_path.exists():
                idx = good_idx[start:stop]
                xb = x_all[idx]
                yb = y_all[idx]
                margin_b, sat_b = multiclass_margin_bisection(
                    model=model,
                    x=xb,
                    y=yb,
                    eps_hi=eval_cfg.eps_hi,
                    pgd_steps=eval_cfg.pgd_steps,
                    bisection_iters=eval_cfg.bisection_iters,
                )
                atomic_save_npz(
                    chunk_path,
                    indices=idx,
                    margin=margin_b.numpy(),
                    saturated=sat_b.numpy(),
                )
                logger.info("saved margin chunk start=%d stop=%d", start, stop)
            margin_processed = stop
            eval_status["margin_processed"] = margin_processed
            eval_status["margin_total"] = margin_total
            save_status(job_path, status)
            margin_bar.set_postfix(processed=f"{margin_processed}/{margin_total}")
        margin_bar.close()

        margin_full = np.full(total, np.nan, dtype=np.float64)
        saturated_full = np.full(total, False, dtype=bool)
        for chunk_path in margin_chunks:
            chunk = np.load(chunk_path)
            idx = chunk["indices"]
            margin_full[idx] = chunk["margin"]
            saturated_full[idx] = chunk["saturated"]
        eval_status["margin_done"] = True
        save_status(job_path, status)

        proj = project_to_2d(x_all.numpy(), method=eval_cfg.projection_method)
        margin_vals = margin_full[good_idx]
        saturated_vals = saturated_full[good_idx]
        rho_all = spearman_rho(ftle_vals[good_idx], margin_vals) if margin_total else float("nan")
        unsat = ~saturated_vals if margin_total else np.array([], dtype=bool)
        rho_unsat = spearman_rho(ftle_vals[good_idx][unsat], margin_vals[unsat]) if unsat.sum() >= 3 else float("nan")

        atomic_save_npz(
            final_npz_path,
            x_proj=proj,
            y=y_all.numpy(),
            pred=pred_np,
            correct=correct_np,
            ftle=ftle_vals,
            sigma_max=sigma_vals,
            margin=margin_full,
            saturated=saturated_full,
            logit_margin=logit_margin,
            rho_all=np.array(rho_all),
            rho_unsat=np.array(rho_unsat),
            test_acc=np.array(payload.get("final_test_acc", np.nan)),
        )

        metrics = {
            "rho_all": rho_all,
            "rho_unsat": rho_unsat,
            "number_correct": int(correct_np.sum()),
            "number_eval_samples": total,
            "predictions_file": str(preds_path.relative_to(job_path)),
            "final_npz": str(final_npz_path.relative_to(job_path)),
            "ftle_chunks_dir": str(ftle_chunks_dir.relative_to(job_path)),
            "margin_chunks_dir": str(margin_chunks_dir.relative_to(job_path)),
            "train_checkpoint": "checkpoints/best.pt",
        }
        atomic_write_json(eval_metrics_path, metrics)
        status[stage] = {
            "state": "done",
            "predictions_done": True,
            "ftle_done": True,
            "margin_done": True,
            "ftle_processed": total,
            "ftle_total": total,
            "margin_processed": margin_total,
            "margin_total": margin_total,
            "rho_all": rho_all,
            "rho_unsat": rho_unsat,
            "output_file": str(final_npz_path.relative_to(job_path)),
            "metrics_file": str(eval_metrics_path.relative_to(job_path)),
        }
        save_status(job_path, status)
        print(
            f"[batch] done eval for {spec['job_id']} rho_all={rho_all:.4f} rho_unsat={rho_unsat:.4f}"
        )
    except Exception as exc:
        logger.exception("eval stage failed")
        print(f"[batch] failed eval for {spec['job_id']}: {exc}")
        record_failure(job_path, stage, status, exc)
        raise


def _run_plot_stage(job_path: Path, spec: Dict[str, Any], status: Dict[str, Any]) -> None:
    stage = "plots"
    plot_cfg = PlotConfig(**spec["plots"])
    if not plot_cfg.run:
        status[stage] = {"state": "done", "skipped": True}
        save_status(job_path, status)
        print(f"[batch] skip plots for {spec['job_id']} (disabled)")
        return
    if status.get(stage, {}).get("state") == "done" and _files_exist(_required_plot_files(job_path)):
        print(f"[batch] skip plots for {spec['job_id']} (already done)")
        return

    logger = stage_logger(job_path, stage)
    log_stage_header(logger, spec)
    print(f"[batch] start plots for {spec['job_id']}")
    status[stage] = {"state": "running"}
    save_status(job_path, status)

    try:
        outputs = make_plots(
            eval_npz_path=job_path / "artifacts" / "ftle_margin_data.npz",
            output_dir=job_path / "artifacts" / "plots",
            bins=plot_cfg.bins,
        )
        status[stage] = {
            "state": "done",
            "outputs": {k: str(Path(v).relative_to(job_path)) for k, v in outputs.items()},
        }
        save_status(job_path, status)
        print(f"[batch] done plots for {spec['job_id']}")
    except Exception as exc:
        logger.exception("plot stage failed")
        print(f"[batch] failed plots for {spec['job_id']}: {exc}")
        record_failure(job_path, stage, status, exc)
        raise


def run_job(job_path: Path, spec: Dict[str, Any], continue_on_error: bool = True) -> Dict[str, Any]:
    ensure_job_dirs(job_path)
    write_spec_once(job_path, spec)
    status = load_status(job_path, spec["job_id"])

    for runner in [_run_train_stage, _run_eval_stage, _run_plot_stage]:
        try:
            runner(job_path, spec, status)
            status = load_status(job_path, spec["job_id"])
        except Exception:
            if not continue_on_error:
                raise
            break
    return status
