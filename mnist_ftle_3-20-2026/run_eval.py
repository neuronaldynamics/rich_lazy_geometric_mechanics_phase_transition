from __future__ import annotations

import argparse

import numpy as np
import torch

from analysis import spearman_rho
from configs import EvalConfig, PathConfig, TrainConfig
from data import load_mnist_tensors
from ftle import compute_ftle_batch
from margin import multiclass_margin_bisection
from paths import ensure_dirs, eval_npz_path
from projection import project_to_2d
from train import load_trained_model
from utils import DEVICE, atomic_save_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=2000)
    args = ap.parse_args()

    paths = PathConfig()
    ensure_dirs(paths)
    train_cfg = TrainConfig(width=args.width, depth=args.depth, gain=args.gain, base_lr=args.lr, seed=args.seed)
    eval_cfg = EvalConfig(eval_subset=args.subset)

    model, payload = load_trained_model(paths, train_cfg)
    _, test_ds = load_mnist_tensors(normalize=False)

    x_all, y_all = test_ds.tensors
    if eval_cfg.eval_subset is not None:
        x_all = x_all[: eval_cfg.eval_subset]
        y_all = y_all[: eval_cfg.eval_subset]

    with torch.no_grad():
        logits = model(x_all.to(DEVICE))
        pred = logits.argmax(dim=1).cpu()
        correct = (pred == y_all)
        logit_margin = (logits[torch.arange(logits.shape[0]), y_all.to(DEVICE)] - logits.topk(2, dim=1).values[:, 1]).cpu().numpy()

    ftle_vals = []
    sigma_vals = []
    for start in range(0, x_all.shape[0], eval_cfg.ftle_batch_size):
        xb = x_all[start : start + eval_cfg.ftle_batch_size]
        lam, sig = compute_ftle_batch(model, xb, depth=train_cfg.depth, exact=True)
        ftle_vals.append(lam)
        sigma_vals.append(sig)
    ftle_vals = np.concatenate(ftle_vals)
    sigma_vals = np.concatenate(sigma_vals)

    good_idx = torch.where(correct)[0]
    x_good = x_all[good_idx]
    y_good = y_all[good_idx]

    margin_vals = []
    saturated_vals = []
    for start in range(0, x_good.shape[0], eval_cfg.margin_batch_size):
        xb = x_good[start : start + eval_cfg.margin_batch_size]
        yb = y_good[start : start + eval_cfg.margin_batch_size]
        margin_b, sat_b = multiclass_margin_bisection(
            model=model,
            x=xb,
            y=yb,
            eps_hi=eval_cfg.eps_hi,
            pgd_steps=eval_cfg.pgd_steps,
            bisection_iters=eval_cfg.bisection_iters,
        )
        margin_vals.append(margin_b.numpy())
        saturated_vals.append(sat_b.numpy())
    margin_vals = np.concatenate(margin_vals) if len(margin_vals) else np.empty(0)
    saturated_vals = np.concatenate(saturated_vals) if len(saturated_vals) else np.empty(0, dtype=bool)

    margin_full = np.full(x_all.shape[0], np.nan, dtype=np.float64)
    saturated_full = np.full(x_all.shape[0], False, dtype=bool)
    margin_full[good_idx.numpy()] = margin_vals
    saturated_full[good_idx.numpy()] = saturated_vals

    proj = project_to_2d(x_all.numpy(), method=eval_cfg.projection_method)

    rho_all = spearman_rho(ftle_vals[good_idx.numpy()], margin_vals) if len(margin_vals) else float("nan")
    unsat = ~saturated_vals if len(saturated_vals) else np.array([], dtype=bool)
    rho_unsat = spearman_rho(ftle_vals[good_idx.numpy()][unsat], margin_vals[unsat]) if unsat.sum() >= 3 else float("nan")

    save_path = eval_npz_path(paths, train_cfg)
    atomic_save_npz(
        save_path,
        x_proj=proj,
        y=y_all.numpy(),
        pred=pred.numpy(),
        correct=correct.numpy(),
        ftle=ftle_vals,
        sigma_max=sigma_vals,
        margin=margin_full,
        saturated=saturated_full,
        logit_margin=logit_margin,
        rho_all=np.array(rho_all),
        rho_unsat=np.array(rho_unsat),
        test_acc=np.array(payload["final_test_acc"]),
    )
    print({"saved": str(save_path), "rho_all": rho_all, "rho_unsat": rho_unsat})


if __name__ == "__main__":
    main()
