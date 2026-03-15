
# phase6_timecourse_analysis.py
"""
Phase 6: Time-resolved geometry analysis for the rich↔lazy transition
---------------------------------------------------------------------

What this script does
- Optionally GENERATES missing training snapshots for selected configs/seeds by
  retraining from scratch with the snapshot-enabled training module.
- Loads snapshot checkpoints at chosen epochs.
- Computes, per snapshot and per seed:
    * m(t) = -rho( max-FTLE(x), adversarial margin(x) )
    * sat_frac(t)
    * anchor_frac(t): fraction of ridge points near the decision boundary
    * ridge_boundary_mean_dist(t)
    * align_mean(t): alignment of maximal stretching direction to boundary normal
                     on ridge points near the boundary
    * boundary_len(t), ridge_len(t)
    * RA_t: representational similarity to initialization (linear CKA)
- Caches all per-snapshot results.
- Aggregates across seeds and writes mean±std timecourses.
- Produces plots for each selected config, and summary plots by width/gain.

Minimal prerequisites
- Place this script in the same working folder as:
    ra_ka_best_method_accstop_snapshots.py
    circle_data_seed0.npz   (or let script generate it)
    rk_ckpts_v4/            (existing checkpoints okay)
- This script will write:
    rk_ckpt_snapshots/      (snapshot checkpoints)
    phase6_cache/           (per-snapshot metric caches)
    phase6_timecourse_state.npz
    plots_phase6_timecourse/

Notes
- m(t) uses a configurable subset of test points to keep cost manageable.
- Geometry is computed on a configurable grid (GEOM_GRID).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import math
import contextlib
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional SciPy for smoothing + distance transform
try:
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    gaussian_filter = None
    distance_transform_edt = None
    _HAS_SCIPY = False

from ra_ka_best_method_accstop_snapshots import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    snapshot_path,
    fmt_float,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
    DEVICE,
    SNAPSHOT_EPOCHS,
)

device = DEVICE

# -------------------- USER CONFIG --------------------
# Focused slice(s) for the timecourse study.
# Start small; this is intended as a targeted, high-evidence experiment.
FOCUS_WIDTHS = [30, 100, 250]
FOCUS_DEPTHS = [6]
FOCUS_GAINS = [0.20, 0.25, 0.30, 0.35, 0.40]  # dense around the transition
FOCUS_LRS = [0.20]
FOCUS_SEEDS = list(range(10))  # increase to 20/50 if you want stronger ensemble evidence

# Snapshot epochs to analyze (must match the epochs you ask training to save)
TIME_EPOCHS = [0, 10, 30, 100, 300, 1000, 2000, 4000]

# Snapshot generation behavior
GENERATE_MISSING_SNAPSHOTS = True
FORCE_RETRAIN_IF_MISSING = True   # retrain from scratch if any snapshot missing for a seed/config

# Dataset / caching
DATA_SEED = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"
CACHE_DIR = "phase6_cache"
STATE_FILE = "phase6_timecourse_state.npz"
PLOT_DIR = "plots_phase6_timecourse"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Geometry grid (lower than 161 keeps the timecourse manageable; increase to 161 if you want)
GEOM_GRID = 129
GEOM_BBOX = (-1.2, 1.2)

# Ridge extraction / anchoring parameters
RIDGE_Q = 0.95
RIDGE_GDOT_TOL = 0.15
SMOOTH_SIGMA = 1.0
MIN_RIDGE_PTS = 10
BOUNDARY_BAND_PX = 2

# m(t) settings
MARGIN_SUBSET = 512
MARGIN_SUBSET_SEED = 4242
EPS_HI = 0.30
PGD_STEPS = 20
BISECTION_ITERS = 10
MARGIN_BATCH = 512

# RA(t) probe subset (can be smaller than full test set)
RA_SUBSET = 1024
RA_SUBSET_SEED = 9999

# AMP
USE_AMP = (device.type == "cuda")
AMP_DTYPE = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

# Versioning
CACHE_VERSION = 1
STATE_VERSION = 1


def autocast_ctx():
    if USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return contextlib.nullcontext()


# -------------------- I/O helpers --------------------
def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


# -------------------- Dataset caching --------------------
def load_or_make_circle_data(cache_path: str, seed: int):
    if os.path.exists(cache_path):
        d = safe_load_npz(cache_path)
        if d is not None:
            xt = torch.tensor(d["xt"], dtype=torch.float32)
            yt = torch.tensor(d["yt"], dtype=torch.float32)
            xe = torch.tensor(d["xe"], dtype=torch.float32)
            ye = torch.tensor(d["ye"], dtype=torch.float32)
            return (xt, yt), (xe, ye)

    np.random.seed(seed)
    torch.manual_seed(seed)
    (xt, yt), (xe, ye) = make_circle(seed=seed)

    atomic_save_npz(
        cache_path,
        xt=xt.cpu().numpy().astype(np.float32),
        yt=yt.cpu().numpy().astype(np.float32),
        xe=xe.cpu().numpy().astype(np.float32),
        ye=ye.cpu().numpy().astype(np.float32),
    )
    return (xt, yt), (xe, ye)


# -------------------- Small stats helpers --------------------
def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xx = x[m].astype(np.float64, copy=False)
    yy = y[m].astype(np.float64, copy=False)
    rx = _rankdata_avg_ties(xx); ry = _rankdata_avg_ties(yy)
    rx -= rx.mean(); ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if den == 0.0:
        return float("nan")
    return float((rx * ry).sum() / den)


def sanitize_lambda(arr: np.ndarray) -> np.ndarray:
    lam = np.asarray(arr, dtype=np.float64)
    lam = np.array(lam, copy=True)
    lam[~np.isfinite(lam)] = np.nan
    return lam


def nanmean_sem(vals: np.ndarray) -> Tuple[float, float, int]:
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=0))
    sem = std / float(np.sqrt(n))
    return mean, sem, n


def linear_cka_features(H0: torch.Tensor, HT: torch.Tensor, eps: float = 1e-12) -> float:
    H0c = H0 - H0.mean(dim=0, keepdim=True)
    HTc = HT - HT.mean(dim=0, keepdim=True)
    A = H0c.T @ HTc
    B = H0c.T @ H0c
    C = HTc.T @ HTc
    num = (A * A).sum()
    den = torch.linalg.norm(B) * torch.linalg.norm(C) + eps
    return float((num / den).detach().cpu())


# -------------------- Snapshot management --------------------
def all_snapshots_exist(N: int, L: int, g: float, lr: float, seed: int, epochs: List[int]) -> bool:
    return all(os.path.exists(snapshot_path(N, L, g, lr, seed, ep)) for ep in epochs)


def ensure_snapshots_for_config(
    N: int, L: int, g: float, lr: float, seed: int, train_loader
) -> None:
    if all_snapshots_exist(N, L, g, lr, seed, TIME_EPOCHS):
        return
    if not GENERATE_MISSING_SNAPSHOTS:
        print(f"[warn] missing snapshots for N={N} L={L} g={g} lr={lr} seed={seed}")
        return
    if FORCE_RETRAIN_IF_MISSING:
        print(f"[snap] generating snapshots by retraining: N={N} L={L} g={g} lr={lr} seed={seed}")
        _ = verify_or_train_checkpoint(
            N, L, g, lr, seed,
            train_loader=train_loader,
            acc_target=TRAIN_ACC_TARGET,
            max_epochs=MAX_EPOCHS,
            fail_policy="none",
            make_snapshots=True,
            snapshot_epochs=TIME_EPOCHS,
            force_retrain=True,
        )
    else:
        # only train if no final checkpoint exists or invalid
        _ = verify_or_train_checkpoint(
            N, L, g, lr, seed,
            train_loader=train_loader,
            acc_target=TRAIN_ACC_TARGET,
            max_epochs=MAX_EPOCHS,
            fail_policy="none",
            make_snapshots=True,
            snapshot_epochs=TIME_EPOCHS,
            force_retrain=False,
        )


def load_snapshot_net(N: int, L: int, g: float, lr: float, seed: int, epoch: int) -> Optional[FC]:
    path = snapshot_path(N, L, g, lr, seed, epoch)
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    net = FC(N, L, gain=g).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


# -------------------- Geometry on a grid --------------------
try:
    from torch.func import jvp
except Exception as e:
    raise RuntimeError("phase6_timecourse_analysis.py requires torch.func.jvp (PyTorch >= 2.0).") from e


@torch.no_grad()
def ftle_theta_field_jvp(net: FC, depth: int, grid: int, bbox: Tuple[float, float], batch: int = 0):
    """
    Compute:
      lam[y,x] = (1/depth) log sigma_max(J_h)
      theta[y,x] = angle of principal right-singular direction in input plane
    where h = net(x, hid=True) is the feature map.
    """
    xs = torch.linspace(bbox[0], bbox[1], grid, device=device)
    ys = torch.linspace(bbox[0], bbox[1], grid, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def hidden(z):
        return net(z, hid=True)

    if batch is None or batch <= 0:
        batch = pts.shape[0]

    lam_out = torch.empty((pts.shape[0],), device=device, dtype=torch.float32)
    theta_out = torch.empty((pts.shape[0],), device=device, dtype=torch.float32)

    with torch.no_grad():
        for s in range(0, pts.shape[0], batch):
            xb = pts[s:s + batch]
            v1 = torch.zeros_like(xb); v1[:, 0] = 1.0
            v2 = torch.zeros_like(xb); v2[:, 1] = 1.0

            _, j1 = jvp(hidden, (xb,), (v1,))
            _, j2 = jvp(hidden, (xb,), (v2,))

            # M = J^T J = [[a,c],[c,b]]
            a = (j1 * j1).sum(dim=1)
            b = (j2 * j2).sum(dim=1)
            c = (j1 * j2).sum(dim=1)

            disc = torch.sqrt(torch.clamp((a - b) * (a - b) + 4.0 * c * c, min=0.0))
            eigmax = 0.5 * ((a + b) + disc)
            sigmax = torch.sqrt(torch.clamp(eigmax, min=0.0))
            lam = (1.0 / depth) * torch.log(sigmax + 1e-12)

            # principal right-singular direction angle
            theta = 0.5 * torch.atan2(2.0 * c, a - b)  # director angle in [-pi/2, pi/2]

            lam_out[s:s + xb.shape[0]] = lam.float()
            theta_out[s:s + xb.shape[0]] = theta.float()

    lam = lam_out.reshape(grid, grid).cpu().numpy().astype(np.float64)
    theta = theta_out.reshape(grid, grid).cpu().numpy().astype(np.float64)
    return lam, theta


@torch.no_grad()
def logit_field(net: FC, grid: int, bbox: Tuple[float, float], batch: int = 32768) -> np.ndarray:
    xs = torch.linspace(bbox[0], bbox[1], grid, device=device)
    ys = torch.linspace(bbox[0], bbox[1], grid, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    z = torch.empty((pts.shape[0],), device=device, dtype=torch.float32)
    for s in range(0, pts.shape[0], batch):
        xb = pts[s:s + batch]
        with autocast_ctx():
            zz = net(xb, grad=True).view(-1)
        z[s:s + xb.shape[0]] = zz.float()
    return z.reshape(grid, grid).cpu().numpy().astype(np.float64)


def grad_hess_2d(f: np.ndarray, dx: float):
    fy, fx = np.gradient(f, dx, dx, edge_order=2)
    fyy, fyx = np.gradient(fy, dx, dx, edge_order=2)
    fxy, fxx = np.gradient(fx, dx, dx, edge_order=2)
    fxy = 0.5 * (fxy + fyx)
    return fx, fy, fxx, fyy, fxy


def hessian_eigs_2x2(fxx: np.ndarray, fyy: np.ndarray, fxy: np.ndarray):
    tr = fxx + fyy
    disc = np.sqrt(np.maximum((fxx - fyy) ** 2 + 4.0 * (fxy ** 2), 0.0))
    eig_min = 0.5 * (tr - disc)
    eig_max = 0.5 * (tr + disc)
    return eig_min, eig_max


def eigvec_min_2x2(fxx: np.ndarray, fyy: np.ndarray, fxy: np.ndarray, eig_min: np.ndarray, eps: float = 1e-12):
    vx = fxy
    vy = eig_min - fxx
    n1 = np.sqrt(vx * vx + vy * vy)

    ax = eig_min - fyy
    ay = fxy
    use_alt = n1 < (eps * 10.0)
    vx = np.where(use_alt, ax, vx)
    vy = np.where(use_alt, ay, vy)

    n = np.sqrt(vx * vx + vy * vy) + eps
    vx /= n
    vy /= n
    return vx, vy


def boundary_mask_from_logits(z: np.ndarray) -> np.ndarray:
    pred = z > 0
    b = np.zeros_like(pred, dtype=bool)
    hc = pred[:, 1:] != pred[:, :-1]
    vc = pred[1:, :] != pred[:-1, :]
    b[:, 1:] |= hc
    b[:, :-1] |= hc
    b[1:, :] |= vc
    b[:-1, :] |= vc
    return b


def ridge_mask_from_ftle(lam: np.ndarray, dx: float):
    if SMOOTH_SIGMA > 0 and gaussian_filter is not None:
        f = gaussian_filter(lam, sigma=SMOOTH_SIGMA, mode="nearest")
    else:
        f = lam.copy()

    fx, fy, fxx, fyy, fxy = grad_hess_2d(f, dx)
    eig_min, eig_max = hessian_eigs_2x2(fxx, fyy, fxy)

    finite = np.isfinite(f)
    if finite.sum() < 10:
        return np.zeros_like(f, dtype=bool), f

    thr = float(np.nanquantile(f[finite], RIDGE_Q))
    high = finite & (f >= thr)

    vx, vy = eigvec_min_2x2(fxx, fyy, fxy, eig_min)
    gdot = fx * vx + fy * vy
    gradn = np.sqrt(fx * fx + fy * fy) + 1e-12

    ridge = high & np.isfinite(eig_min) & (eig_min < 0.0) & (np.abs(gdot) <= RIDGE_GDOT_TOL * gradn)
    if ridge.sum() < MIN_RIDGE_PTS:
        ridge = high & np.isfinite(eig_min) & (eig_min < 0.0)
    return ridge.astype(bool), f


def ridge_length(ridge: np.ndarray, dx: float) -> float:
    rid = ridge.astype(bool)
    Hh = np.logical_and(rid[:, 1:], rid[:, :-1]).sum()
    Hv = np.logical_and(rid[1:, :], rid[:-1, :]).sum()
    Hd1 = np.logical_and(rid[1:, 1:], rid[:-1, :-1]).sum()
    Hd2 = np.logical_and(rid[1:, :-1], rid[:-1, 1:]).sum()
    return float(dx * (Hh + Hv) + dx * np.sqrt(2.0) * (Hd1 + Hd2))


def boundary_length(boundary: np.ndarray, dx: float) -> float:
    b = boundary.astype(bool)
    e = int((b[:, 1:] != b[:, :-1]).sum() + (b[1:, :] != b[:-1, :]).sum())
    return float(e) * dx


def ridge_boundary_metrics(
    ridge: np.ndarray,
    z_field: np.ndarray,
    theta_field: np.ndarray,
    dx: float,
) -> Dict[str, float]:
    """
    Returns:
      anchor_frac
      ridge_boundary_mean_dist
      align_mean
      boundary_len
      ridge_len
    """
    boundary = boundary_mask_from_logits(z_field)

    if ridge.sum() == 0 or boundary.sum() == 0:
        return dict(
            anchor_frac=float("nan"),
            ridge_boundary_mean_dist=float("nan"),
            align_mean=float("nan"),
            boundary_len=boundary_length(boundary, dx),
            ridge_len=ridge_length(ridge, dx),
        )

    # distance to boundary (in pixels then physical units)
    if distance_transform_edt is not None:
        dist_pix = distance_transform_edt(~boundary)
    else:
        # fallback: very coarse Manhattan distance via repeated dilations would be more code;
        # if scipy missing, use a simple but slower exact computation.
        by, bx = np.where(boundary)
        ry, rx = np.where(np.ones_like(boundary, dtype=bool))
        # squared Euclidean distance to nearest boundary pixel
        dist_pix = np.empty_like(z_field, dtype=np.float64)
        for i in range(z_field.shape[0]):
            for j in range(z_field.shape[1]):
                d2 = (by - i) ** 2 + (bx - j) ** 2
                dist_pix[i, j] = np.sqrt(float(np.min(d2))) if d2.size else np.nan

    dist = dist_pix * dx
    ridge_dist = dist[ridge]
    ridge_boundary_mean_dist = float(np.nanmean(ridge_dist)) if ridge_dist.size else float("nan")
    anchor_frac = float(np.mean(ridge_dist <= (BOUNDARY_BAND_PX * dx))) if ridge_dist.size else float("nan")

    # boundary normal from logit gradient
    fx, fy, *_ = grad_hess_2d(z_field, dx)
    nrm = np.sqrt(fx * fx + fy * fy) + 1e-12
    nx = fx / nrm
    ny = fy / nrm

    # principal stretching line field
    vx = np.cos(theta_field)
    vy = np.sin(theta_field)

    # alignment only on ridge pixels near the boundary
    near = ridge & (dist_pix <= float(BOUNDARY_BAND_PX))
    align = np.abs(vx * nx + vy * ny)  # line-field sign removed
    align_vals = align[near & np.isfinite(align)]
    align_mean = float(np.nanmean(align_vals)) if align_vals.size else float("nan")

    return dict(
        anchor_frac=anchor_frac,
        ridge_boundary_mean_dist=ridge_boundary_mean_dist,
        align_mean=align_mean,
        boundary_len=boundary_length(boundary, dx),
        ridge_len=ridge_length(ridge, dx),
    )


# -------------------- m(t): FTLE vs adversarial margin --------------------
def pgd_batch(net: FC, X: torch.Tensor, y: torch.Tensor, eps: torch.Tensor, k: int = 20) -> torch.Tensor:
    B = X.shape[0]
    eps2 = eps.view(B, 1)
    delta = torch.zeros_like(X)

    for _ in range(k):
        delta.requires_grad_(True)
        with autocast_ctx():
            out = net(X + delta)
            loss = -(y.float() * out.float()).sum()
        grad = torch.autograd.grad(loss, delta, create_graph=False, retain_graph=False)[0]
        step = eps2 / 10.0
        delta = (delta + step * grad.sign()).detach()
        delta = torch.max(torch.min(delta, eps2), -eps2)
    return (X + delta).detach()


def margin_batch(net: FC, X: torch.Tensor, y: torch.Tensor,
                 eps_hi: float = 0.30, bisection_iters: int = 10, pgd_steps: int = 20) -> torch.Tensor:
    B = X.shape[0]
    lo = torch.zeros((B,), device=X.device, dtype=X.dtype)
    hi = torch.full((B,), eps_hi, device=X.device, dtype=X.dtype)

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd_batch(net, X, y, eps=mid, k=pgd_steps)
        with torch.no_grad():
            with autocast_ctx():
                pred = torch.sign(net(adv))
            success = (pred != y).view(-1)
        hi = torch.where(success, mid, hi)
        lo = torch.where(success, lo, mid)
    return hi


@torch.no_grad()
def ftle_points(net: FC, depth: int, X: torch.Tensor, batch: int = 0) -> np.ndarray:
    """
    Compute max-FTLE at arbitrary input points X [B,2], using JVP.
    """
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def hidden(z):
        return net(z, hid=True)

    if batch is None or batch <= 0:
        batch = X.shape[0]

    out = torch.empty((X.shape[0],), device=device, dtype=torch.float32)
    for s in range(0, X.shape[0], batch):
        xb = X[s:s + batch]
        v1 = torch.zeros_like(xb); v1[:, 0] = 1.0
        v2 = torch.zeros_like(xb); v2[:, 1] = 1.0

        _, j1 = jvp(hidden, (xb,), (v1,))
        _, j2 = jvp(hidden, (xb,), (v2,))

        a = (j1 * j1).sum(dim=1)
        b = (j2 * j2).sum(dim=1)
        c = (j1 * j2).sum(dim=1)

        disc = torch.sqrt(torch.clamp((a - b) * (a - b) + 4.0 * c * c, min=0.0))
        eigmax = 0.5 * ((a + b) + disc)
        sigmax = torch.sqrt(torch.clamp(eigmax, min=0.0))
        lam = (1.0 / depth) * torch.log(sigmax + 1e-12)
        out[s:s + xb.shape[0]] = lam.float()

    return out.cpu().numpy().astype(np.float64)


# -------------------- Per-snapshot cache --------------------
def time_cache_path(N: int, L: int, g: float, lr: float, seed: int, epoch: int) -> str:
    return os.path.join(
        CACHE_DIR,
        f"time_N{N}_L{L}_g{fmt_float(g)}_lr{fmt_float(lr)}_seed{seed}_ep{epoch}_grid{GEOM_GRID}_msub{MARGIN_SUBSET}.npz",
    )


def time_cache_ok(d: Optional[Dict[str, np.ndarray]]) -> bool:
    if d is None:
        return False
    if int(np.array(d.get("cache_version", 0)).item()) != CACHE_VERSION:
        return False
    return bool(np.array(d.get("finished", False)).item())


def compute_snapshot_metrics(
    N: int, L: int, g: float, lr: float, seed: int, epoch: int,
    net: FC,
    X_margin: torch.Tensor,
    y_margin: torch.Tensor,
    X_ra: torch.Tensor,
    H0_ra: torch.Tensor,
) -> Dict[str, np.ndarray]:
    path = time_cache_path(N, L, g, lr, seed, epoch)
    cached = safe_load_npz(path) if os.path.exists(path) else None
    if time_cache_ok(cached):
        return cached

    # Geometry on grid
    lam_grid, theta_grid = ftle_theta_field_jvp(net, depth=L, grid=GEOM_GRID, bbox=GEOM_BBOX)
    z_grid = logit_field(net, grid=GEOM_GRID, bbox=GEOM_BBOX)
    dx = float((GEOM_BBOX[1] - GEOM_BBOX[0]) / (GEOM_GRID - 1))
    ridge, _ = ridge_mask_from_ftle(lam_grid, dx=dx)
    geom = ridge_boundary_metrics(ridge, z_grid, theta_grid, dx=dx)

    # m(t) on a subset of test points
    lam_pts = ftle_points(net, depth=L, X=X_margin, batch=MARGIN_BATCH)
    margins = margin_batch(
        net, X_margin, y_margin,
        eps_hi=EPS_HI,
        bisection_iters=BISECTION_ITERS,
        pgd_steps=PGD_STEPS,
    ).float().cpu().numpy().astype(np.float64)

    rho = spearman_rho(lam_pts, margins)
    m = -rho
    sat_frac = float(np.mean(margins >= (EPS_HI - 1e-6))) if margins.size else float("nan")

    # RA(t): compare hidden features to initialization features
    with torch.inference_mode():
        HT = net(X_ra, hid=True)
    RA_t = linear_cka_features(H0_ra, HT)

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),

        N=np.array(N, np.int32),
        L=np.array(L, np.int32),
        gain=np.array(g, np.float32),
        base_lr=np.array(lr, np.float32),
        seed=np.array(seed, np.int32),
        epoch=np.array(epoch, np.int32),

        m=np.array(m, np.float64),
        rho=np.array(rho, np.float64),
        sat_frac=np.array(sat_frac, np.float64),
        RA_t=np.array(RA_t, np.float64),

        anchor_frac=np.array(geom["anchor_frac"], np.float64),
        ridge_boundary_mean_dist=np.array(geom["ridge_boundary_mean_dist"], np.float64),
        align_mean=np.array(geom["align_mean"], np.float64),
        boundary_len=np.array(geom["boundary_len"], np.float64),
        ridge_len=np.array(geom["ridge_len"], np.float64),
    )
    atomic_save_npz(path, **out)
    return out


# -------------------- State save/load --------------------
def save_state(path: str, summary_rows: List[dict]) -> None:
    # Save as a flat table-like npz
    if len(summary_rows) == 0:
        atomic_save_npz(path, state_version=np.array(STATE_VERSION, np.int32))
        return

    keys = sorted(summary_rows[0].keys())
    arrs = {k: np.array([row[k] for row in summary_rows]) for k in keys}
    arrs["state_version"] = np.array(STATE_VERSION, np.int32)
    atomic_save_npz(path, **arrs)


# -------------------- Plotting --------------------
def plot_timecurve(epochs, mean, sem, title, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    m = np.isfinite(mean)
    x = np.array(epochs, dtype=np.float64)[m]
    y = np.asarray(mean, dtype=np.float64)[m]
    e = np.asarray(sem, dtype=np.float64)[m]
    if y.size == 0:
        print(f"[skip] no finite points for {title}")
        return
    plt.plot(x, y, "-o")
    if np.any(np.isfinite(e)):
        plt.fill_between(x, y - e, y + e, alpha=0.2)
    plt.xscale("symlog", linthresh=1.0)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_config_panels(config_rows: List[dict], out_path_prefix: str):
    """
    config_rows = one (N,L,g,lr) aggregated over seeds, different epochs
    """
    epochs = sorted(set(int(r["epoch"]) for r in config_rows))
    def series(key):
        means, sems = [], []
        for ep in epochs:
            vals = np.array([r[key] for r in config_rows if int(r["epoch"]) == ep], dtype=np.float64)
            mu, se, _ = nanmean_sem(vals)
            means.append(mu); sems.append(se)
        return np.array(means), np.array(sems)

    m_mean, m_sem = series("m")
    a_mean, a_sem = series("anchor_frac")
    d_mean, d_sem = series("ridge_boundary_mean_dist")
    al_mean, al_sem = series("align_mean")
    b_mean, b_sem = series("boundary_len")
    ra_mean, ra_sem = series("RA_t")

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("m(t) = -rho(FTLE, margin)", m_mean, m_sem, "m"),
        ("anchor fraction", a_mean, a_sem, "anchor"),
        ("ridge-boundary mean distance", d_mean, d_sem, "dist"),
        ("alignment to boundary normal", al_mean, al_sem, "align"),
        ("decision-boundary length", b_mean, b_sem, "boundary len"),
        ("RA(t) to initialization", ra_mean, ra_sem, "RA_t"),
    ]
    for axi, (title, mean, sem, ylab) in zip(ax.ravel(), panels):
        m = np.isfinite(mean)
        x = np.array(epochs, dtype=np.float64)[m]
        y = np.asarray(mean)[m]
        e = np.asarray(sem)[m]
        if y.size:
            axi.plot(x, y, "-o")
            if np.any(np.isfinite(e)):
                axi.fill_between(x, y - e, y + e, alpha=0.2)
        axi.set_xscale("symlog", linthresh=1.0)
        axi.set_title(title)
        axi.set_xlabel("epoch")
        axi.set_ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_panel.png", dpi=220)
    plt.close(fig)

    # phase-plane style: anchor vs m, alignment vs m over time
    plt.figure(figsize=(6, 4))
    plt.plot(m_mean, a_mean, "-o")
    for x, y, ep in zip(m_mean, a_mean, epochs):
        if np.isfinite(x) and np.isfinite(y):
            plt.text(x, y, str(ep), fontsize=8)
    plt.xlabel("m(t)")
    plt.ylabel("anchor fraction")
    plt.title("Turning-on of boundary anchoring")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_anchor_vs_m.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(m_mean, al_mean, "-o")
    for x, y, ep in zip(m_mean, al_mean, epochs):
        if np.isfinite(x) and np.isfinite(y):
            plt.text(x, y, str(ep), fontsize=8)
    plt.xlabel("m(t)")
    plt.ylabel("alignment")
    plt.title("Turning-on of boundary-normal stretching alignment")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_align_vs_m.png", dpi=220)
    plt.close()


# -------------------- Main --------------------
def main():
    print("[device]", device)
    if device.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))

    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)

    # Fixed subsets for m(t) and RA(t)
    gen_m = torch.Generator(device="cpu").manual_seed(MARGIN_SUBSET_SEED)
    idx_m = torch.randperm(xe.shape[0], generator=gen_m)[:min(MARGIN_SUBSET, xe.shape[0])]
    X_margin = xe[idx_m].to(device)
    y_margin = ye[idx_m].to(device)

    gen_ra = torch.Generator(device="cpu").manual_seed(RA_SUBSET_SEED)
    idx_ra = torch.randperm(xe.shape[0], generator=gen_ra)[:min(RA_SUBSET, xe.shape[0])]
    X_ra = xe[idx_ra].to(device)

    # Precompute H0_ra per (N,L,g,seed) because it only depends on initialization
    init_H0_cache = {}

    summary_rows = []

    for N in FOCUS_WIDTHS:
        for L in FOCUS_DEPTHS:
            for lr in FOCUS_LRS:
                for g in FOCUS_GAINS:
                    print(f"\n[config] N={N} L={L} g={g} lr={lr}")

                    # ensure snapshots
                    for sd in FOCUS_SEEDS:
                        ensure_snapshots_for_config(N, L, g, lr, sd, train_loader=train_loader)

                    # per config/seed/epoch rows
                    config_rows = []

                    for sd in FOCUS_SEEDS:
                        # init features for RA(t)
                        key = (N, L, g, sd)
                        if key not in init_H0_cache:
                            torch.manual_seed(sd)
                            np.random.seed(sd)
                            # Use the same SEED_BASE+seed convention as the training module
                            torch.manual_seed(sd)
                            net0 = FC(N, L, gain=g).to(device)
                            net0.eval()
                            with torch.inference_mode():
                                init_H0_cache[key] = net0(X_ra, hid=True)
                            del net0

                        H0_ra = init_H0_cache[key]

                        for ep in TIME_EPOCHS:
                            net = load_snapshot_net(N, L, g, lr, sd, ep)
                            if net is None:
                                continue
                            out = compute_snapshot_metrics(
                                N, L, g, lr, sd, ep,
                                net=net,
                                X_margin=X_margin,
                                y_margin=y_margin,
                                X_ra=X_ra,
                                H0_ra=H0_ra,
                            )
                            row = {k: (float(np.array(v).item()) if np.array(v).shape == () else np.array(v))
                                   for k, v in out.items() if k not in ("cache_version", "finished")}
                            config_rows.append(row)
                            summary_rows.append(row)

                    # make per-config plots
                    if len(config_rows) > 0:
                        tag = f"N{N}_L{L}_g{fmt_float(g)}_lr{fmt_float(lr)}"
                        plot_config_panels(config_rows, os.path.join(PLOT_DIR, tag))

    save_state(STATE_FILE, summary_rows)
    print(f"[saved] {STATE_FILE}")
    print(f"[plots] saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
