import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import contextlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from ra_ka_best_method_accstop import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    ckpt_path,
    DEVICE,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
    fmt_float,
)

# -------------------- GPU knobs --------------------
device = DEVICE
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# -------------------- USER CONFIG --------------------
WIDTHS   = [10, 20, 30, 50, 100, 150, 200, 250]
DEPTHS   = [2, 4, 6, 8, 10, 12, 14, 16]
GAINS    = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
BASE_LRS = [0.05, 0.075, 0.10, 0.20, 0.30, 0.40]
SEEDS    = [0, 1, 2, 3, 4, 5]

# Attack / evaluation
EPS_HI          = 0.30
PGD_STEPS       = 20
BISECTION_ITERS = 10

# FTLE grid sampling
FTLE_GRID = 161
BBOX      = (-1.2, 1.2)

# Resume / caching
FTLE_DIR    = "ftle"
CACHE_DIR   = "phase2_cache"
GRID_STATE  = "phase2_grid_state.npz"
PLOT_DIR    = "plots"

# Data caching (IMPORTANT for resume consistency)
DATA_SEED       = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

# Save partial seed progress every N points (power-outage safe)
SAVE_EVERY_POINTS = 200

# Control behavior
DO_COMPUTE = True
DO_PLOT    = True

# -------------------- SPEED KNOBS --------------------
MARGIN_BATCH = 8192
USE_AMP      = True
AMP_DTYPE    = torch.bfloat16
USE_COMPILE  = False          # optional torch.compile
LOSS_FP32    = True           # compute PGD loss in FP32 (slightly more stable)
CUDA_EMPTY_CACHE_EACH_SEED = False  # set True only if you get OOM/fragmentation

# -------------------- VERSIONING --------------------
# Keep your existing values so you don't invalidate old caches/state:
CACHE_VERSION = 2
GRID_VERSION  = 3

# -------------------- NUMERICAL SAFETY --------------------
FTLE_ABS_MAX_OK = 100.0
LOG_EXP_CLIP    = 700.0

# -------------------- torch.func FTLE --------------------
try:
    from torch.func import jvp
except Exception as e:
    raise RuntimeError("This script needs torch.func.jvp (PyTorch >= 2.0).") from e


def autocast_ctx():
    if USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return contextlib.nullcontext()


def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


def safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


def sanitize_lambda(arr: np.ndarray) -> np.ndarray:
    lam = np.asarray(arr, dtype=np.float64)
    lam = np.array(lam, copy=True)
    lam[~np.isfinite(lam)] = np.nan
    return lam


def spearman_rho_only(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = rankdata(x[m], method="average")
    ry = rankdata(y[m], method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


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
    random.seed(seed)
    torch.manual_seed(seed)

    (xt, yt), (xe, ye) = make_circle()

    atomic_save_npz(
        cache_path,
        xt=xt.cpu().numpy().astype(np.float32),
        yt=yt.cpu().numpy().astype(np.float32),
        xe=xe.cpu().numpy().astype(np.float32),
        ye=ye.cpu().numpy().astype(np.float32),
    )
    return (xt, yt), (xe, ye)


# -------------------- Checkpoint loading --------------------
def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader) -> Optional[FC]:
    # Always verify/train through the phase-1 helper, and SKIP on failure.
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS,
        fail_policy="none",   # <---- IMPORTANT
    )
    if net is None:
        print(f"[skip-model] N={N} L={L} g={gain} lr={base_lr} seed={seed} failed to reach acc_target={TRAIN_ACC_TARGET:.3f}")
        return None
    net.eval()
    return net


# -------------------- FTLE grid (cached) --------------------
def ftle_grid_path(N: int, L: int, gain: float, base_lr: float, seed: int, grid: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(FTLE_DIR, f"ftle_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_g{grid}.npy")


def ftle_grid_looks_corrupt(arr: np.ndarray, grid: int) -> bool:
    if arr is None or arr.shape != (grid, grid):
        return True
    if np.isinf(arr).any():
        return True
    m = np.nanmax(np.abs(arr))
    return bool(np.isfinite(m) and m > 1e3)


def ftle_field_jvp(net: FC, depth: int, grid: int, bbox: Tuple[float, float], batch: int = 0) -> np.ndarray:
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

    out = torch.empty((pts.shape[0],), device=device, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, pts.shape[0], batch):
            xb = pts[s:s + batch]
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
            lam = torch.where(torch.isfinite(lam), lam, torch.full_like(lam, float("nan")))
            out[s:s + xb.shape[0]] = lam.float()

    return out.reshape(grid, grid).cpu().numpy().astype(np.float32)


def load_ftle_grid(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader, grid: int) -> np.ndarray:
    path = ftle_grid_path(N, L, gain, base_lr, seed, grid)

    if os.path.exists(path):
        arr = safe_load_npy(path)
        if arr is not None and (not ftle_grid_looks_corrupt(arr, grid)):
            return arr
        print(f"[recompute-ftle] {os.path.basename(path)} corrupt -> recomputing")
        try:
            os.remove(path)
        except OSError:
            pass

    net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    fld = ftle_field_jvp(net, depth=L, grid=grid, bbox=BBOX)
    atomic_save_npy(path, fld)
    return fld


def precompute_ftle_indices(X_cpu: np.ndarray, grid: int, bbox: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    gx = grid - 1
    gy = grid - 1
    xmin, xmax = bbox
    ii = ((X_cpu[:, 0] - xmin) / (xmax - xmin) * gx).astype(np.int64)
    jj = ((X_cpu[:, 1] - xmin) / (xmax - xmin) * gy).astype(np.int64)
    np.clip(ii, 0, gx, out=ii)
    np.clip(jj, 0, gy, out=jj)
    return ii, jj


# -------------------- Batched PGD + margin --------------------
def pgd_batch(net: FC, X: torch.Tensor, y: torch.Tensor, eps: torch.Tensor, k: int) -> torch.Tensor:
    """
    X: [B,2], y: [B,1], eps: [B]
    """
    B = X.shape[0]
    eps2 = eps.view(B, 1)
    step = eps2 / 10.0
    delta = torch.zeros_like(X)

    for _ in range(k):
        delta.requires_grad_(True)

        with autocast_ctx():
            out = net(X + delta)
            if LOSS_FP32:
                loss = -(y.float() * out.float()).sum()
            else:
                loss = -(y * out).sum()

        grad = torch.autograd.grad(loss, delta, create_graph=False, retain_graph=False)[0]

        with torch.no_grad():
            delta.add_(step * grad.sign())
            delta.clamp_(-eps2, eps2)

        delta = delta.detach()

    return (X + delta).detach()


def margin_batch(net: FC, X: torch.Tensor, y: torch.Tensor,
                 eps_hi: float, bisection_iters: int, pgd_steps: int) -> torch.Tensor:
    """
    Returns eps*: [B]
    Decision rule matches old code: success if torch.sign(net(adv)) != y (keep 0 as 0).
    """
    B = X.shape[0]
    lo = torch.zeros((B,), device=X.device, dtype=X.dtype)
    hi = torch.full((B,), eps_hi, device=X.device, dtype=X.dtype)

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd_batch(net, X, y, eps=mid, k=pgd_steps)

        with torch.inference_mode():
            with autocast_ctx():
                pred = torch.sign(net(adv))
            success = (pred != y).view(-1)

        hi = torch.where(success, mid, hi)
        lo = torch.where(success, lo, mid)

    return hi


# -------------------- Per-seed caching --------------------
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedstats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_grid{FTLE_GRID}_dseed{DATA_SEED}.npz",
    )


def cache_version_of(d: Optional[Dict[str, np.ndarray]]) -> int:
    if d is None or "cache_version" not in d:
        return 0
    return int(np.array(d["cache_version"]).item())


def is_finished_seed_cache(d: Optional[Dict[str, np.ndarray]], n_test: int) -> bool:
    if d is None:
        return False
    if not bool(np.array(d.get("finished", False)).item()):
        return False
    if cache_version_of(d) != CACHE_VERSION:
        return False
    if "margins" not in d or d["margins"].shape[0] != n_test:
        return False
    if "ftle_vals" not in d or d["ftle_vals"].shape[0] != n_test:
        return False
    return True


def compute_or_resume_seed_stats(
    N: int, L: int, gain: float, base_lr: float, seed: int,
    train_loader,
    X_test: torch.Tensor, y_test: torch.Tensor,
    ftle_ii: np.ndarray, ftle_jj: np.ndarray,
) -> Dict[str, np.ndarray]:
    path = seed_cache_path(N, L, gain, base_lr, seed)
    n_test = X_test.shape[0]

    # HARD GATE: if training failed to reach target, skip everything for this seed/config.
    net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    if net is None:
        margins   = np.full(n_test, np.nan, dtype=np.float32)
        ftle_vals = np.full(n_test, np.nan, dtype=np.float32)
        out = dict(
            cache_version=np.array(CACHE_VERSION, dtype=np.int32),
            finished=np.array(True),
            train_ok=np.array(False),
            N=np.array(N), L=np.array(L),
            gain=np.array(gain, dtype=np.float32),
            base_lr=np.array(base_lr, dtype=np.float32),
            seed=np.array(seed, dtype=np.int32),
            n_test=np.array(n_test, dtype=np.int32),
            margins=margins,
            ftle_vals=ftle_vals,
            G_lambda=np.array(np.nan, dtype=np.float64),
            G_J=np.array(np.nan, dtype=np.float64),
            rho_lambda_margin=np.array(np.nan, dtype=np.float64),
            rho_J_margin=np.array(np.nan, dtype=np.float64),
        )
        atomic_save_npz(path, **out)
        return out

    cached = safe_load_npz(path) if os.path.exists(path) else None
    if is_finished_seed_cache(cached, n_test):
        return cached

    # margins: reuse if present (even if cache_version mismatch)
    if cached is not None and "margins" in cached and cached["margins"].shape[0] == n_test:
        margins = cached["margins"].astype(np.float32, copy=True)
    else:
        margins = np.full(n_test, np.nan, dtype=np.float32)

    # FTLE: recompute from grid (cheap) to avoid older-cache garbage
    fld = load_ftle_grid(N, L, gain, base_lr, seed, train_loader=train_loader, grid=FTLE_GRID)
    ftle_vals = fld[ftle_jj, ftle_ii].astype(np.float32, copy=False)

    absmax = np.nanmax(np.abs(ftle_vals))
    if np.isfinite(absmax) and absmax > FTLE_ABS_MAX_OK:
        print(f"[warn] |λ|max={absmax:.2e} too large for N={N} L={L} g={gain} lr={base_lr} seed={seed} -> recomputing FTLE")
        try:
            os.remove(ftle_grid_path(N, L, gain, base_lr, seed, FTLE_GRID))
        except OSError:
            pass
        fld = load_ftle_grid(N, L, gain, base_lr, seed, train_loader=train_loader, grid=FTLE_GRID)
        ftle_vals = fld[ftle_jj, ftle_ii].astype(np.float32, copy=False)

    todo = np.where(~np.isfinite(margins))[0]

    if todo.size:
        net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
        net.eval()

        if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
            net = torch.compile(net, mode="reduce-overhead")

        for p in net.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        done = 0
        last_save = 0

        for start in range(0, todo.size, MARGIN_BATCH):
            idx_np = todo[start:start + MARGIN_BATCH]
            idx_t = torch.as_tensor(idx_np, device=X_test.device, dtype=torch.long)

            eps_star = margin_batch(
                net,
                X_test[idx_t],
                y_test[idx_t],
                eps_hi=EPS_HI,
                bisection_iters=BISECTION_ITERS,
                pgd_steps=PGD_STEPS,
            )

            margins[idx_np] = eps_star.float().cpu().numpy()
            done += idx_np.size

            if (done - last_save) >= SAVE_EVERY_POINTS or (done == todo.size):
                last_save = done
                atomic_save_npz(
                    path,
                    cache_version=np.array(CACHE_VERSION, np.int32),
                    finished=np.array(False),
                    N=np.array(N, np.int32), L=np.array(L, np.int32),
                    gain=np.array(gain, np.float32),
                    base_lr=np.array(base_lr, np.float32),
                    seed=np.array(seed, np.int32),
                    n_test=np.array(n_test, np.int32),
                    margins=margins,
                    ftle_vals=ftle_vals,
                )
                dt = (time.time() - t0) / 60.0
                print(f"[save-partial] {os.path.basename(path)}  done={int(np.isfinite(margins).sum())}/{n_test}  dt={dt:.1f} min")

        del net
        if CUDA_EMPTY_CACHE_EACH_SEED and device.type == "cuda":
            torch.cuda.empty_cache()

    # seed stats
    lam = sanitize_lambda(ftle_vals)
    G_lambda = float(np.nanvar(lam))
    jac_norms = np.exp(np.clip(L * lam, -LOG_EXP_CLIP, LOG_EXP_CLIP))
    G_J = float(np.nanvar(jac_norms))

    rho_lambda_seed = spearman_rho_only(lam, margins.astype(np.float64))
    rho_J_seed = spearman_rho_only(jac_norms, margins.astype(np.float64))

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),
        N=np.array(N, np.int32), L=np.array(L, np.int32),
        gain=np.array(gain, np.float32),
        base_lr=np.array(base_lr, np.float32),
        seed=np.array(seed, np.int32),
        n_test=np.array(n_test, np.int32),
        margins=margins.astype(np.float32),
        ftle_vals=ftle_vals.astype(np.float32),
        G_lambda=np.array(G_lambda, np.float64),
        G_J=np.array(G_J, np.float64),
        rho_lambda_margin=np.array(rho_lambda_seed, np.float64),
        rho_J_margin=np.array(rho_J_seed, np.float64),
    )
    atomic_save_npz(path, **out)
    return out


# -------------------- Aggregation over seeds (pooled Spearman) --------------------
def aggregate_config_pooled(seed_dicts: List[Dict[str, np.ndarray]], L: int) -> Dict[str, float]:
    """
    Pooled Spearman across all samples of the *successful* seeds only.
    Also computes sat_frac correctly as fraction among finite margins.
    """
    # 1) keep only successful seeds
    good = []
    for d in seed_dicts:
        ok = bool(np.array(d.get("train_ok", True)).item())
        if ok:
            good.append(d)

    if len(good) == 0:
        return dict(
            G_lambda_mean=float("nan"),
            G_J_mean=float("nan"),
            rho_lambda_mean=float("nan"),
            rho_J_mean=float("nan"),
            sat_frac=float("nan"),
            rho_lambda_unsat=float("nan"),
            n_good=0,
        )

    # means of per-seed variances (only successful seeds)
    G_lambda_mean = float(np.nanmean([float(d["G_lambda"]) for d in good]))
    G_J_mean      = float(np.nanmean([float(d["G_J"]) for d in good]))

    # 2) pool samples across successful seeds
    lam_all = np.concatenate([sanitize_lambda(d["ftle_vals"]) for d in good], axis=0)
    m_all   = np.concatenate([d["margins"].astype(np.float64, copy=False) for d in good], axis=0)

    rho_lambda = spearman_rho_only(lam_all, m_all)

    jac_all = np.exp(np.clip(L * lam_all, -LOG_EXP_CLIP, LOG_EXP_CLIP))
    rho_J   = spearman_rho_only(jac_all, m_all)

    # 3) sat_frac computed among FINITE margins only (fix!)
    finite_m = np.isfinite(m_all)
    if finite_m.sum() == 0:
        sat_frac = float("nan")
        rho_lambda_unsat = float("nan")
    else:
        sat_mask = finite_m & (m_all >= (EPS_HI - 1e-6))
        sat_frac = float(sat_mask.sum() / finite_m.sum())

        unsat_mask = finite_m & (m_all < (EPS_HI - 1e-6))
        rho_lambda_unsat = (
            spearman_rho_only(lam_all[unsat_mask], m_all[unsat_mask])
            if unsat_mask.sum() >= 3 else float("nan")
        )

    return dict(
        G_lambda_mean=G_lambda_mean,
        G_J_mean=G_J_mean,
        rho_lambda_mean=rho_lambda,
        rho_J_mean=rho_J,
        sat_frac=sat_frac,
        rho_lambda_unsat=rho_lambda_unsat,
        n_good=len(good),
    )


# -------------------- Grid state --------------------
def save_grid_state(path: str,
                    widths, depths, gains, base_lrs, seeds,
                    G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, np.int32),
        data_seed=np.array(DATA_SEED, np.int32),
        widths=np.array(widths, np.int32),
        depths=np.array(depths, np.int32),
        gains=np.array(gains, np.float32),
        base_lrs=np.array(base_lrs, np.float32),
        seeds=np.array(seeds, np.int32),
        G_lambda_map=G_lambda_map.astype(np.float64),
        G_J_map=G_J_map.astype(np.float64),
        rho_lambda_map=rho_lambda_map.astype(np.float64),
        rho_J_map=rho_J_map.astype(np.float64),
        done_map=done_map.astype(np.bool_),
    )


def try_load_grid_state(path: str, widths, depths, gains, base_lrs, seeds):
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None
    if int(np.array(d.get("grid_version", 0)).item()) != GRID_VERSION:
        return None
    if int(np.array(d.get("data_seed", -1)).item()) != DATA_SEED:
        return None
    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        return None
    return d


def run_grid_resume(widths, depths, gains, base_lrs, seeds, train_loader, X_test, y_test, ftle_ii, ftle_jj):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))

    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        G_lambda_map = loaded["G_lambda_map"]
        G_J_map = loaded["G_J_map"]
        rho_lambda_map = loaded["rho_lambda_map"]
        rho_J_map = loaded["rho_J_map"]
        done_map = loaded["done_map"].astype(bool)
        print(f"[grid-state] loaded {done_map.sum()}/{done_map.size} cells")
    else:
        G_lambda_map = np.full(shape, np.nan, np.float64)
        G_J_map = np.full(shape, np.nan, np.float64)
        rho_lambda_map = np.full(shape, np.nan, np.float64)
        rho_J_map = np.full(shape, np.nan, np.float64)
        done_map = np.zeros(shape, bool)

    total = done_map.size

    for gi, gain in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue

                    print(f"\n[cell] N={N} L={L} g={gain} lr={lr}")
                    seed_stats = []
                    for sd in seeds:
                        seed_stats.append(
                            compute_or_resume_seed_stats(
                                N, L, gain, lr, sd,
                                train_loader=train_loader,
                                X_test=X_test, y_test=y_test,
                                ftle_ii=ftle_ii, ftle_jj=ftle_jj,
                            )
                        )

                    agg = aggregate_config_pooled(seed_stats, L=L)
                    print(f"[cell-done] seeds_used={agg.get('n_good', 'NA')}/{len(seeds)} ...")

                    G_lambda_map[gi, li, di, wi] = agg["G_lambda_mean"]
                    G_J_map[gi, li, di, wi] = agg["G_J_mean"]
                    rho_lambda_map[gi, li, di, wi] = agg["rho_lambda_mean"]
                    rho_J_map[gi, li, di, wi] = agg["rho_J_mean"]
                    done_map[gi, li, di, wi] = True

                    done = int(done_map.sum())
                    print(f"[cell-done] ({done}/{total})  Gλ={agg['G_lambda_mean']:.3e}  ρλ={agg['rho_lambda_mean']:.3f}  ρλ(unsat)={agg['rho_lambda_unsat']:.3f}  sat={agg['sat_frac']:.3f}")

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map,
                    )

    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map,
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        G_lambda_map=G_lambda_map, G_J_map=G_J_map,
        rho_lambda_map=rho_lambda_map, rho_J_map=rho_J_map,
        done_map=done_map,
    )


# -------------------- Plotting --------------------
def plot_heatmap(mat2d: np.ndarray, widths: List[int], depths: List[int],
                 title: str, out_path: str, vmin=None, vmax=None, log10: bool = False):
    plt.figure(figsize=(6, 4))
    M = mat2d.copy()
    if log10:
        M = np.log10(M + 1e-12)
    M = np.ma.masked_invalid(M)
    im = plt.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Width N")
    plt.ylabel("Depth L")
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_all_slices(grid: Dict):
    widths = grid["widths"]
    depths = grid["depths"]
    gains = grid["gains"]
    base_lrs = grid["base_lrs"]

    G_lambda_map = grid["G_lambda_map"]
    rho_lambda_map = grid["rho_lambda_map"]
    G_J_map = grid["G_J_map"]
    rho_J_map = grid["rho_J_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            gstr = fmt_float(float(g))
            lrstr = fmt_float(float(lr))

            plot_heatmap(G_lambda_map[gi, li], widths, depths,
                         title=f"log10 Var[λ]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_log10_Glambda_g{gstr}_lr{lrstr}.png"),
                         log10=True)
            plot_heatmap(rho_lambda_map[gi, li], widths, depths,
                         title=f"rho(λ, margin) [pooled]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_rho_lambda_g{gstr}_lr{lrstr}.png"),
                         vmin=-1, vmax=1)
            plot_heatmap(G_J_map[gi, li], widths, depths,
                         title=f"log10 Var[||J||]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_log10_GJ_g{gstr}_lr{lrstr}.png"),
                         log10=True)
            plot_heatmap(rho_J_map[gi, li], widths, depths,
                         title=f"rho(||J||, margin) [pooled]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_rho_J_g{gstr}_lr{lrstr}.png"),
                         vmin=-1, vmax=1)


# -------------------- Entry --------------------
if __name__ == "__main__":
    os.makedirs(FTLE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("[device]", device)
    if device.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))

    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)

    # Precompute grid indices ONCE (big speed win)
    ii, jj = precompute_ftle_indices(xe.cpu().numpy(), FTLE_GRID, BBOX)

    X_test = xe.to(device)
    y_test = ye.to(device)

    if DO_COMPUTE:
        grid = run_grid_resume(
            WIDTHS, DEPTHS, GAINS, BASE_LRS, SEEDS,
            train_loader=train_loader,
            X_test=X_test, y_test=y_test,
            ftle_ii=ii, ftle_jj=jj,
        )
    else:
        d = safe_load_npz(GRID_STATE)
        if d is None:
            raise RuntimeError(f"No grid state found at {GRID_STATE}. Run with DO_COMPUTE=True first.")
        grid = dict(
            widths=d["widths"].astype(int).tolist(),
            depths=d["depths"].astype(int).tolist(),
            gains=d["gains"].astype(float).tolist(),
            base_lrs=d["base_lrs"].astype(float).tolist(),
            seeds=d["seeds"].astype(int).tolist(),
            G_lambda_map=d["G_lambda_map"],
            G_J_map=d["G_J_map"],
            rho_lambda_map=d["rho_lambda_map"],
            rho_J_map=d["rho_J_map"],
            done_map=d["done_map"],
        )

    if DO_PLOT:
        print("[plot] generating figures ...")
        plot_all_slices(grid)
        print("[plot] saved to:", PLOT_DIR)
