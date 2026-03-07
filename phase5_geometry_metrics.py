import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

import random
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, distance_transform_edt, label
from torch.func import jvp

from ra_ka_best_method_accstop import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    fmt_float,
    DEVICE,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
)

# -------------------- CONFIG --------------------
PHASE2_GRID_STATE = 'phase2_grid_state.npz'
DATA_SEED = 0
DATA_CACHE_FILE = f'circle_data_seed{DATA_SEED}.npz'

FTLE_DIR = 'ftle'
CACHE_DIR = 'ftle_geom_cache'
GRID_STATE = 'phase5_geometry_state.npz'
os.makedirs(CACHE_DIR, exist_ok=True)

# Compute only for phase2-done cells by default
PHASE2_DONE_ONLY = True

# Geometry controls
BOUNDARY_GRID = 161
BOUNDARY_BBOX = (-1.2, 1.2)
BOUNDARY_BATCH = 32768
RIDGE_Q = 0.95
RIDGE_GDOT_TOL = 0.15
SMOOTH_SIGMA = 1.0
MIN_RIDGE_PTS = 10
ANCHOR_PIXELS = 2.0      # ridge pixel considered boundary-anchored if within this many grid cells
ALIGN_BAND_PIXELS = 2.0  # compute alignment in this band around boundary

# Optional expensive metric
COMPUTE_ALIGN = True

# Versioning
CACHE_VERSION = 1
GRID_VERSION = 1
AUX_META = dict(
    boundary_grid=np.array(BOUNDARY_GRID, np.int32),
    boundary_bbox=np.array(BOUNDARY_BBOX, np.float32),
    ridge_q=np.array(RIDGE_Q, np.float32),
    ridge_gdot_tol=np.array(RIDGE_GDOT_TOL, np.float32),
    smooth_sigma=np.array(SMOOTH_SIGMA, np.float32),
    anchor_pixels=np.array(ANCHOR_PIXELS, np.float32),
    align_band_pixels=np.array(ALIGN_BAND_PIXELS, np.float32),
    compute_align=np.array(int(COMPUTE_ALIGN), np.int32),
)

# -------------------- utils --------------------
def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        print(f'[warn] failed to load {path}: {e}')
        return None


def safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except Exception as e:
        print(f'[warn] failed to load {path}: {e}')
        return None


def _scalar(d: Dict[str, np.ndarray], k: str, default=None):
    if d is None or k not in d:
        return default
    return np.array(d[k]).item()


def fmt_seed_path(N: int, L: int, g: float, lr: float, seed: int) -> str:
    return os.path.join(
        CACHE_DIR,
        f'geom_N{N}_L{L}_g{fmt_float(g)}_lr{fmt_float(lr)}_seed{seed}_grid{BOUNDARY_GRID}_dseed{DATA_SEED}.npz',
    )


def ftle_grid_path(N: int, L: int, g: float, lr: float, seed: int, grid: int) -> str:
    return os.path.join(
        FTLE_DIR,
        f'ftle_N{N}_L{L}_g{fmt_float(g)}_lr{fmt_float(lr)}_seed{seed}_g{grid}.npy',
    )


def geom_cache_ok(d: Optional[Dict[str, np.ndarray]]) -> bool:
    if d is None:
        return False
    if int(_scalar(d, 'cache_version', 0)) != CACHE_VERSION:
        return False
    if not bool(_scalar(d, 'finished', False)):
        return False
    for k, v in AUX_META.items():
        if k not in d:
            return False
        arr = d[k]
        if arr.shape != v.shape:
            return False
        if np.issubdtype(arr.dtype, np.floating):
            if not np.allclose(arr.astype(np.float64), v.astype(np.float64), rtol=0, atol=0):
                return False
        else:
            if not np.array_equal(arr, v):
                return False
    need = ['ridge_len', 'ridge_components', 'ridge_endpoints', 'ridge_junctions',
            'ridge_sharpness', 'ridge_boundary_anchor_frac', 'ridge_boundary_mean_dist',
            'boundary_len']
    if bool(int(COMPUTE_ALIGN)):
        need += ['align_mean', 'align_p90']
    return all(k in d for k in need)


# -------------------- dataset / model loading --------------------
def load_or_make_circle_data(cache_path: str, seed: int):
    if os.path.exists(cache_path):
        d = safe_load_npz(cache_path)
        if d is not None:
            xt = torch.tensor(d['xt'], dtype=torch.float32)
            yt = torch.tensor(d['yt'], dtype=torch.float32)
            xe = torch.tensor(d['xe'], dtype=torch.float32)
            ye = torch.tensor(d['ye'], dtype=torch.float32)
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


def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader) -> Optional[FC]:
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS,
        fail_policy='none',
    )
    if net is None:
        print(f'[skip-model] N={N} L={L} g={gain} lr={base_lr} seed={seed} failed acc target')
        return None
    net.eval()
    return net


# -------------------- geometry from scalar field --------------------
def _grad_hess_2d(f: np.ndarray, dx: float):
    fy, fx = np.gradient(f, dx, dx, edge_order=2)
    fyy, fyx = np.gradient(fy, dx, dx, edge_order=2)
    fxy, fxx = np.gradient(fx, dx, dx, edge_order=2)
    fxy = 0.5 * (fxy + fyx)
    return fx, fy, fxx, fyy, fxy


def _hessian_eigs_2x2(fxx: np.ndarray, fyy: np.ndarray, fxy: np.ndarray):
    tr = fxx + fyy
    disc = np.sqrt(np.maximum((fxx - fyy) ** 2 + 4.0 * (fxy ** 2), 0.0))
    eig_min = 0.5 * (tr - disc)
    eig_max = 0.5 * (tr + disc)
    return eig_min, eig_max


def _eigvec_min_2x2(fxx: np.ndarray, fyy: np.ndarray, fxy: np.ndarray, eig_min: np.ndarray, eps: float = 1e-12):
    vx = fxy
    vy = eig_min - fxx
    n1 = np.sqrt(vx * vx + vy * vy)
    ax = eig_min - fyy
    ay = fxy
    use_alt = n1 < (10.0 * eps)
    vx = np.where(use_alt, ax, vx)
    vy = np.where(use_alt, ay, vy)
    n = np.sqrt(vx * vx + vy * vy) + eps
    return vx / n, vy / n


def _ridge_mask(lam: np.ndarray, dx: float):
    f = gaussian_filter(lam, sigma=float(SMOOTH_SIGMA), mode='nearest') if SMOOTH_SIGMA > 0 else lam
    finite = np.isfinite(f)
    if finite.sum() < 10:
        return np.zeros_like(f, dtype=bool), f, None, None, None, None

    fx, fy, fxx, fyy, fxy = _grad_hess_2d(f, dx)
    eig_min, eig_max = _hessian_eigs_2x2(fxx, fyy, fxy)
    vx, vy = _eigvec_min_2x2(fxx, fyy, fxy, eig_min)
    gradn = np.sqrt(fx * fx + fy * fy) + 1e-12
    gdot = fx * vx + fy * vy
    thr = float(np.nanquantile(f[finite], RIDGE_Q))
    high = finite & (f >= thr)
    ridge = high & np.isfinite(eig_min) & (eig_min < 0.0) & (np.abs(gdot) <= float(RIDGE_GDOT_TOL) * gradn)
    if ridge.sum() < MIN_RIDGE_PTS:
        ridge = high & np.isfinite(eig_min) & (eig_min < 0.0)
    return ridge.astype(bool), f, eig_min, fx, fy, (fxx, fyy, fxy)


def _adjacency_counts(mask: np.ndarray) -> Tuple[int, int, int]:
    H, W = mask.shape
    m = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode='constant', constant_values=False)
    deg = np.zeros((H, W), dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            deg += m[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    endpoints = int((mask & (deg == 1)).sum())
    junctions = int((mask & (deg >= 3)).sum())
    _, comps = label(mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    return int(comps), endpoints, junctions


def _ridge_length(mask: np.ndarray, dx: float) -> float:
    Hh = np.logical_and(mask[:, 1:], mask[:, :-1]).sum()
    Hv = np.logical_and(mask[1:, :], mask[:-1, :]).sum()
    Hd1 = np.logical_and(mask[1:, 1:], mask[:-1, :-1]).sum()
    Hd2 = np.logical_and(mask[1:, :-1], mask[:-1, 1:]).sum()
    return float(dx * (Hh + Hv) + dx * np.sqrt(2.0) * (Hd1 + Hd2))


def boundary_from_logits(logits: np.ndarray) -> np.ndarray:
    pred = logits > 0.0
    b = np.zeros_like(pred, dtype=bool)
    hchg = pred[:, 1:] != pred[:, :-1]
    vchg = pred[1:, :] != pred[:-1, :]
    b[:, 1:] |= hchg
    b[:, :-1] |= hchg
    b[1:, :] |= vchg
    b[:-1, :] |= vchg
    return b


# -------------------- model-evaluated fields --------------------
@torch.no_grad()
def logits_grid(net: FC, grid: int, bbox: Tuple[float, float], batch: int = 32768) -> np.ndarray:
    xs = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    ys = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    Xg, Yg = torch.meshgrid(xs, ys, indexing='xy')
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)
    out = torch.empty((pts.shape[0],), device=DEVICE, dtype=torch.float32)
    net.eval()
    for s in range(0, pts.shape[0], batch):
        xb = pts[s:s + batch]
        out[s:s + xb.shape[0]] = net(xb, grad=True).view(-1).float()
    return out.reshape(grid, grid).cpu().numpy().astype(np.float32)


@torch.no_grad()
def theta_grid(net: FC, grid: int, bbox: Tuple[float, float], batch: int = 0) -> np.ndarray:
    """
    Principal stretching direction of J^T J as an angle theta in [-pi/2, pi/2).
    For line-field use, sign is irrelevant; alignment uses |dot|.
    """
    xs = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    ys = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    Xg, Yg = torch.meshgrid(xs, ys, indexing='xy')
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)
    if batch is None or batch <= 0:
        batch = pts.shape[0]

    def hidden(z):
        return net(z, hid=True)

    th = torch.empty((pts.shape[0],), device=DEVICE, dtype=torch.float32)
    for s in range(0, pts.shape[0], batch):
        xb = pts[s:s + batch]
        v1 = torch.zeros_like(xb); v1[:, 0] = 1.0
        v2 = torch.zeros_like(xb); v2[:, 1] = 1.0
        _, j1 = jvp(hidden, (xb,), (v1,))
        _, j2 = jvp(hidden, (xb,), (v2,))
        a = (j1 * j1).sum(dim=1)
        b = (j2 * j2).sum(dim=1)
        c = (j1 * j2).sum(dim=1)
        th[s:s + xb.shape[0]] = 0.5 * torch.atan2(2.0 * c, a - b)
    return th.reshape(grid, grid).cpu().numpy().astype(np.float32)


# -------------------- per-seed geometry --------------------
def compute_seed_geometry(N: int, L: int, g: float, lr: float, seed: int, train_loader) -> Dict[str, np.ndarray]:
    path = fmt_seed_path(N, L, g, lr, seed)
    cached = safe_load_npz(path) if os.path.exists(path) else None
    if geom_cache_ok(cached):
        return cached

    ftle_path = ftle_grid_path(N, L, g, lr, seed, BOUNDARY_GRID)
    lam = safe_load_npy(ftle_path)
    if lam is None or lam.shape != (BOUNDARY_GRID, BOUNDARY_GRID):
        # can't do geometry without FTLE grid; try training only if checkpoint exists? safest: skip
        net = load_or_train_net(N, L, g, lr, seed, train_loader=train_loader)
        if net is None:
            out = dict(cache_version=np.array(CACHE_VERSION, np.int32), finished=np.array(True), train_ok=np.array(False),
                       N=np.array(N), L=np.array(L), gain=np.array(g, np.float32), base_lr=np.array(lr, np.float32), seed=np.array(seed, np.int32),
                       ridge_len=np.array(np.nan), ridge_components=np.array(np.nan), ridge_endpoints=np.array(np.nan), ridge_junctions=np.array(np.nan),
                       ridge_sharpness=np.array(np.nan), ridge_boundary_anchor_frac=np.array(np.nan), ridge_boundary_mean_dist=np.array(np.nan),
                       boundary_len=np.array(np.nan), align_mean=np.array(np.nan), align_p90=np.array(np.nan),
                       **AUX_META)
            atomic_save_npz(path, **out)
            return out
        # if FTLE grid missing but model exists, user should generate phase2 FTLE first; skip gracefully
        out = dict(cache_version=np.array(CACHE_VERSION, np.int32), finished=np.array(True), train_ok=np.array(False),
                   N=np.array(N), L=np.array(L), gain=np.array(g, np.float32), base_lr=np.array(lr, np.float32), seed=np.array(seed, np.int32),
                   ridge_len=np.array(np.nan), ridge_components=np.array(np.nan), ridge_endpoints=np.array(np.nan), ridge_junctions=np.array(np.nan),
                   ridge_sharpness=np.array(np.nan), ridge_boundary_anchor_frac=np.array(np.nan), ridge_boundary_mean_dist=np.array(np.nan),
                   boundary_len=np.array(np.nan), align_mean=np.array(np.nan), align_p90=np.array(np.nan),
                   **AUX_META)
        atomic_save_npz(path, **out)
        return out

    dx = float((BOUNDARY_BBOX[1] - BOUNDARY_BBOX[0]) / (BOUNDARY_GRID - 1))
    ridge, lam_s, eig_min, fx, fy, _ = _ridge_mask(lam.astype(np.float64), dx)

    # trained net only needed for boundary + alignment
    net = load_or_train_net(N, L, g, lr, seed, train_loader=train_loader)
    if net is None:
        out = dict(cache_version=np.array(CACHE_VERSION, np.int32), finished=np.array(True), train_ok=np.array(False),
                   N=np.array(N), L=np.array(L), gain=np.array(g, np.float32), base_lr=np.array(lr, np.float32), seed=np.array(seed, np.int32),
                   ridge_len=np.array(np.nan), ridge_components=np.array(np.nan), ridge_endpoints=np.array(np.nan), ridge_junctions=np.array(np.nan),
                   ridge_sharpness=np.array(np.nan), ridge_boundary_anchor_frac=np.array(np.nan), ridge_boundary_mean_dist=np.array(np.nan),
                   boundary_len=np.array(np.nan), align_mean=np.array(np.nan), align_p90=np.array(np.nan),
                   **AUX_META)
        atomic_save_npz(path, **out)
        return out

    logits = logits_grid(net, grid=BOUNDARY_GRID, bbox=BOUNDARY_BBOX, batch=BOUNDARY_BATCH)
    boundary = boundary_from_logits(logits)
    # boundary length (same sign-change edge-count metric as phase3 BL, but recomputed here for consistency)
    hchg = (logits[:, 1:] > 0) != (logits[:, :-1] > 0)
    vchg = (logits[1:, :] > 0) != (logits[:-1, :] > 0)
    boundary_len = float(dx * (hchg.sum() + vchg.sum()))

    # ridge descriptors
    RL = _ridge_length(ridge, dx)
    RC, RE, RJ = _adjacency_counts(ridge)
    rs = (-eig_min)[ridge] if eig_min is not None else np.array([], dtype=np.float64)
    RS = float(np.nanmean(rs)) if rs.size else float('nan')

    # anchoring to decision boundary
    dist = distance_transform_edt(~boundary) * dx
    anchor_radius = float(ANCHOR_PIXELS * dx)
    ridge_pts = ridge & np.isfinite(dist)
    if ridge_pts.sum() > 0:
        anchor_frac = float(np.mean(dist[ridge_pts] <= anchor_radius))
        mean_dist = float(np.mean(dist[ridge_pts]))
    else:
        anchor_frac, mean_dist = float('nan'), float('nan')

    # stretching-direction alignment to boundary normal near the boundary
    if COMPUTE_ALIGN:
        th = theta_grid(net, grid=BOUNDARY_GRID, bbox=BOUNDARY_BBOX, batch=BOUNDARY_BATCH)
        # boundary normal from logits gradient
        fy_s, fx_s = np.gradient(logits.astype(np.float64), dx, dx, edge_order=2)
        nrm = np.sqrt(fx_s * fx_s + fy_s * fy_s) + 1e-12
        nx = fx_s / nrm
        ny = fy_s / nrm
        vx = np.cos(th)
        vy = np.sin(th)
        align = np.abs(vx * nx + vy * ny)
        align_mask = ridge & (dist <= float(ALIGN_BAND_PIXELS * dx)) & np.isfinite(align)
        if align_mask.sum() > 0:
            align_mean = float(np.mean(align[align_mask]))
            align_p90 = float(np.quantile(align[align_mask], 0.90))
        else:
            align_mean, align_p90 = float('nan'), float('nan')
    else:
        align_mean, align_p90 = float('nan'), float('nan')

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),
        train_ok=np.array(True),
        N=np.array(N), L=np.array(L), gain=np.array(g, np.float32), base_lr=np.array(lr, np.float32), seed=np.array(seed, np.int32),
        ridge_len=np.array(RL, np.float64),
        ridge_components=np.array(RC, np.float64),
        ridge_endpoints=np.array(RE, np.float64),
        ridge_junctions=np.array(RJ, np.float64),
        ridge_sharpness=np.array(RS, np.float64),
        ridge_boundary_anchor_frac=np.array(anchor_frac, np.float64),
        ridge_boundary_mean_dist=np.array(mean_dist, np.float64),
        boundary_len=np.array(boundary_len, np.float64),
        align_mean=np.array(align_mean, np.float64),
        align_p90=np.array(align_p90, np.float64),
        **AUX_META,
    )
    atomic_save_npz(path, **out)
    return out


# -------------------- aggregate into maps --------------------
def save_grid_state(path: str, widths, depths, gains, lrs, seeds, maps: Dict[str, np.ndarray], done_map: np.ndarray):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, np.int32),
        data_seed=np.array(DATA_SEED, np.int32),
        widths=np.array(widths, np.int32),
        depths=np.array(depths, np.int32),
        gains=np.array(gains, np.float32),
        base_lrs=np.array(lrs, np.float32),
        seeds=np.array(seeds, np.int32),
        **AUX_META,
        **{k: v.astype(np.float64) for k, v in maps.items()},
        done_map=done_map.astype(np.bool_),
    )


def try_load_grid_state(path: str, widths, depths, gains, lrs, seeds) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None
    if int(_scalar(d, 'grid_version', 0)) != GRID_VERSION:
        return None
    if int(_scalar(d, 'data_seed', -1)) != DATA_SEED:
        return None
    if not np.array_equal(np.array(widths, np.int32), d.get('widths')):
        return None
    if not np.array_equal(np.array(depths, np.int32), d.get('depths')):
        return None
    if not np.allclose(np.array(gains, np.float32), d.get('gains').astype(np.float32), rtol=0, atol=0):
        return None
    if not np.allclose(np.array(lrs, np.float32), d.get('base_lrs').astype(np.float32), rtol=0, atol=0):
        return None
    if not np.array_equal(np.array(seeds, np.int32), d.get('seeds')):
        return None
    for k, v in AUX_META.items():
        if k not in d:
            return None
        arr = d[k]
        if np.issubdtype(arr.dtype, np.floating):
            if not np.allclose(arr.astype(np.float64), v.astype(np.float64), rtol=0, atol=0):
                return None
        else:
            if not np.array_equal(arr, v):
                return None
    return d


def load_axes_from_phase2_or_defaults():
    widths = [10, 20, 30, 50, 100, 150, 200, 250]
    depths = [2, 4, 6, 8, 10, 12, 14, 16]
    gains = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lrs = [0.05, 0.075, 0.10, 0.20, 0.30, 0.40]
    seeds = [0, 1, 2, 3, 4, 5]
    done_mask = None
    if os.path.exists(PHASE2_GRID_STATE):
        d = safe_load_npz(PHASE2_GRID_STATE)
        if d is not None:
            widths = d['widths'].astype(int).tolist()
            depths = d['depths'].astype(int).tolist()
            gains = d['gains'].astype(float).tolist()
            lrs = d['base_lrs'].astype(float).tolist()
            seeds = d['seeds'].astype(int).tolist()
            if PHASE2_DONE_ONLY and 'done_map' in d:
                done_mask = d['done_map'].astype(bool)
            print('[axes] loaded axes from phase2_grid_state.npz')
    return widths, depths, gains, lrs, seeds, done_mask


def run_geometry_grid(widths, depths, gains, lrs, seeds, train_loader, phase2_done_mask):
    shape = (len(gains), len(lrs), len(depths), len(widths))
    metric_names = [
        'ridge_len_map', 'ridge_components_map', 'ridge_endpoints_map', 'ridge_junctions_map',
        'ridge_sharpness_map', 'ridge_boundary_anchor_frac_map', 'ridge_boundary_mean_dist_map',
        'boundary_len_map', 'align_mean_map', 'align_p90_map'
    ]
    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, lrs, seeds)
    if loaded is not None:
        maps = {k: loaded[k] for k in metric_names}
        done_map = loaded['done_map'].astype(bool)
        print(f'[geom-grid] loaded {done_map.sum()}/{done_map.size} cells')
    else:
        maps = {k: np.full(shape, np.nan, np.float64) for k in metric_names}
        done_map = np.zeros(shape, dtype=bool)

    total = done_map.size
    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue
                    if phase2_done_mask is not None and not phase2_done_mask[gi, li, di, wi]:
                        continue
                    print(f'\n[cell] N={N} L={L} g={g} lr={lr}')
                    rows = []
                    for sd in seeds:
                        d = compute_seed_geometry(int(N), int(L), float(g), float(lr), int(sd), train_loader=train_loader)
                        ok = bool(_scalar(d, 'train_ok', True))
                        if ok:
                            rows.append(d)
                    if len(rows) == 0:
                        done_map[gi, li, di, wi] = True
                        save_grid_state(GRID_STATE, widths, depths, gains, lrs, seeds, maps, done_map)
                        print('[cell-done] no successful seeds')
                        continue
                    def mean_key(k):
                        return float(np.mean([float(r[k]) for r in rows]))
                    maps['ridge_len_map'][gi, li, di, wi] = mean_key('ridge_len')
                    maps['ridge_components_map'][gi, li, di, wi] = mean_key('ridge_components')
                    maps['ridge_endpoints_map'][gi, li, di, wi] = mean_key('ridge_endpoints')
                    maps['ridge_junctions_map'][gi, li, di, wi] = mean_key('ridge_junctions')
                    maps['ridge_sharpness_map'][gi, li, di, wi] = mean_key('ridge_sharpness')
                    maps['ridge_boundary_anchor_frac_map'][gi, li, di, wi] = mean_key('ridge_boundary_anchor_frac')
                    maps['ridge_boundary_mean_dist_map'][gi, li, di, wi] = mean_key('ridge_boundary_mean_dist')
                    maps['boundary_len_map'][gi, li, di, wi] = mean_key('boundary_len')
                    maps['align_mean_map'][gi, li, di, wi] = mean_key('align_mean')
                    maps['align_p90_map'][gi, li, di, wi] = mean_key('align_p90')
                    done_map[gi, li, di, wi] = True
                    print(f"[cell-done] ({done_map.sum()}/{total}) anchor={maps['ridge_boundary_anchor_frac_map'][gi,li,di,wi]:.3f} align={maps['align_mean_map'][gi,li,di,wi]:.3f} RL={maps['ridge_len_map'][gi,li,di,wi]:.3e}")
                    save_grid_state(GRID_STATE, widths, depths, gains, lrs, seeds, maps, done_map)
    save_grid_state(GRID_STATE, widths, depths, gains, lrs, seeds, maps, done_map)
    return maps


def main():
    print('[device]', DEVICE)
    if DEVICE.type == 'cuda':
        print('[gpu]', torch.cuda.get_device_name(0))
    widths, depths, gains, lrs, seeds, phase2_done_mask = load_axes_from_phase2_or_defaults()
    print('[grid] widths', widths)
    print('[grid] depths', depths)
    print('[grid] gains', gains)
    print('[grid] lrs', lrs)
    print('[grid] seeds', seeds)

    (xt, yt), _ = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=DEVICE)
    run_geometry_grid(widths, depths, gains, lrs, seeds, train_loader, phase2_done_mask)
    print('[saved]', GRID_STATE)


if __name__ == '__main__':
    main()
