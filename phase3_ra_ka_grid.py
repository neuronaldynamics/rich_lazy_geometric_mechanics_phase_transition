import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import random
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ra_ka_best_method_accstop import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    ckpt_path,
    fmt_float,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
)

# ---------------- GPU knobs ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# ---------------- Paths / cache ----------------
PHASE2_GRID_STATE = "phase2_grid_state.npz"   # to auto-load axes
DATA_SEED = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

CACHE_DIR  = "ra_ka_cache"
GRID_STATE = "ra_ka_grid_state.npz"
PLOT_DIR   = "plots_ra_ka"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- RA/KA config ----------------
SEED_BASE = 0

# Probe set: set PROBE_SUBSET (e.g. 2048) if you want faster RA
PROBE_SUBSET = None

# KA subset
KA_SUBSET = 64
KA_SUBSET_SEED = 12345

# Keep your existing values so you don't invalidate old caches/state:
CACHE_VERSION = 1
GRID_VERSION  = 1

# Optional: only compute cells that phase2 finished
PHASE2_DONE_ONLY = False

# ---------------- Phase3 extras: (A)/(B)/(C) ----------------
AUX_VERSION = 1  # bump if you change definitions

# (A) Boundary metric on the *same grid spec* as FTLE (no JVP needed)
BOUNDARY_GRID  = 161
BOUNDARY_BBOX  = (-1.2, 1.2)
BOUNDARY_BATCH = 32768  # forward batch for boundary eval

# (B) Functional probe subset for across-seed similarity (logits)
FUNC_SUBSET      = 1024     # 512/1024/2048
FUNC_SUBSET_SEED = 2026

# (C) Parameter-noise robustness
NOISE_ALPHA    = 0.02       # try: 0.005, 0.01, 0.02, 0.05
NOISE_SAMPLES  = 1          # 1 is cheapest; 3 gives smoother averages
NOISE_ON_BIAS  = True
FUNC_BATCH     = 65536      # forward batch for logits_func

# ---------------- I/O utils ----------------
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


# ---------------- Dataset caching ----------------
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


# ---------------- Model loading ----------------
def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader) -> Optional[FC]:
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


# ---------------- Alignment math ----------------
def frob_cosine(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> float:
    num = (A * B).sum()
    den = torch.linalg.norm(A) * torch.linalg.norm(B) + eps
    return float((num / den).detach().cpu())


@torch.no_grad()
def linear_cka_features(H0: torch.Tensor, HT: torch.Tensor, eps: float = 1e-12) -> float:
    H0c = H0 - H0.mean(dim=0, keepdim=True)
    HTc = HT - HT.mean(dim=0, keepdim=True)

    A = H0c.T @ HTc
    B = H0c.T @ H0c
    C = HTc.T @ HTc

    num = (A * A).sum()
    den = torch.linalg.norm(B) * torch.linalg.norm(C) + eps
    return float((num / den).detach().cpu())


# ---------------- NTK alignment (KA) ----------------
try:
    from torch.func import functional_call, vmap, grad
    _HAS_TORCHFUNC = True
except Exception:
    _HAS_TORCHFUNC = False


def grad_matrix(net: FC, xs: torch.Tensor) -> torch.Tensor:
    net.eval()

    if _HAS_TORCHFUNC:
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        names = list(params.keys())

        def f(p, b, x):
            y = functional_call(net, (p, b), (x.unsqueeze(0),), kwargs={"grad": True})
            return y.squeeze()

        g = vmap(grad(f), in_dims=(None, None, 0))(params, buffers, xs)
        flats = [g[name].reshape(xs.shape[0], -1) for name in names]
        return torch.cat(flats, dim=1)

    # fallback loop
    rows = []
    params_list = [p for p in net.parameters() if p.requires_grad]
    for i in range(xs.shape[0]):
        net.zero_grad(set_to_none=True)
        y = net(xs[i:i + 1], grad=True)
        y.sum().backward()
        flat = [p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params_list]
        rows.append(torch.cat(flat))
    return torch.stack(rows, 0)


@torch.no_grad()
def ntk_align(net_init: FC, net_trained: FC, X_ka: torch.Tensor) -> float:
    with torch.enable_grad():
        G0 = grad_matrix(net_init, X_ka)
        GT = grad_matrix(net_trained, X_ka)
    K0 = G0 @ G0.T
    KT = GT @ GT.T
    return frob_cosine(KT, K0)


# ---------------- (A) boundary edge-count / length ----------------
@torch.no_grad()
def boundary_edgecount(net: FC, grid: int, bbox: Tuple[float, float], batch: int) -> Tuple[float, float]:
    """
    Boundary length proxy: count sign-changes along edges of a uniform grid.
    Uses pre-tanh logits (net(..., grad=True)) so sign corresponds to decision boundary.
    Returns:
      edges_count (float), length_estimate (float)
    """
    xs = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    ys = torch.linspace(bbox[0], bbox[1], grid, device=DEVICE)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    net.eval()
    logits = torch.empty((pts.shape[0],), device=DEVICE, dtype=torch.float32)

    for s in range(0, pts.shape[0], batch):
        xb = pts[s:s + batch]
        z = net(xb, grad=True).squeeze()
        logits[s:s + xb.shape[0]] = z.float()

    pred = (logits.reshape(grid, grid) > 0).detach().cpu().numpy()

    edges = int((pred[:, 1:] != pred[:, :-1]).sum() + (pred[1:, :] != pred[:-1, :]).sum())

    cell = float((bbox[1] - bbox[0]) / (grid - 1))
    length_est = float(edges) * cell
    return float(edges), float(length_est)


# ---------------- (B) functional logits + similarities ----------------
@torch.no_grad()
def logits_on_X(net: FC, X: torch.Tensor, batch: int = 65536) -> np.ndarray:
    """
    Returns pre-tanh logits on X, shape [n].
    """
    net.eval()
    out = []
    for s in range(0, X.shape[0], batch):
        xb = X[s:s + batch]
        z = net(xb, grad=True).squeeze()
        out.append(z.float().detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def cosine_centered(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return num / den


def mean_pairwise_centered_cos(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("nan")
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_centered(vectors[i], vectors[j]))
    return float(np.mean(sims)) if sims else float("nan")


def mean_pairwise_agreement(preds_bool: List[np.ndarray]) -> float:
    if len(preds_bool) < 2:
        return float("nan")
    agrees = []
    for i in range(len(preds_bool)):
        for j in range(i + 1, len(preds_bool)):
            agrees.append(float(np.mean(preds_bool[i] == preds_bool[j])))
    return float(np.mean(agrees)) if agrees else float("nan")


# ---------------- (C) parameter-noise robustness ----------------
@torch.no_grad()
def param_noise_metrics(
    net: FC,
    X_func: torch.Tensor,
    y_func: torch.Tensor,
    alpha: float,
    samples: int,
    noise_on_bias: bool,
    seed: int,
) -> Tuple[float, float]:
    """
    Adds Gaussian noise to parameters (scaled by tensor std), measures:
      sens = 1 - centered_cos(clean_logits, noisy_logits)
      acc_drop = clean_acc - noisy_acc
    Returns mean over 'samples'.
    """
    clean = logits_on_X(net, X_func, batch=FUNC_BATCH)
    y_np = y_func.squeeze().detach().cpu().numpy()
    clean_acc = float(np.mean(((clean > 0).astype(np.float32) * 2 - 1) == y_np))

    params = [p for p in net.parameters()]
    orig = [p.detach().clone() for p in params]

    sens_list = []
    drop_list = []

    for s in range(samples):
        g = torch.Generator(device=DEVICE)
        g.manual_seed(int(1000003 * (seed + 1) + 97 * (s + 1)))

        # add noise
        for p in params:
            if (not noise_on_bias) and (p.ndim == 1):
                continue
            std = float(p.detach().float().std().item())
            if (not np.isfinite(std)) or std == 0.0:
                continue
            noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g)
            p.add_(noise * (alpha * std))


        noisy = logits_on_X(net, X_func, batch=FUNC_BATCH)

        sens = 1.0 - cosine_centered(clean, noisy)
        noisy_acc = float(np.mean(((noisy > 0).astype(np.float32) * 2 - 1) == y_np))

        sens_list.append(float(sens))
        drop_list.append(float(clean_acc - noisy_acc))

        # restore
        for p, o in zip(params, orig):
            p.copy_(o)

    return float(np.mean(sens_list)), float(np.mean(drop_list))


# ---------------- Per-seed caching ----------------
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedrakastats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_kasub{KA_SUBSET}_dseed{DATA_SEED}.npz",
    )


def cache_ok(d: Optional[Dict[str, np.ndarray]]) -> bool:
    if d is None:
        return False
    if not bool(np.array(d.get("finished", False)).item()):
        return False
    if int(np.array(d.get("cache_version", 0)).item()) != CACHE_VERSION:
        return False
    return True


def _scalar(d: Dict[str, np.ndarray], k: str, default=None):
    if d is None or k not in d:
        return default
    return np.array(d[k]).item()


def aux_meta_ok(d: Optional[Dict[str, np.ndarray]], n_func: int) -> bool:
    if d is None:
        return False
    if int(_scalar(d, "aux_version", 0)) != int(AUX_VERSION):
        return False
    if int(_scalar(d, "func_n", -1)) != int(n_func):
        return False
    if int(_scalar(d, "func_seed", -1)) != int(FUNC_SUBSET_SEED):
        return False
    if not np.isclose(float(_scalar(d, "noise_alpha", np.nan)), float(NOISE_ALPHA), rtol=0, atol=0):
        return False
    if int(_scalar(d, "noise_samples", -1)) != int(NOISE_SAMPLES):
        return False
    if int(_scalar(d, "boundary_grid", -1)) != int(BOUNDARY_GRID):
        return False
    if "boundary_bbox" not in d:
        return False
    if not np.allclose(d["boundary_bbox"].astype(np.float32), np.array(BOUNDARY_BBOX, np.float32)):
        return False
    return True


def compute_or_load_seed_ra_ka(
    N: int, L: int, gain: float, base_lr: float, seed: int,
    train_loader,
    X_probe: torch.Tensor,
    X_ka: torch.Tensor,
    X_func: torch.Tensor,
    y_func: torch.Tensor,
) -> Dict[str, np.ndarray]:
    path = seed_cache_path(N, L, gain, base_lr, seed)
    cached = safe_load_npz(path) if os.path.exists(path) else None
    n_func = int(X_func.shape[0])

    # Fast return if cache has everything
    if cached is not None:
        if bool(np.array(cached.get("finished", False)).item()) and int(_scalar(cached, "cache_version", 0)) == CACHE_VERSION:
            train_ok = bool(np.array(cached.get("train_ok", True)).item())
            if not train_ok:
                return cached

            # check required keys for full completion
            need_keys = ["RA", "KA", "boundary_len", "boundary_edges", "logits_func", "noise_sens", "noise_acc_drop"]
            keys_ok = all(k in cached for k in need_keys)
            shape_ok = ("logits_func" in cached and cached["logits_func"].shape == (n_func,))
            meta_ok = aux_meta_ok(cached, n_func)

            if keys_ok and shape_ok and meta_ok:
                return cached

    # Determine what is missing (so we don't recompute KA if we don't need to)
    have_RA = cached is not None and ("RA" in cached) and np.isfinite(cached["RA"]).item()
    have_KA = cached is not None and ("KA" in cached) and np.isfinite(cached["KA"]).item()

    meta_ok = aux_meta_ok(cached, n_func)
    have_boundary = meta_ok and cached is not None and ("boundary_len" in cached) and ("boundary_edges" in cached)
    have_logits   = meta_ok and cached is not None and ("logits_func" in cached) and (cached["logits_func"].shape == (n_func,))
    have_noise    = meta_ok and cached is not None and ("noise_sens" in cached) and ("noise_acc_drop" in cached)

    need_RA = not have_RA
    need_KA = not have_KA
    need_boundary = not have_boundary
    need_logits = not have_logits
    need_noise = not have_noise

    # If nothing to do, return cached (shouldn't happen, but safe)
    if cached is not None and (not need_RA) and (not need_KA) and (not need_boundary) and (not need_logits) and (not need_noise):
        return cached

    # Load trained net only if we need anything at all
    net_tr = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    if net_tr is None:
        out = dict(
            cache_version=np.array(CACHE_VERSION, np.int32),
            aux_version=np.array(AUX_VERSION, np.int32),
            finished=np.array(True),
            train_ok=np.array(False),
            N=np.array(N, np.int32), L=np.array(L, np.int32),
            gain=np.array(gain, np.float32),
            base_lr=np.array(base_lr, np.float32),
            seed=np.array(seed, np.int32),

            func_n=np.array(n_func, np.int32),
            func_seed=np.array(FUNC_SUBSET_SEED, np.int32),
            noise_alpha=np.array(NOISE_ALPHA, np.float32),
            noise_samples=np.array(NOISE_SAMPLES, np.int32),
            boundary_grid=np.array(BOUNDARY_GRID, np.int32),
            boundary_bbox=np.array(BOUNDARY_BBOX, np.float32),

            RA=np.array(np.nan, np.float64),
            KA=np.array(np.nan, np.float64),
            boundary_edges=np.array(np.nan, np.float64),
            boundary_len=np.array(np.nan, np.float64),
            logits_func=np.full((n_func,), np.nan, np.float32),
            noise_sens=np.array(np.nan, np.float64),
            noise_acc_drop=np.array(np.nan, np.float64),
        )
        atomic_save_npz(path, **out)
        return out

    net_tr.eval()

    # Compute RA/KA only if needed
    ra = float(cached["RA"]) if (cached is not None and "RA" in cached and np.isfinite(cached["RA"]).item()) else np.nan
    ka = float(cached["KA"]) if (cached is not None and "KA" in cached and np.isfinite(cached["KA"]).item()) else np.nan

    if need_RA or need_KA:
        # deterministic init for net_init (only if needed)
        torch.manual_seed(SEED_BASE + seed)
        np.random.seed(SEED_BASE + seed)
        random.seed(SEED_BASE + seed)

        net_init = FC(N, L, gain=gain).to(DEVICE)
        net_init.eval()

        if need_RA:
            with torch.inference_mode():
                H0 = net_init(X_probe, hid=True)
                HT = net_tr(X_probe, hid=True)
            ra = linear_cka_features(H0, HT)

        if need_KA:
            ka = ntk_align(net_init, net_tr, X_ka)

        del net_init

    # (A) boundary metric (cheap forward passes on a grid)
    if need_boundary:
        b_edges, b_len = boundary_edgecount(net_tr, grid=BOUNDARY_GRID, bbox=BOUNDARY_BBOX, batch=BOUNDARY_BATCH)
    else:
        b_edges = float(cached["boundary_edges"])
        b_len   = float(cached["boundary_len"])

    # (B) functional logits on fixed subset
    if need_logits:
        logits_func = logits_on_X(net_tr, X_func, batch=FUNC_BATCH).astype(np.float32)
    else:
        logits_func = cached["logits_func"].astype(np.float32, copy=False)

    # (C) parameter-noise robustness
    if need_noise:
        noise_sens, noise_drop = param_noise_metrics(
            net_tr, X_func=X_func, y_func=y_func,
            alpha=float(NOISE_ALPHA),
            samples=int(NOISE_SAMPLES),
            noise_on_bias=bool(NOISE_ON_BIAS),
            seed=int(seed),
        )
    else:
        noise_sens = float(cached["noise_sens"])
        noise_drop = float(cached["noise_acc_drop"])

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        aux_version=np.array(AUX_VERSION, np.int32),
        finished=np.array(True),
        train_ok=np.array(True),

        N=np.array(N, np.int32), L=np.array(L, np.int32),
        gain=np.array(gain, np.float32),
        base_lr=np.array(base_lr, np.float32),
        seed=np.array(seed, np.int32),

        func_n=np.array(n_func, np.int32),
        func_seed=np.array(FUNC_SUBSET_SEED, np.int32),
        noise_alpha=np.array(NOISE_ALPHA, np.float32),
        noise_samples=np.array(NOISE_SAMPLES, np.int32),
        boundary_grid=np.array(BOUNDARY_GRID, np.int32),
        boundary_bbox=np.array(BOUNDARY_BBOX, np.float32),

        RA=np.array(ra, np.float64),
        KA=np.array(ka, np.float64),

        boundary_edges=np.array(b_edges, np.float64),
        boundary_len=np.array(b_len, np.float64),
        logits_func=logits_func,
        noise_sens=np.array(noise_sens, np.float64),
        noise_acc_drop=np.array(noise_drop, np.float64),
    )
    atomic_save_npz(path, **out)
    return out


# ---------------- Grid state ----------------
def save_grid_state(path: str,
                    widths, depths, gains, base_lrs, seeds,
                    RA_map, KA_map, RA_std_map, KA_std_map,
                    BL_map, BL_std_map,
                    NS_map, NS_std_map,
                    ND_map, ND_std_map,
                    FS_cos_map, FS_agree_map,
                    done_map):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, np.int32),
        data_seed=np.array(DATA_SEED, np.int32),
        widths=np.array(widths, np.int32),
        depths=np.array(depths, np.int32),
        gains=np.array(gains, np.float32),
        base_lrs=np.array(base_lrs, np.float32),
        seeds=np.array(seeds, np.int32),
        KA_SUBSET=np.array(KA_SUBSET, np.int32),

        # meta for new metrics
        aux_version=np.array(AUX_VERSION, np.int32),
        FUNC_SUBSET=np.array(FUNC_SUBSET, np.int32),
        FUNC_SUBSET_SEED=np.array(FUNC_SUBSET_SEED, np.int32),
        NOISE_ALPHA=np.array(NOISE_ALPHA, np.float32),
        NOISE_SAMPLES=np.array(NOISE_SAMPLES, np.int32),
        BOUNDARY_GRID=np.array(BOUNDARY_GRID, np.int32),
        BOUNDARY_BBOX=np.array(BOUNDARY_BBOX, np.float32),

        RA_map=RA_map.astype(np.float64),
        KA_map=KA_map.astype(np.float64),
        RA_std_map=RA_std_map.astype(np.float64),
        KA_std_map=KA_std_map.astype(np.float64),

        BL_map=BL_map.astype(np.float64),
        BL_std_map=BL_std_map.astype(np.float64),
        NS_map=NS_map.astype(np.float64),
        NS_std_map=NS_std_map.astype(np.float64),
        ND_map=ND_map.astype(np.float64),
        ND_std_map=ND_std_map.astype(np.float64),
        FS_cos_map=FS_cos_map.astype(np.float64),
        FS_agree_map=FS_agree_map.astype(np.float64),

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
    if int(np.array(d.get("KA_SUBSET", -1)).item()) != KA_SUBSET:
        return None

    # new meta checks
    if int(np.array(d.get("aux_version", 0)).item()) != AUX_VERSION:
        return None
    if int(np.array(d.get("FUNC_SUBSET", -1)).item()) != FUNC_SUBSET:
        return None
    if int(np.array(d.get("FUNC_SUBSET_SEED", -1)).item()) != FUNC_SUBSET_SEED:
        return None
    if not np.isclose(float(np.array(d.get("NOISE_ALPHA", np.nan)).item()), float(NOISE_ALPHA), rtol=0, atol=0):
        return None
    if int(np.array(d.get("NOISE_SAMPLES", -1)).item()) != NOISE_SAMPLES:
        return None
    if int(np.array(d.get("BOUNDARY_GRID", -1)).item()) != BOUNDARY_GRID:
        return None
    if "BOUNDARY_BBOX" not in d or not np.allclose(d["BOUNDARY_BBOX"].astype(np.float32), np.array(BOUNDARY_BBOX, np.float32)):
        return None

    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        return None

    # Require new maps exist
    for k in ["BL_map","NS_map","ND_map","FS_cos_map","FS_agree_map"]:
        if k not in d:
            return None

    return d



# ---------------- Axes ----------------
def load_axes_from_phase2_or_defaults():
    widths  = [10, 20, 30, 50, 100, 150, 200, 250]
    depths  = [2, 4, 6, 8, 10, 12, 14, 16]
    gains   = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    base_lrs = [0.05, 0.075, 0.10, 0.20, 0.30, 0.40]
    seeds   = [0, 1, 2, 3, 4, 5]
    done_mask = None

    if os.path.exists(PHASE2_GRID_STATE):
        d = safe_load_npz(PHASE2_GRID_STATE)
        if d is not None:
            widths = d["widths"].astype(int).tolist()
            depths = d["depths"].astype(int).tolist()
            gains = d["gains"].astype(float).tolist()
            base_lrs = d["base_lrs"].astype(float).tolist()
            seeds = d["seeds"].astype(int).tolist()
            if PHASE2_DONE_ONLY and "done_map" in d:
                done_mask = d["done_map"].astype(bool)
            print("[axes] loaded axes from phase2_grid_state.npz")
    return widths, depths, gains, base_lrs, seeds, done_mask


# ---------------- Main runner ----------------
def run_ra_ka_grid(widths, depths, gains, base_lrs, seeds, train_loader,X_probe: torch.Tensor, X_ka: torch.Tensor, X_func: torch.Tensor, y_func: torch.Tensor, phase2_done_mask):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))

    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        RA_map = loaded["RA_map"]
        KA_map = loaded["KA_map"]
        RA_std_map = loaded["RA_std_map"]
        KA_std_map = loaded["KA_std_map"]

        BL_map = loaded["BL_map"]
        BL_std_map = loaded["BL_std_map"]
        NS_map = loaded["NS_map"]
        NS_std_map = loaded["NS_std_map"]
        ND_map = loaded["ND_map"]
        ND_std_map = loaded["ND_std_map"]
        FS_cos_map = loaded["FS_cos_map"]
        FS_agree_map = loaded["FS_agree_map"]

        done_map = loaded["done_map"].astype(bool)
        print(f"[ra/ka grid] loaded {done_map.sum()}/{done_map.size} cells")

    else:
        RA_map = np.full(shape, np.nan, np.float64)
        KA_map = np.full(shape, np.nan, np.float64)
        RA_std_map = np.full(shape, np.nan, np.float64)
        KA_std_map = np.full(shape, np.nan, np.float64)
        BL_map = np.full(shape, np.nan, np.float64)
        BL_std_map = np.full(shape, np.nan, np.float64)
        NS_map = np.full(shape, np.nan, np.float64)
        NS_std_map = np.full(shape, np.nan, np.float64)
        ND_map = np.full(shape, np.nan, np.float64)
        ND_std_map = np.full(shape, np.nan, np.float64)
        FS_cos_map = np.full(shape, np.nan, np.float64)
        FS_agree_map = np.full(shape, np.nan, np.float64)
        done_map = np.zeros(shape, bool)

    total = done_map.size

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue
                    if phase2_done_mask is not None and not phase2_done_mask[gi, li, di, wi]:
                        continue

                    print(f"\n[cell] N={N} L={L} g={g} lr={lr}")

                    ra_vals, ka_vals = [], []
                    b_len_vals = []
                    noise_sens_vals = []
                    noise_drop_vals = []
                    logits_list = []

                    for sd in seeds:
                        sdat = compute_or_load_seed_ra_ka(
                            N, L, g, lr, sd,
                            train_loader=train_loader,
                            X_probe=X_probe,
                            X_ka=X_ka,
                            X_func=X_func,
                            y_func=y_func,
                        )

                        ok = bool(np.array(sdat.get("train_ok", True)).item())
                        if not ok:
                            continue
                        ra_vals.append(float(sdat["RA"]))
                        ka_vals.append(float(sdat["KA"]))
                        b_len_vals.append(float(sdat["boundary_len"]))
                        noise_sens_vals.append(float(sdat["noise_sens"]))
                        noise_drop_vals.append(float(sdat["noise_acc_drop"]))
                        logits_list.append(sdat["logits_func"].astype(np.float64, copy=False))

                    FS_cos = mean_pairwise_centered_cos(logits_list)
                    FS_agree = mean_pairwise_agreement([(z > 0) for z in logits_list])

                    if len(ra_vals) == 0:
                        RA_map[gi, li, di, wi] = np.nan
                        KA_map[gi, li, di, wi] = np.nan
                        RA_std_map[gi, li, di, wi] = np.nan
                        KA_std_map[gi, li, di, wi] = np.nan
                        BL_map[gi, li, di, wi] = np.nan
                        BL_std_map[gi, li, di, wi] = np.nan
                        NS_map[gi, li, di, wi] = np.nan
                        NS_std_map[gi, li, di, wi] = np.nan
                        ND_map[gi, li, di, wi] = np.nan
                        ND_std_map[gi, li, di, wi] = np.nan
                        FS_cos_map[gi, li, di, wi] = np.nan
                        FS_agree_map[gi, li, di, wi] = np.nan
                    else:
                        RA_map[gi, li, di, wi] = float(np.mean(ra_vals))
                        KA_map[gi, li, di, wi] = float(np.mean(ka_vals))
                        RA_std_map[gi, li, di, wi] = float(np.std(ra_vals, ddof=0))
                        KA_std_map[gi, li, di, wi] = float(np.std(ka_vals, ddof=0))
                        BL_map[gi, li, di, wi] = float(np.mean(b_len_vals))
                        BL_std_map[gi, li, di, wi] = float(np.std(b_len_vals, ddof=0))

                        NS_map[gi, li, di, wi] = float(np.mean(noise_sens_vals))
                        NS_std_map[gi, li, di, wi] = float(np.std(noise_sens_vals, ddof=0))

                        ND_map[gi, li, di, wi] = float(np.mean(noise_drop_vals))
                        ND_std_map[gi, li, di, wi] = float(np.std(noise_drop_vals, ddof=0))

                        FS_cos_map[gi, li, di, wi] = float(FS_cos)
                        FS_agree_map[gi, li, di, wi] = float(FS_agree)
                    done_map[gi, li, di, wi] = True

                    done = int(done_map.sum())
                    #print(f"[cell-done] ({done}/{total})  RA={RA_map[gi,li,di,wi]:.3f}±{RA_std_map[gi,li,di,wi]:.3f}  "
                    #      f"KA={KA_map[gi,li,di,wi]:.3f}±{KA_std_map[gi,li,di,wi]:.3f}")
                    print(f"[cell-done] ({done}/{total})  "
                          f"RA={RA_map[gi,li,di,wi]:.3f}±{RA_std_map[gi,li,di,wi]:.3f}  "
                          f"KA={KA_map[gi,li,di,wi]:.3f}±{KA_std_map[gi,li,di,wi]:.3f}  "
                          f"BL={BL_map[gi,li,di,wi]:.3e}  FS_cos={FS_cos_map[gi,li,di,wi]:.3f}  "
                          f"noise_sens={NS_map[gi,li,di,wi]:.3e}")

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        RA_map, KA_map, RA_std_map, KA_std_map,
                        BL_map, BL_std_map,
                        NS_map, NS_std_map,
                        ND_map, ND_std_map,
                        FS_cos_map, FS_agree_map,
                        done_map,
                    )
    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        RA_map, KA_map, RA_std_map, KA_std_map,
        BL_map, BL_std_map,
        NS_map, NS_std_map,
        ND_map, ND_std_map,
        FS_cos_map, FS_agree_map,
        done_map,
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        RA_map=RA_map, KA_map=KA_map,
        RA_std_map=RA_std_map, KA_std_map=KA_std_map,
        done_map=done_map, 
        BL_map=BL_map, BL_std_map=BL_std_map,
        NS_map=NS_map, NS_std_map=NS_std_map,
        ND_map=ND_map, ND_std_map=ND_std_map,
        FS_cos_map=FS_cos_map,
        FS_agree_map=FS_agree_map,
    )


# ---------------- Plotting ----------------
def plot_heatmap(mat2d: np.ndarray, widths: List[int], depths: List[int],
                 title: str, out_path: str, vmin=0, vmax=1):
    plt.figure(figsize=(6, 4))
    M = np.ma.masked_invalid(mat2d)
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


def plot_ra_ka_slices(grid: Dict):
    widths = grid["widths"]
    depths = grid["depths"]
    gains = grid["gains"]
    lrs = grid["base_lrs"]

    RA = grid["RA_map"]
    KA = grid["KA_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            gstr = fmt_float(float(g))
            lrstr = fmt_float(float(lr))

            plot_heatmap(RA[gi, li], widths, depths,
                         title=f"RA (linear CKA)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_RA_g{gstr}_lr{lrstr}.png"))
            plot_heatmap(KA[gi, li], widths, depths,
                         title=f"KA (NTK align)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_KA_g{gstr}_lr{lrstr}.png"))


# ---------------- Entry ----------------
if __name__ == "__main__":
    print("[device]", DEVICE)
    if DEVICE.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))
        print("[torch.func]", _HAS_TORCHFUNC)

    widths, depths, gains, base_lrs, seeds, phase2_done_mask = load_axes_from_phase2_or_defaults()
    print("[grid] widths", widths)
    print("[grid] depths", depths)
    print("[grid] gains", gains)
    print("[grid] lrs", base_lrs)
    print("[grid] seeds", seeds)

    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=DEVICE)

    X_probe_full = xe.to(DEVICE)
    y_probe_full = ye.to(DEVICE)

    genf = torch.Generator(device="cpu").manual_seed(FUNC_SUBSET_SEED)
    n_func = min(FUNC_SUBSET, X_probe_full.shape[0])
    idxf = torch.randperm(X_probe_full.shape[0], generator=genf)[:n_func].to(torch.long)

    X_func = X_probe_full[idxf.to(DEVICE)]
    y_func = y_probe_full[idxf.to(DEVICE)]

    if PROBE_SUBSET is None or PROBE_SUBSET >= X_probe_full.shape[0]:
        X_probe = X_probe_full
    else:
        gen = torch.Generator(device="cpu").manual_seed(999)
        idxp = torch.randperm(X_probe_full.shape[0], generator=gen)[:PROBE_SUBSET].to(torch.long)
        X_probe = X_probe_full[idxp.to(DEVICE)]

    gen = torch.Generator(device="cpu").manual_seed(KA_SUBSET_SEED)
    idx = torch.randperm(X_probe_full.shape[0], generator=gen)[:KA_SUBSET].to(torch.long)
    X_ka = X_probe_full[idx.to(DEVICE)]

    grid = run_ra_ka_grid(
        widths, depths, gains, base_lrs, seeds,
        train_loader=train_loader,
        X_probe=X_probe,
        X_ka=X_ka,
        X_func=X_func,
        y_func=y_func,
        phase2_done_mask=phase2_done_mask,
    )

    print("[saved]", GRID_STATE)
    plot_ra_ka_slices(grid)
    print("[plots] saved to", PLOT_DIR)
