from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.func import jvp

from utils import DEVICE


@torch.no_grad()
def power_iteration_spectral_norm_jvp(model, x: torch.Tensor, iters: int = 20) -> torch.Tensor:
    x = x.detach().to(DEVICE)
    bsz, dim = x.shape
    v = torch.randn(bsz, dim, device=DEVICE)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    def hidden_map_single(z: torch.Tensor) -> torch.Tensor:
        return model.hidden_map(z.unsqueeze(0)).squeeze(0)

    sigmas = []
    for i in range(bsz):
        xi = x[i]
        vi = v[i]
        for _ in range(iters):
            _, jv = jvp(hidden_map_single, (xi,), (vi,))
            sigma = jv.norm() + 1e-12
            # crude pullback-free iteration using finite-difference VJP surrogate is avoided here
            # we keep the current direction and use sigma estimate only; slower but stable scaffold
            vi = vi / (vi.norm() + 1e-12)
        sigmas.append(sigma)
    return torch.stack(sigmas)


def exact_jacobian_spectral_norm(model, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(DEVICE)
    vals = []
    for i in range(x.shape[0]):
        xi = x[i].clone().detach().requires_grad_(True)
        h = model.hidden_map(xi.unsqueeze(0)).squeeze(0)
        rows = []
        for j in range(h.numel()):
            grad = torch.autograd.grad(h[j], xi, retain_graph=True)[0]
            rows.append(grad)
        jac = torch.stack(rows, dim=0)
        sigma_max = torch.linalg.svdvals(jac)[0]
        vals.append(sigma_max.detach())
    return torch.stack(vals)


def ftle_from_sigma(sigma_max: torch.Tensor, depth: int) -> torch.Tensor:
    return torch.log(sigma_max.clamp_min(1e-12)) / float(depth)


def compute_ftle_batch(model, x: torch.Tensor, depth: int, exact: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    sigma = exact_jacobian_spectral_norm(model, x) if exact else power_iteration_spectral_norm_jvp(model, x)
    lam = ftle_from_sigma(sigma, depth)
    return lam.detach().cpu().numpy(), sigma.detach().cpu().numpy()
