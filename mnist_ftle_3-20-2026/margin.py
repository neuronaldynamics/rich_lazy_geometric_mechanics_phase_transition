from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import DEVICE


def pgd_multiclass(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int) -> torch.Tensor:
    x0 = x.detach().to(DEVICE)
    y = y.detach().to(DEVICE)
    adv = x0.clone().detach()

    step = max(eps / max(steps // 2, 1), 1e-4)
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
        adv = adv.clamp(0.0, 1.0)
    return adv.detach()


@torch.no_grad()
def is_success(model, adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = model(adv.to(DEVICE)).argmax(dim=1)
    return pred != y.to(DEVICE)


def multiclass_margin_bisection(model, x: torch.Tensor, y: torch.Tensor, eps_hi: float, pgd_steps: int, bisection_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.zeros(x.shape[0], device=DEVICE)
    hi = torch.full((x.shape[0],), float(eps_hi), device=DEVICE)

    # first check upper bracket
    adv_hi = pgd_multiclass(model, x, y, eps_hi, pgd_steps)
    succ_hi = is_success(model, adv_hi, y)

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd_multiclass(model, x, y, eps=1.0, steps=1)  # placeholder init shape
        for i in range(x.shape[0]):
            adv[i:i+1] = pgd_multiclass(model, x[i:i+1], y[i:i+1], float(mid[i].item()), pgd_steps)
        succ = is_success(model, adv, y)
        hi = torch.where(succ, mid, hi)
        lo = torch.where(succ, lo, mid)

    margin = hi
    saturated = ~succ_hi
    return margin.detach().cpu(), saturated.detach().cpu()
