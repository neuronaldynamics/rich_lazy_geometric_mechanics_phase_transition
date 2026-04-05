from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def autocast_if_cuda(dtype: torch.dtype = torch.bfloat16) -> Iterator[None]:
    if DEVICE.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def count_correct(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == y).sum().item())
