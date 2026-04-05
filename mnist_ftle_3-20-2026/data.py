from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def load_mnist_tensors(root: str | None = None, normalize: bool = False, download: bool | None = None) -> Tuple[TensorDataset, TensorDataset]:
    root = root or os.environ.get("MNIST_ROOT", "./data")
    if download is None:
        download = _env_flag("MNIST_DOWNLOAD", True)

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    transform = transforms.Compose(transform_list)

    try:
        train_ds = datasets.MNIST(root=root, train=True, download=False, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    except RuntimeError:
        if not download:
            raise RuntimeError(
                f"MNIST was not found under root={root!r} and downloading is disabled. "
                "Set MNIST_ROOT to an existing dataset directory or enable downloading with MNIST_DOWNLOAD=1."
            )
        try:
            train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
            test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        except RuntimeError as exc:
            raise RuntimeError(
                f"MNIST is not available under root={root!r} and automatic download failed. "
                "On clusters this is often an SSL/certificate issue. "
                "If you already have MNIST elsewhere, set MNIST_ROOT=/path/to/data and rerun. "
                "If you want to forbid download attempts, set MNIST_DOWNLOAD=0."
            ) from exc

    def to_tensor_dataset(ds: datasets.MNIST) -> TensorDataset:
        xs = torch.stack([img.view(-1) for img, _ in ds], dim=0)
        ys = torch.tensor([label for _, label in ds], dtype=torch.long)
        return TensorDataset(xs, ys)

    return to_tensor_dataset(train_ds), to_tensor_dataset(test_ds)


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
