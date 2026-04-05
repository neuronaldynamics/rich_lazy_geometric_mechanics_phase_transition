from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from configs import PathConfig, TrainConfig
from data import load_mnist_tensors, make_loader
from models import make_model
from paths import ckpt_path
from utils import DEVICE, atomic_write_json, count_correct, set_seed


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        total += y.numel()
        correct += count_correct(logits, y)
    return correct / max(total, 1)


def train_one(paths: PathConfig, cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    train_ds, test_ds = load_mnist_tensors(normalize=False)
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True)
    test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False)

    model = make_model(cfg.width, cfg.depth, cfg.gain).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr)

    best_test_acc = 0.0
    save_path = ckpt_path(paths, cfg)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate_accuracy(model, train_loader)
        test_acc = evaluate_accuracy(model, test_loader)
        best_test_acc = max(best_test_acc, test_acc)

        if test_acc >= cfg.target_test_acc:
            break

    payload = {
        "model_state": model.state_dict(),
        "train_config": asdict(cfg),
        "best_test_acc": best_test_acc,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
    }
    torch.save(payload, save_path)
    return {
        "final_train_acc": float(train_acc),
        "final_test_acc": float(test_acc),
        "best_test_acc": float(best_test_acc),
        "checkpoint": str(save_path),
    }


def train_job(job_path: Path, cfg: TrainConfig, logger=None) -> Dict[str, float]:
    set_seed(cfg.seed)
    train_ds, test_ds = load_mnist_tensors(normalize=False)
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True)
    test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False)

    model = make_model(cfg.width, cfg.depth, cfg.gain).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr)

    checkpoints_dir = job_path / "checkpoints"
    artifacts_dir = job_path / "artifacts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints_dir / "latest.pt"
    best_path = checkpoints_dir / "best.pt"
    train_metrics_path = artifacts_dir / "train_metrics.json"

    start_epoch = 1
    best_test_acc = 0.0
    best_state = None
    history: list[dict[str, float | int]] = []
    resumed = False

    if latest_path.exists():
        payload = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_test_acc = float(payload.get("best_test_acc", 0.0))
        best_state = payload.get("best_model_state")
        history = payload.get("history", [])
        resumed = True
        if logger is not None:
            logger.info("resuming from %s at epoch=%d", latest_path, start_epoch)

    final_train_acc = float("nan")
    final_test_acc = float("nan")
    stopped_early = False

    epoch_bar = tqdm(
        range(start_epoch, cfg.max_epochs + 1),
        desc=f"{job_path.name} train",
        initial=max(start_epoch - 1, 0),
        total=cfg.max_epochs,
        leave=True,
    )
    for epoch in epoch_bar:
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate_accuracy(model, train_loader)
        test_acc = evaluate_accuracy(model, test_loader)
        final_train_acc = float(train_acc)
        final_test_acc = float(test_acc)
        history.append(
            {
                "epoch": epoch,
                "train_acc": final_train_acc,
                "test_acc": final_test_acc,
                "best_test_acc": float(max(best_test_acc, final_test_acc)),
            }
        )

        improved = test_acc >= best_test_acc
        if improved:
            best_test_acc = float(test_acc)
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_config": asdict(cfg),
                    "best_test_acc": best_test_acc,
                    "final_test_acc": final_test_acc,
                    "final_train_acc": final_train_acc,
                },
                best_path,
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "train_config": asdict(cfg),
                "best_test_acc": best_test_acc,
                "best_model_state": best_state if best_state is not None else model.state_dict(),
                "history": history,
                "final_test_acc": final_test_acc,
                "final_train_acc": final_train_acc,
            },
            latest_path,
        )

        if logger is not None:
            logger.info(
                "epoch=%d/%d train_acc=%.4f test_acc=%.4f best_test_acc=%.4f",
                epoch,
                cfg.max_epochs,
                final_train_acc,
                final_test_acc,
                best_test_acc,
            )
        epoch_bar.set_postfix(
            train_acc=f"{final_train_acc:.4f}",
            test_acc=f"{final_test_acc:.4f}",
            best=f"{best_test_acc:.4f}",
        )

        if test_acc >= cfg.target_test_acc:
            stopped_early = True
            break
    epoch_bar.close()

    if not best_path.exists():
        torch.save(
            {
                "model_state": best_state if best_state is not None else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": cfg.max_epochs,
                "train_config": asdict(cfg),
                "best_test_acc": best_test_acc,
                "final_test_acc": final_test_acc,
                "final_train_acc": final_train_acc,
            },
            best_path,
        )

    metrics = {
        "resumed": resumed,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "last_epoch": int(history[-1]["epoch"]) if history else 0,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "best_test_acc": float(best_test_acc),
        "latest_checkpoint": str(latest_path.relative_to(job_path)),
        "best_checkpoint": str(best_path.relative_to(job_path)),
        "history": history,
    }
    atomic_write_json(train_metrics_path, metrics)
    return metrics


def load_trained_model(paths: PathConfig, cfg: TrainConfig):
    save_path = ckpt_path(paths, cfg)
    payload = torch.load(save_path, map_location=DEVICE)
    model = make_model(cfg.width, cfg.depth, cfg.gain).to(DEVICE)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def load_job_model(job_path: Path, cfg: TrainConfig, checkpoint_name: str = "best.pt"):
    checkpoint_path = job_path / "checkpoints" / checkpoint_name
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    model = make_model(cfg.width, cfg.depth, cfg.gain).to(DEVICE)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
