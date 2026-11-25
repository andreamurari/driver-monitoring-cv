"""Train a classifier on State Farm.

Usage:
    python -m dms.train --config configs/resnet18.yaml
    python -m dms.train --config configs/resnet18.yaml --overfit-batch   # local sanity check
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dms.data.loaders import build_dataloaders
from dms.models.classifier import build_model, count_parameters, get_normalization, set_backbone_trainable


def set_seed(seed: int) -> None:
    """Same seed -> same initial head weights, same batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    """One pass over the training set. Returns {"loss": ..., "acc": ...}."""
    model.train()  # enables dropout and BatchNorm statistics updates
    total_loss, correct, seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()  # gradients accumulate: clear the previous batch's
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()  # computes gradients, does not change the weights
        optimizer.step()  # updates the weights

        # loss.item() is the batch mean: multiply by the batch size to sum correctly
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        seen += x.size(0)

    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()  # no gradients in evaluation: less memory, faster
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    """Loss and accuracy on a loader, without updating the weights.

    The mean CrossEntropy loss is the log-loss (the Kaggle competition metric).
    """
    model.eval()  # BatchNorm uses the learned statistics, dropout off
    total_loss, correct, seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        seen += x.size(0)

    return {"loss": total_loss / seen, "acc": correct / seen}


def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )


def run_phase(name, model, loaders, criterion, cfg_phase, device, writer, run_dir, state):
    """One training phase (head only or full fine-tuning) with a cosine schedule."""
    set_backbone_trainable(model, cfg_phase["train_backbone"])
    optimizer = make_optimizer(model, cfg_phase["lr"], cfg_phase["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_phase["epochs"])
    print(f"[{name}] parameters (total, trainable): {count_parameters(model)}")
    for _ in range(cfg_phase["epochs"]):
        state["epoch"] += 1
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        e = state["epoch"]
        for split, m in (("train", tr), ("val", va)):
            for k, v in m.items():
                writer.add_scalar(f"{k}/{split}", v, e)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], e)
        print(
            f"[{name}] epoch {e}: train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.3f} | {time.time() - t0:.0f}s"
        )

        if va["loss"] < state["best_val_loss"]:
            state["best_val_loss"] = va["loss"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            state["best_epoch"] = e

        state["history"].append({ "epoch": e,
                                 "phase": name,
                                 "train_loss": tr["loss"],
                                 "train_acc": tr["acc"],
                                 "val_loss": va["loss"],
                                 "val_acc": va["acc"] })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit-batch", action="store_true",
                        help="train repeatedly on a single batch: the loss must go to ~0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, torch {torch.__version__}, cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")

    run_dir = Path("runs") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model first: the loaders use its pretraining normalisation
    model = build_model(**cfg["model"]).to(device)
    mean, std = get_normalization(model)
    loaders = build_dataloaders(**cfg["data"], seed=cfg["seed"], mean=mean, std=std)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    if args.overfit_batch:
        overfit_one_batch(model, loaders["train"], criterion, device)
        return

    writer = SummaryWriter(run_dir / "tb")
    state = {"epoch": 0, "best_epoch": 0, "history": [], "best_val_loss": float("inf")}
    for phase_name in ("head", "finetune"):
        run_phase(phase_name, model, loaders, criterion, cfg[phase_name], device,
                  writer, run_dir, state)
    writer.close()

    # Evaluate the best checkpoint on val (the test set is used once, at the end of the project)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    final = evaluate(model, loaders["val"], criterion, device)
    final["best_epoch_val_loss"] = state["best_val_loss"]
    final["best_epoch"] = state["best_epoch"]
    (run_dir / "metrics.json").write_text(json.dumps({"config": cfg, "val": final, "history": state["history"]}, indent=2))
    print("final val:", final)


def overfit_one_batch(model, loader, criterion, device, steps: int = 100) -> None:
    """Sanity check: on a single batch a working network reaches a loss of ~0.

    If it does not, there is a bug (labels, loss, optimizer) and a long training run on
    Kaggle makes no sense.
    """
    set_backbone_trainable(model, True)
    optimizer = make_optimizer(model, lr=1e-3, weight_decay=0.0)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"step {step}: loss {loss.item():.4f} acc {acc:.3f}")


if __name__ == "__main__":  # required on Windows with num_workers > 0
    main()
