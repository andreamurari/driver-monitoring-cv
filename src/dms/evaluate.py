"""Detailed evaluation of a trained run: predictions, per-class and per-driver metrics.

Usage (locally, on the validation set):
    python -m dms.evaluate --run runs/kaggle/resnet18_320_s0

Writes:
    <run>/predictions_val.csv                 one row per image (git-ignored, lives in runs/)
    reports/metrics/<name>_val_detail.json    aggregate metrics (publishable)
    reports/figures/confusion_<name>.png      normalised confusion matrix (publishable)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dms.data.dataset import CLASSES
from dms.data.loaders import build_dataloaders
from dms.models.classifier import build_model, get_normalization


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Probabilities for every image of the loader, in the same order: array (N, 10)."""
    model.eval()
    all_probs = []
    for x, _ in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)  # 10 logits -> 10 probabilities summing to 1
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)  # list of (batch, 10) -> (N, 10)


def compute_metrics(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """df has the columns "subject", "y_true", "y_pred" (class indices); probs is (N, 10)."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, log_loss)

    y_true, y_pred = df["y_true"], df["y_pred"]
    labels = list(range(len(CLASSES)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, probs, labels=labels),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels, normalize="true").tolist(),
        # per-driver accuracy: mean of right/wrong, grouped by subject
        "per_driver": (y_true == y_pred).groupby(df["subject"]).mean().round(3).to_dict(),
    }


def plot_confusion(cm: list[list[float]], title: str, path: Path) -> None:
    """Confusion matrix as an image: rows = true class, columns = predicted class."""
    import matplotlib
    matplotlib.use("Agg")  # no window, file only
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder with metrics.json and best.pt")
    parser.add_argument("--split", default="val")  # the test set is used once, at the end
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    name = cfg["name"]

    # The saved config has Kaggle paths: use the local defaults
    data_cfg = {k: v for k, v in cfg["data"].items() if k not in ("csv_path", "img_root")}
    data_cfg["num_workers"] = args.num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"]["name"], pretrained=False)  # weights come from best.pt
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device)

    mean, std = get_normalization(model)
    loader = build_dataloaders(**data_cfg, seed=cfg["seed"], mean=mean, std=std)[args.split]
    print(f"{name}: {len(loader.dataset)} {args.split} images on {device}")

    probs = predict(model, loader, device)

    # The val loader does not shuffle: row i of probs is row i of the DataFrame
    df = loader.dataset.df[["img", "subject", "classname"]].copy()
    df["y_true"] = df["classname"].map(CLASSES.index)
    df["y_pred"] = probs.argmax(axis=1)
    df["confidence"] = probs.max(axis=1)
    for i, c in enumerate(CLASSES):
        df[f"p_{c}"] = probs[:, i]
    df.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)

    metrics = compute_metrics(df, probs)
    out = Path("reports/metrics") / f"{name}_{args.split}_detail.json"
    out.write_text(json.dumps(metrics, indent=2))
    plot_confusion(metrics["confusion"], f"{name} ({args.split})",
                   Path("reports/figures") / f"confusion_{name}.png")

    print(f"accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
          f"log-loss {metrics['log_loss']:.4f}")
    print("per driver:", metrics["per_driver"])
    print(f"saved: {out}, reports/figures/confusion_{name}.png, "
          f"{run_dir / f'predictions_{args.split}.csv'}")


if __name__ == "__main__":  # required on Windows with num_workers > 0
    main()
