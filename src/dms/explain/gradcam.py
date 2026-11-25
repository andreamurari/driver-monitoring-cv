"""Grad-CAM: where a CNN looks when it picks a class.

Usage (locally; the output images contain dataset photos and stay in scratch/, git-ignored):
    python -m dms.explain.gradcam --run runs/kaggle/resnet18_320_s0 --top 30

Takes the `top` most confident errors from <run>/predictions_val.csv and, for each, saves to
scratch/gradcam/ the image with the map of the predicted (wrong) class and of the true class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dms.data.dataset import CLASSES
from dms.data.transforms import _resize_ops, get_transforms
from dms.models.classifier import build_model, get_normalization


class GradCAM:
    """Grad-CAM on a convolutional layer (for the ResNet: model.layer4)."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        self.activations = None  # layer output in the forward pass: (1, C, h, w)
        self.gradients = None    # gradient of the class score w.r.t. that output: (1, C, h, w)
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        """Hook: PyTorch calls it on every forward pass of the layer, passing its output."""
        self.activations = output.detach()
        # A second hook, on the tensor: called in the backward pass with its gradient
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Map (H, W) with values between 0 and 1, at the resolution of x."""
        # Gradients are needed here: no torch.no_grad()
        self.model.zero_grad()
        logits = self.model(x)          # the forward pass fills self.activations
        logits[0, class_idx].backward()  # the backward pass fills self.gradients

        # 1. one weight per map: how much the class depends on it (mean gradient) -> (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 2. weighted sum of the maps, only the part that pushes towards the class -> (1, 1, h, w)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # 3. from h x w (10 x 10 at 320) to the input resolution
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # 4. values between 0 and 1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam[0, 0].cpu().numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay the map ('jet' colours) on the uint8 RGB image."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]  # (H, W, 3) between 0 and 1
    return ((1 - alpha) * image / 255 + alpha * heat).clip(0, 1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--img-root", default="data/raw/imgs/train")
    parser.add_argument("--out", default="scratch/gradcam")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    if not cfg["model"]["name"].startswith("resnet"):
        raise SystemExit("Grad-CAM here is written for ResNets (layer4)")
    size, mode = cfg["data"]["size"], cfg["data"]["mode"]

    model = build_model(cfg["model"]["name"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    mean, std = get_normalization(model)
    to_tensor = get_transforms(False, size, mode, mean=mean, std=std)
    cam_fn = GradCAM(model, model.layer4)

    pred = pd.read_csv(run_dir / "predictions_val.csv")
    errors = pred[pred.y_true != pred.y_pred].sort_values("confidence", ascending=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(errors.head(args.top).itertuples()):
        pil = Image.open(Path(args.img_root) / row.classname / row.img).convert("RGB")
        x = to_tensor(pil).unsqueeze(0)
        # The image the model sees (resized + bands), without normalisation
        shown = np.array(pil)
        for op in _resize_ops(size, mode):
            shown = np.array(op(Image.fromarray(shown)))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(shown)
        axes[0].set_title(f"#{i} {row.subject} true {row.classname}")
        for ax, idx, label in ((axes[1], row.y_pred, "predicted"), (axes[2], row.y_true, "true")):
            ax.imshow(overlay(shown, cam_fn(x, int(idx))))
            ax.set_title(f"{label}: {CLASSES[idx]} (p={getattr(row, f'p_{CLASSES[idx]}'):.2f})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{row.subject}_{row.classname}_as_{CLASSES[row.y_pred]}.png",
                    dpi=90)
        plt.close(fig)
    print(f"saved {min(args.top, len(errors))} images to {out}/ (git-ignored: do not publish)")


if __name__ == "__main__":
    main()
