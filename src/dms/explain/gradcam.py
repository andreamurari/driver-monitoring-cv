"""Grad-CAM: dove guarda una CNN quando sceglie una classe.

Uso (in locale; le immagini prodotte contengono foto del dataset e restano in scratch/, gitignored):
    python -m dms.explain.gradcam --run runs/kaggle/resnet18_320_s0 --top 30

Prende i `top` errori più sicuri da <run>/predictions_val.csv e per ognuno salva in
scratch/gradcam/ l'immagine con la mappa della classe prevista (sbagliata) e di quella vera.
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
    """Grad-CAM su uno strato convoluzionale (per la ResNet: model.layer4)."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        self.activations = None  # uscita dello strato nel forward: (1, C, h, w)
        self.gradients = None    # gradiente della classe rispetto a quell'uscita: (1, C, h, w)
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        """Hook: PyTorch lo chiama a ogni forward dello strato, passandogli la sua uscita."""
        self.activations = output.detach()
        # Un secondo hook, sul tensore: verrà chiamato nel backward con il suo gradiente
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Mappa (H, W) con valori tra 0 e 1, alla risoluzione di x."""
        # Qui i gradienti servono: niente torch.no_grad()
        self.model.zero_grad()
        logits = self.model(x)          # il forward riempie self.activations
        logits[0, class_idx].backward()  # il backward riempie self.gradients

        # 1. un peso per mappa: quanto la classe dipende da essa (gradiente medio) -> (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 2. somma pesata delle mappe, solo la parte che spinge verso la classe -> (1, 1, h, w)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # 3. da h x w (10 x 10 a 320) alla risoluzione dell'input
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # 4. valori tra 0 e 1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam[0, 0].cpu().numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Sovrappone la mappa (colori 'jet') all'immagine RGB uint8."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]  # (H, W, 3) tra 0 e 1
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
        raise SystemExit("Grad-CAM qui è pensato per le ResNet (strato layer4)")
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
        # La stessa immagine che vede il modello (ridimensionata + bande), senza normalizzazione
        shown = np.array(pil)
        for op in _resize_ops(size, mode):
            shown = np.array(op(Image.fromarray(shown)))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(shown)
        axes[0].set_title(f"#{i} {row.subject} vera {row.classname}")
        for ax, idx, label in ((axes[1], row.y_pred, "prevista"), (axes[2], row.y_true, "vera")):
            ax.imshow(overlay(shown, cam_fn(x, int(idx))))
            ax.set_title(f"{label}: {CLASSES[idx]} (p={getattr(row, f'p_{CLASSES[idx]}'):.2f})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{row.subject}_{row.classname}_as_{CLASSES[row.y_pred]}.png",
                    dpi=90)
        plt.close(fig)
    print(f"salvate {min(args.top, len(errors))} immagini in {out}/ (gitignored: non pubblicarle)")


if __name__ == "__main__":
    main()
