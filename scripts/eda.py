"""EDA di base: classi, conducenti, immagini campione. Scrive in reports/.

Uso: python scripts/eda.py --csv data/raw/driver_imgs_list.csv --imgs data/raw/imgs/train
Le immagini campione restano in reports/samples/ (che e' in .gitignore): non
vanno committate finche' non hai letto le regole di State Farm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dms.data.dataset import CLASS_NAMES
from dms.data.splits import load_driver_list


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/driver_imgs_list.csv")
    p.add_argument("--imgs", default=None, help="cartella con le sottocartelle c0..c9")
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_driver_list(args.csv)

    per_class = df["classname"].value_counts().sort_index()
    per_subject = df["subject"].value_counts().sort_index()
    cross = df.groupby(["subject", "classname"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    per_class.plot.bar(ax=ax)
    ax.set_title("Immagini per classe")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "eda_classes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    per_subject.plot.bar(ax=ax)
    ax.set_title("Immagini per conducente")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "eda_subjects.png", dpi=150)
    plt.close(fig)

    lines = [
        "# EDA State Farm",
        "",
        f"- Immagini etichettate: {len(df)}",
        f"- Conducenti: {df['subject'].nunique()}",
        f"- Immagini per conducente: min {per_subject.min()}, mediana {int(per_subject.median())}, max {per_subject.max()}",
        f"- Rapporto classe piu' grande / piu' piccola: {per_class.max() / per_class.min():.2f}",
        f"- Coppie conducente-classe vuote: {int((cross == 0).sum().sum())}",
        "",
        "| Classe | Significato | Immagini |",
        "| --- | --- | --- |",
    ]
    for c, n in per_class.items():
        lines.append(f"| {c} | {CLASS_NAMES.get(c, '')} | {n} |")
    (out / "eda.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.imgs:
        from PIL import Image

        samples = out / "samples"
        samples.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax, c in zip(axes.ravel(), sorted(df["classname"].unique())):
            row = df[df["classname"] == c].sample(1, random_state=0).iloc[0]
            ax.imshow(Image.open(Path(args.imgs) / c / row["img"]))
            ax.set_title(f"{c}: {CLASS_NAMES.get(c, '')}", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(samples / "one_per_class.png", dpi=120)
        plt.close(fig)
        print(f"Campioni salvati in {samples}")


if __name__ == "__main__":
    main()
