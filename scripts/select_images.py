"""Sceglie le immagini da annotare (riquadri di telefono, bicchiere, mano, volto).

Regole:
  1. split per conducente: ogni immagine resta nello split del suo conducente (splits.json),
     così il detector si valuta su persone mai viste, come il classificatore;
  2. classi bilanciate per oggetto: c6 (l'unica fonte di bicchieri) sovra-campionata;
  3. fotogrammi distanziati: dentro ogni gruppo si prendono righe a intervalli regolari,
     perché righe vicine nel CSV sono fotogrammi quasi identici.

Uso:
    python scripts/select_images.py                      # primo lotto (150)
    python scripts/select_images.py --batch 2 --scale 1.5  # lotto successivo, senza ripetere

Scrive annotations/selection.csv: solo nomi di file, nessuna immagine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dms.data.splits import load_driver_list, load_splits

# Immagini per classe State Farm nel primo lotto (150 in tutto)
PER_CLASS = {
    "c6": 30,                                     # bere: l'unica fonte di bicchieri
    "c1": 15, "c2": 15, "c3": 15, "c4": 15,       # telefono: messaggi e chiamata, due mani
    "c0": 12, "c5": 12, "c7": 12, "c8": 12, "c9": 12,  # mani e volto, senza oggetti
}
SPLIT_SHARE = {"train": 4 / 6, "val": 1 / 6, "test": 1 / 6}  # circa 100 / 25 / 25
OUT = Path("annotations/selection.csv")


def pick_spaced(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """n righe del gruppo a intervalli regolari (non a caso), nell'ordine del CSV."""
    if len(group) <= n:
        return group
    positions = np.linspace(0, len(group) - 1, n).round().astype(int)  # equidistanti, dal primo all'ultimo
    return group.iloc[positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/driver_imgs_list.csv")
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0, help="moltiplica PER_CLASS")
    args = parser.parse_args()

    df = load_driver_list(args.csv)  # colonne: subject, classname, img (ordine del CSV)
    splits = load_splits(args.splits)

    # Lotti successivi: niente immagini già scelte
    previous = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["img"])
    df = df[~df["img"].isin(previous["img"])]

    chosen = []
    for split, subjects in splits.items():
        part = df[df["subject"].isin(subjects)]
        for classname, n_class in PER_CLASS.items():
            n = round(n_class * args.scale * SPLIT_SHARE[split])
            # Ordinando per conducente, le posizioni equidistanti cadono su tutti i conducenti
            # dello split in proporzione alle loro immagini
            group = part[part["classname"] == classname].sort_values("subject", kind="stable")
            chosen.append(pick_spaced(group, n).assign(split=split))

    new = pd.concat(chosen).assign(batch=args.batch)
    out = pd.concat([previous, new], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"lotto {args.batch}: {len(new)} immagini, totale {len(out)}")
    print(new.pivot_table(index="classname", columns="split", values="img", aggfunc="count",
                          fill_value=0))


if __name__ == "__main__":
    main()
