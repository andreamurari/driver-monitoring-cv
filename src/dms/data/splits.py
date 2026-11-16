"""Split per conducente per State Farm Distracted Driver.

Le immagini dello stesso conducente sono quasi identiche tra loro (fotogrammi
consecutivi): uno split casuale per immagine fa trapelare informazione dal
train al test e gonfia l'accuratezza. Qui si divide per `subject`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"subject", "classname", "img"}


def load_driver_list(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {csv_path}: {sorted(missing)}")
    return df


def make_group_splits(
    df: pd.DataFrame, n_val: int = 4, n_test: int = 4, seed: int = 0
) -> dict[str, list[str]]:
    """Assegna interi conducenti a train/val/test in modo riproducibile."""
    subjects = sorted(df["subject"].unique())
    if n_val + n_test >= len(subjects):
        raise ValueError(
            f"Servono più conducenti: {len(subjects)} disponibili, "
            f"{n_val + n_test} richiesti per val+test"
        )
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test : n_test + n_val])
    train = sorted(shuffled[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


def assert_no_overlap(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(splits[a]) & set(splits[b])
            if common:
                raise AssertionError(f"Conducenti in comune tra {a} e {b}: {sorted(common)}")


def summarize(df: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Immagini e conducenti per split, e conteggio per classe."""
    rows = []
    for name, subs in splits.items():
        part = df[df["subject"].isin(subs)]
        row = {"split": name, "conducenti": len(subs), "immagini": len(part)}
        counts = part["classname"].value_counts().sort_index()
        row.update({c: int(counts.get(c, 0)) for c in sorted(df["classname"].unique())})
        rows.append(row)
    return pd.DataFrame(rows)


def save_splits(splits: dict[str, list[str]], seed: int, path: str | Path) -> None:
    payload = {"seed": seed, **splits}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    return {k: payload[k] for k in ("train", "val", "test")}


def random_image_split(df: pd.DataFrame, frac_val: float = 0.15, frac_test: float = 0.15, seed: int = 0):
    """Split casuale per immagine, SOLO per l'esperimento sul leakage (giorno 2)."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(shuffled) * frac_test)
    n_val = int(len(shuffled) * frac_val)
    return {
        "test": shuffled.iloc[:n_test],
        "val": shuffled.iloc[n_test : n_test + n_val],
        "train": shuffled.iloc[n_test + n_val :],
    }
