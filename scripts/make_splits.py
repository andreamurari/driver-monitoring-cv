"""Crea splits.json con split per conducente.

Uso: python scripts/make_splits.py --csv data/raw/driver_imgs_list.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms.data.splits import (
    assert_no_overlap,
    load_driver_list,
    make_group_splits,
    save_splits,
    summarize,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/driver_imgs_list.csv")
    p.add_argument("--out", default="splits.json")
    p.add_argument("--n-val", type=int, default=4)
    p.add_argument("--n-test", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = load_driver_list(args.csv)
    splits = make_group_splits(df, args.n_val, args.n_test, args.seed)
    assert_no_overlap(splits)
    save_splits(splits, args.seed, args.out)

    print(f"{df['subject'].nunique()} conducenti, {len(df)} immagini")
    print(summarize(df, splits).to_string(index=False))
    print(f"Salvato in {args.out}")


if __name__ == "__main__":
    main()
