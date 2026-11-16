import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms.data.splits import (
    assert_no_overlap,
    make_group_splits,
    random_image_split,
)


@pytest.fixture()
def df() -> pd.DataFrame:
    rows = []
    for s in range(26):
        for c in range(10):
            for k in range(5):
                rows.append({"subject": f"p{s:03d}", "classname": f"c{c}", "img": f"img_{s}_{c}_{k}.jpg"})
    return pd.DataFrame(rows)


def test_no_driver_shared_between_splits(df):
    splits = make_group_splits(df, n_val=4, n_test=4, seed=0)
    assert_no_overlap(splits)
    for a, b in itertools.combinations(splits, 2):
        assert not set(splits[a]) & set(splits[b])


def test_every_driver_assigned_exactly_once(df):
    splits = make_group_splits(df)
    assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(assigned) == sorted(df["subject"].unique())
    assert len(assigned) == len(set(assigned))


def test_split_sizes(df):
    splits = make_group_splits(df, n_val=4, n_test=4)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (18, 4, 4)


def test_deterministic_for_same_seed(df):
    assert make_group_splits(df, seed=1) == make_group_splits(df, seed=1)


def test_different_seed_changes_split(df):
    assert make_group_splits(df, seed=1) != make_group_splits(df, seed=2)


def test_overlap_is_detected():
    with pytest.raises(AssertionError):
        assert_no_overlap({"train": ["p1", "p2"], "val": ["p2"], "test": ["p3"]})


def test_too_many_holdout_drivers_raises(df):
    with pytest.raises(ValueError):
        make_group_splits(df, n_val=13, n_test=13)


def test_random_image_split_does_leak_drivers(df):
    """Documenta il motivo dell'esperimento: lo split casuale mescola i conducenti."""
    parts = random_image_split(df)
    assert set(parts["train"]["subject"]) & set(parts["test"]["subject"])
