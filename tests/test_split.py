"""Tests for the shared chip-id split -- the mechanism that guarantees
the U-Net and YOLO notebooks evaluate on exactly the same held-out data.
"""

import pytest

from src.split import load_split, save_split, split_chip_ids


def _chip_ids(n: int) -> list[str]:
    return [f"chip_{i:04d}" for i in range(n)]


def test_split_preserves_every_id_exactly_once():
    ids = _chip_ids(100)
    result = split_chip_ids(ids)

    all_returned = result["train"] + result["val"] + result["test"]
    assert sorted(all_returned) == sorted(ids)
    assert len(set(all_returned)) == len(ids)  # no id duplicated across splits


def test_split_respects_approximate_fractions():
    ids = _chip_ids(1000)
    result = split_chip_ids(ids, train_frac=0.7, val_frac=0.15)

    assert len(result["train"]) == 700
    assert len(result["val"]) == 150
    assert len(result["test"]) == 150


def test_split_is_deterministic_given_the_same_seed():
    ids = _chip_ids(200)
    first = split_chip_ids(ids, seed=42)
    second = split_chip_ids(ids, seed=42)
    assert first == second


def test_split_is_order_independent():
    """Same set of ids, different input order (e.g. two different
    directory listings) -- must produce the identical split, not a
    different one depending on filesystem ordering."""
    ids = _chip_ids(50)
    shuffled_input = list(reversed(ids))

    assert split_chip_ids(ids, seed=1) == split_chip_ids(shuffled_input, seed=1)


def test_split_differs_with_a_different_seed():
    ids = _chip_ids(200)
    first = split_chip_ids(ids, seed=1)
    second = split_chip_ids(ids, seed=2)
    assert first != second


def test_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        split_chip_ids(_chip_ids(10), train_frac=0.9, val_frac=0.5)  # sums to > 1


def test_split_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        split_chip_ids(["a", "b", "a"])


def test_save_and_load_split_round_trips(tmp_path):
    split = split_chip_ids(_chip_ids(30))
    path = tmp_path / "split.json"

    save_split(split, path)
    loaded = load_split(path)

    assert loaded == split
