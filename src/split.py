"""Splits SpaceNet chip ids into train/val/test sets, once, shared by both
the U-Net and YOLO data-prep pipelines -- the whole reason both models
end up evaluated on the exact same held-out chips. See the practice
plan's pitfall on spatial data leakage: splitting by anything finer than
a whole chip (e.g. by individual building or pixel) lets buildings in
the same tile leak between splits and makes both models' scores look
better than they really are.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def split_chip_ids(
    chip_ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Splits a list of chip ids into train/val/test, by whole chip --
    never by individual building or pixel. Deterministic given the same
    input and seed, so re-running this reproduces the same split rather
    than silently drifting between the U-Net and YOLO notebooks (or
    between reruns of the same notebook).
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1) and sum to less than 1")
    if len(set(chip_ids)) != len(chip_ids):
        raise ValueError("chip_ids contains duplicates")

    # Sorted first so the shuffle is deterministic regardless of the
    # order chip_ids happened to arrive in (e.g. from an unordered
    # directory listing).
    shuffled = sorted(chip_ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(split, indent=2))


def load_split(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())
