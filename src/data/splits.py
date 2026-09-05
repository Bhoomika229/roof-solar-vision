"""Reproducible train/val/test split generation.

Splits are computed once from a seeded shuffle of discovered sample ids and persisted as plain
text files (one id per line) under ``data/splits/`` so every script/run reads the exact same
partition regardless of dataset iteration order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.dataset import list_sample_ids


def make_splits(
    dataset_root: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1, and train_frac must be > 0")

    ids = list_sample_ids(dataset_root)
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1 if n - n_train > 1 else 0, int(round(n * val_frac)))
    n_train = min(n_train, n - 2) if n >= 3 else n_train

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def save_splits(splits: dict[str, list[str]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        (out_dir / f"{split_name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def load_splits(splits_dir: str | Path) -> dict[str, list[str]]:
    splits_dir = Path(splits_dir)
    result: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file '{path}' not found. Run `python scripts/prepare_data.py` first."
            )
        text = path.read_text(encoding="utf-8").strip()
        result[split_name] = text.split("\n") if text else []
    return result
