#!/usr/bin/env python
"""Discover a dataset (real RID layout or synthetic layout) and create reproducible splits.

Usage:
    python scripts/prepare_data.py --config configs/unet.yaml
    python scripts/prepare_data.py --root data/processed/synthetic --splits-dir data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import DatasetDiscoveryError, discover_dataset_root, list_sample_ids
from src.data.splits import make_splits, save_splits
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="YAML config providing dataset.root / dataset.splits_dir.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        root = args.root or cfg.dataset.root
        splits_dir = args.splits_dir or cfg.dataset.splits_dir
        seed = cfg.get("seed", args.seed)
    else:
        root = args.root or "data/processed/synthetic"
        splits_dir = args.splits_dir or "data/splits"
        seed = args.seed

    logger.info(f"Discovering dataset at '{root}'...")
    try:
        layout = discover_dataset_root(root)
    except DatasetDiscoveryError as exc:
        logger.error(str(exc))
        logger.error("See data/README.md for how to obtain RID or run scripts/generate_demo_data.py.")
        sys.exit(1)

    logger.info(f"Found images dir: {layout['images']}")
    logger.info(f"Found segments mask dir: {layout['segments']}")
    logger.info(f"Found superstructures mask dir: {layout['superstructures']}")

    ids = list_sample_ids(root)
    logger.info(f"Discovered {len(ids)} samples with matching image/mask ids.")

    splits = make_splits(root, train_frac=args.train_frac, val_frac=args.val_frac, seed=seed)
    save_splits(splits, splits_dir)
    for name, split_ids in splits.items():
        logger.info(f"  {name}: {len(split_ids)} samples")
    logger.info(f"Splits written to {Path(splits_dir).resolve()}")


if __name__ == "__main__":
    main()
