#!/usr/bin/env python
"""Generate a synthetic roof-image dataset so the full pipeline runs without RID.

Usage:
    python scripts/generate_demo_data.py --num-samples 200 --image-size 256 --seed 42

Writes to ``data/processed/synthetic/`` (or ``--out-dir``):
    images/{id}.png
    masks_segments/{id}.png
    masks_superstructures/{id}.png
    metadata/{id}.json
    index.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import cv2

from src.data.synthetic import SyntheticRoofGenerator
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--segment-scheme", type=int, default=9, choices=[5, 9, 17])
    parser.add_argument("--gsd", type=float, default=0.10, help="Synthetic ground-sampling distance (m/px).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/processed/synthetic")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    segments_dir = out_dir / "masks_segments"
    superstructures_dir = out_dir / "masks_superstructures"
    metadata_dir = out_dir / "metadata"
    for d in (images_dir, segments_dir, superstructures_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    generator = SyntheticRoofGenerator(
        image_size=args.image_size,
        segment_scheme=args.segment_scheme,
        gsd_m_per_px=args.gsd,
        seed=args.seed,
    )

    index_rows = []
    for i in range(args.num_samples):
        sample_id = f"synthetic_{i:05d}"
        sample = generator.generate(sample_id)

        cv2.imwrite(str(images_dir / f"{sample_id}.png"), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(segments_dir / f"{sample_id}.png"), sample.segment_mask)
        cv2.imwrite(str(superstructures_dir / f"{sample_id}.png"), sample.superstructure_mask)
        (metadata_dir / f"{sample_id}.json").write_text(json.dumps(sample.metadata, indent=2), encoding="utf-8")

        index_rows.append({
            "sample_id": sample_id,
            "archetype": sample.metadata["archetype"],
            "gsd_m_per_px": sample.metadata["gsd_m_per_px"],
            "num_obstacles": len(sample.metadata["obstacles"]),
        })

        if (i + 1) % 50 == 0 or (i + 1) == args.num_samples:
            logger.info(f"Generated {i + 1}/{args.num_samples} synthetic samples.")

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    logger.info(f"Synthetic dataset written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
