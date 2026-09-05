#!/usr/bin/env python
"""Run the full roof-analysis pipeline on a single image and print/save a report.

Usage:
    python scripts/predict.py --image path/to/roof.jpg --gsd 0.1 \
        --segment-checkpoint models/unet_segment_best.pt \
        --superstructure-checkpoint models/unet_superstructure_best.pt

If no checkpoints are given (or found), falls back to synthetic-demo mode: generates one
synthetic sample and analyzes its ground-truth masks directly, so the command is always
runnable even before any model has been trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background
from src.data.synthetic import SyntheticRoofGenerator
from src.models.factory import build_model
from src.pipeline import SolarConfig, run_pipeline
from src.training.trainer import load_checkpoint
from src.utils.config import Config
from src.utils.device import get_device
from src.utils.logging import get_logger
from src.visualization.plotting import (
    plot_panel_placement,
    plot_sample_overview,
    plot_usable_area,
    save_figure,
)

logger = get_logger(__name__)


def _load_model_for_checkpoint(checkpoint_path: str, architecture: str, num_classes: int, image_size: int):
    cfg = Config({"model": {"architecture": architecture, "num_classes": num_classes, "base_channels": 32, "depth": 4},
                  "dataset": {"image_size": image_size}})
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path, device=get_device())
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None, help="Path to an aerial image. Omit for synthetic demo mode.")
    parser.add_argument("--gsd", type=float, default=None, help="Ground sampling distance in meters/pixel. If omitted, an assumed default is used and clearly labeled.")
    parser.add_argument("--segment-checkpoint", type=str, default="models/unet_segment_best.pt")
    parser.add_argument("--superstructure-checkpoint", type=str, default="models/unet_superstructure_best.pt")
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "segformer"])
    parser.add_argument("--segment-scheme", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--panel-width-m", type=float, default=1.0)
    parser.add_argument("--panel-height-m", type=float, default=1.7)
    parser.add_argument("--panel-power-w", type=float, default=450.0)
    parser.add_argument("--spacing-m", type=float, default=0.02)
    parser.add_argument("--roof-edge-margin-m", type=float, default=0.3)
    parser.add_argument("--obstacle-clearance-m", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="results/predictions")
    args = parser.parse_args()

    solar_config = SolarConfig(
        panel_width_m=args.panel_width_m, panel_height_m=args.panel_height_m,
        panel_power_w=args.panel_power_w, spacing_m=args.spacing_m,
        roof_edge_margin_m=args.roof_edge_margin_m, obstacle_clearance_m=args.obstacle_clearance_m,
        segment_scheme=args.segment_scheme,
    )

    segment_mask = None
    superstructure_mask = None
    segment_model = None
    superstructure_model = None
    sample_id = "input_image"

    if args.image is not None:
        image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {args.image}")
        sample_id = Path(args.image).stem

        seg_ckpt = Path(args.segment_checkpoint)
        obs_ckpt = Path(args.superstructure_checkpoint)
        if seg_ckpt.exists():
            n_seg = len(segment_classes_with_background(args.segment_scheme))
            segment_model = _load_model_for_checkpoint(str(seg_ckpt), args.architecture, n_seg, args.image_size)
            logger.info(f"Loaded segment-task checkpoint from {seg_ckpt}")
        else:
            logger.warning(f"No segment checkpoint found at {seg_ckpt}. Run scripts/train.py first, or "
                            f"analyze a synthetic sample instead (omit --image).")
            sys.exit(1)

        if obs_ckpt.exists():
            n_obs = len(SUPERSTRUCTURE_CLASSES)
            superstructure_model = _load_model_for_checkpoint(str(obs_ckpt), args.architecture, n_obs, args.image_size)
            logger.info(f"Loaded superstructure-task checkpoint from {obs_ckpt}")
        else:
            logger.warning(f"No superstructure checkpoint found at {obs_ckpt}. Continuing without obstacle detection.")
    else:
        logger.info("No --image given: running synthetic-demo mode (ground-truth masks, no model inference).")
        generator = SyntheticRoofGenerator(image_size=args.image_size, segment_scheme=args.segment_scheme, gsd_m_per_px=args.gsd or 0.10)
        sample = generator.generate("demo_sample")
        image = sample.image
        segment_mask = sample.segment_mask
        superstructure_mask = sample.superstructure_mask
        sample_id = sample.sample_id
        if args.gsd is None:
            args.gsd = sample.metadata["gsd_m_per_px"]

    result = run_pipeline(
        image, solar_config=solar_config, gsd_m_per_px=args.gsd,
        segment_mask=segment_mask, superstructure_mask=superstructure_mask,
        segment_model=segment_model, superstructure_model=superstructure_model,
        model_image_size=args.image_size,
    )

    print("\n".join(result.report_lines()))
    print(f"\n{result.uncertainty.explanation}")

    out_dir = Path(args.out_dir)
    fig1 = plot_sample_overview(image, result.segment_mask, result.superstructure_mask, args.segment_scheme, title=sample_id)
    save_figure(fig1, out_dir / f"{sample_id}_overview.png")

    fig2 = plot_usable_area(image, None, result.usable.usable_polygon, sum(result.obstacle_polygons_by_class.values(), []))
    save_figure(fig2, out_dir / f"{sample_id}_usable_area.png")

    fig3 = plot_panel_placement(image, result.usable.usable_polygon, result.placement)
    save_figure(fig3, out_dir / f"{sample_id}_panel_placement.png")

    report = {
        "sample_id": sample_id,
        "gsd_m_per_px": result.gsd_m_per_px,
        "gsd_is_assumed": result.gsd_is_assumed,
        "roof_area_m2": result.usable.roof_area.area_m2,
        "usable_area_m2": result.usable.usable_area.area_m2,
        "obstacle_area_m2": result.usable.obstacle_area.area_m2,
        "orientation": result.orientation.dominant_direction,
        "pitch_estimate_deg": result.pitch.pitch_deg_typical,
        "pitch_range_deg": [result.pitch.pitch_deg_low, result.pitch.pitch_deg_high],
        "num_panels": len(result.placement.panels),
        "capacity_kwp": result.capacity.capacity_kwp,
        "uncertainty_range": [result.uncertainty.panel_count_low, result.uncertainty.panel_count_high],
        "uncertainty_level": result.uncertainty.uncertainty_level,
    }
    (out_dir / f"{sample_id}_report.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sample_id}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved figures and report to {out_dir}")


if __name__ == "__main__":
    main()
