#!/usr/bin/env python
"""Experiment framework comparing three pipeline configurations on the test split:

    Experiment A: 2D roof segmentation only
        roof area from the predicted segment mask; usable area == roof area (no geometric
        refinement); panel count via the naive area/panel_area ratio (the baseline this whole
        project argues against, included here specifically to quantify how much it overstates).
    Experiment B: roof segmentation + geometric reasoning
        usable area = roof polygon eroded by the safety margin (real polygon geometry); panel
        count from real rectangle placement on that polygon; obstacles are NOT removed.
    Experiment C: roof segmentation + geometric reasoning + obstacle detection
        the full pipeline (src/pipeline.py): usable area additionally subtracts detected
        obstacle polygons (+ clearance); panel placement avoids them.

All three experiments run the model(s) actually trained by scripts/train.py on the actual test
split; nothing here is fabricated. A "reference" (oracle) value for each quantity is computed
the same way as Experiment C but from GROUND-TRUTH masks instead of predictions — this is an
internally consistent experimental reference (perfect segmentation + perfect obstacle
detection), not an independently validated real-world ground truth, and is documented as such.

Usage:
    python scripts/run_experiment.py --config configs/unet.yaml \
        --segment-checkpoint models/unet_segment_best.pt \
        --superstructure-checkpoint models/unet_superstructure_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.dataset import RoofDataset
from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background
from src.data.splits import load_splits
from src.evaluation.evaluator import evaluate_model
from src.geometry.orientation import estimate_orientation
from src.geometry.polygons import mask_to_polygons, union_polygons
from src.geometry.roof_area import estimate_area
from src.models.factory import build_model
from src.optimization.capacity import estimate_capacity
from src.optimization.panel_placement import (
    optimize_placement_multi_orientation,
    panel_count_naive_area_ratio,
)
from src.pipeline import SolarConfig, orientation_row_angle, run_pipeline
from src.training.trainer import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _roof_polygon_from_mask(mask: np.ndarray, num_classes: int):
    polys = []
    for cls_id in range(1, num_classes):
        polys.extend(mask_to_polygons(mask, cls_id))
    return union_polygons(polys)


def _experiment_a(segment_mask: np.ndarray, num_seg_classes: int, gsd: float, solar: SolarConfig) -> dict:
    roof_polygon = _roof_polygon_from_mask(segment_mask, num_seg_classes)
    roof_area = estimate_area(float(roof_polygon.area), gsd)
    panel_w_px, panel_h_px = solar.panel_width_m / gsd, solar.panel_height_m / gsd
    num_panels = int(panel_count_naive_area_ratio(roof_polygon, panel_w_px, panel_h_px))
    capacity = estimate_capacity(max(num_panels, 0), solar.panel_power_w)
    return {"roof_area_m2": roof_area.area_m2, "usable_area_m2": roof_area.area_m2,
            "num_panels": num_panels, "capacity_kwp": capacity.capacity_kwp}


def _experiment_b(segment_mask: np.ndarray, num_seg_classes: int, gsd: float, solar: SolarConfig) -> dict:
    roof_polygon = _roof_polygon_from_mask(segment_mask, num_seg_classes)
    roof_area = estimate_area(float(roof_polygon.area), gsd)
    margin_px = solar.roof_edge_margin_m / gsd
    eroded = roof_polygon.buffer(-margin_px) if margin_px > 0 else roof_polygon
    usable_area = estimate_area(float(eroded.area), gsd)

    orientation = estimate_orientation(segment_mask, solar.segment_scheme)
    row_angle = orientation_row_angle(orientation)
    panel_w_px, panel_h_px = solar.panel_width_m / gsd, solar.panel_height_m / gsd
    spacing_px = solar.spacing_m / gsd
    placement = optimize_placement_multi_orientation(eroded, panel_w_px, panel_h_px, spacing_px, base_orientation_deg=row_angle)
    capacity = estimate_capacity(len(placement.panels), solar.panel_power_w)
    return {"roof_area_m2": roof_area.area_m2, "usable_area_m2": usable_area.area_m2,
            "num_panels": len(placement.panels), "capacity_kwp": capacity.capacity_kwp}


def _experiment_c(segment_mask: np.ndarray, superstructure_mask: np.ndarray, gsd: float, solar: SolarConfig) -> dict:
    dummy_image = np.zeros((*segment_mask.shape, 3), dtype=np.uint8)
    result = run_pipeline(dummy_image, solar_config=solar, gsd_m_per_px=gsd,
                           segment_mask=segment_mask, superstructure_mask=superstructure_mask)
    return {"roof_area_m2": result.usable.roof_area.area_m2, "usable_area_m2": result.usable.usable_area.area_m2,
            "num_panels": len(result.placement.panels), "capacity_kwp": result.capacity.capacity_kwp}


def _rel_error(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return abs(value - reference) / reference * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/unet.yaml")
    parser.add_argument("--segment-checkpoint", type=str, default="models/unet_segment_best.pt")
    parser.add_argument("--superstructure-checkpoint", type=str, default="models/unet_superstructure_best.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    seg_classes = segment_classes_with_background(cfg.dataset.segment_scheme)
    obs_classes = SUPERSTRUCTURE_CLASSES

    seg_ckpt_path = Path(args.segment_checkpoint)
    obs_ckpt_path = Path(args.superstructure_checkpoint)
    if not seg_ckpt_path.exists():
        logger.error(f"Segment checkpoint not found at {seg_ckpt_path}. Run scripts/train.py first "
                      f"(python scripts/train.py --config {args.config} --task segment). Aborting -- "
                      f"experiment results will NOT be fabricated.")
        sys.exit(1)

    cfg_seg = load_config(args.config)
    cfg_seg.model["num_classes"] = len(seg_classes)
    segment_model = build_model(cfg_seg)
    seg_checkpoint_info = load_checkpoint(segment_model, seg_ckpt_path, device=device)
    logger.info(f"Loaded segment model (epoch {seg_checkpoint_info['epoch']}, "
                f"val mIoU={seg_checkpoint_info['val_summary']['mean_iou']:.4f}).")

    superstructure_model = None
    if obs_ckpt_path.exists():
        cfg_obs = load_config(args.config)
        cfg_obs.model["num_classes"] = len(obs_classes)
        superstructure_model = build_model(cfg_obs)
        obs_checkpoint_info = load_checkpoint(superstructure_model, obs_ckpt_path, device=device)
        logger.info(f"Loaded superstructure model (epoch {obs_checkpoint_info['epoch']}, "
                    f"val mIoU={obs_checkpoint_info['val_summary']['mean_iou']:.4f}).")
    else:
        logger.warning(f"No superstructure checkpoint at {obs_ckpt_path}. Experiment C will run "
                        f"with an all-background (no obstacles detected) fallback, and this is "
                        f"reported explicitly rather than silently substituted.")

    splits = load_splits(cfg.dataset.splits_dir)
    test_ids = splits["test"] if args.max_samples is None else splits["test"][: args.max_samples]
    dataset = RoofDataset(cfg.dataset.root, test_ids, transform=None, segment_scheme=cfg.dataset.segment_scheme)
    logger.info(f"Running experiments on {len(dataset)} test samples.")

    solar = SolarConfig(
        panel_width_m=cfg.solar.panel_width_m, panel_height_m=cfg.solar.panel_height_m,
        panel_power_w=cfg.solar.panel_power_w, spacing_m=cfg.solar.spacing_m,
        roof_edge_margin_m=cfg.solar.roof_edge_margin_m, obstacle_clearance_m=cfg.solar.obstacle_clearance_m,
        segment_scheme=cfg.dataset.segment_scheme,
    )

    from src.data.transforms import build_eval_transforms
    eval_tf = build_eval_transforms(cfg.dataset.image_size)

    results_per_experiment: dict[str, list[dict]] = {"A": [], "B": [], "C": []}

    segment_model.eval()
    if superstructure_model is not None:
        superstructure_model.eval()

    for i in range(len(dataset)):
        raw_item = dataset[i]
        image_np = (raw_item["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gt_segment = raw_item["segment_mask"].numpy()
        gt_superstructure = raw_item["superstructure_mask"].numpy()
        metadata = raw_item["metadata"]
        gsd = metadata.get("gsd_m_per_px", cfg.dataset.gsd_m_per_px)

        transformed = eval_tf(image=image_np, mask=gt_segment, mask2=gt_superstructure)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        gt_seg_resized = transformed["mask"].numpy()
        gt_obs_resized = transformed["mask2"].numpy()

        with torch.no_grad():
            seg_logits = segment_model(image_tensor)
        pred_segment = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if superstructure_model is not None:
            with torch.no_grad():
                obs_logits = superstructure_model(image_tensor)
            pred_superstructure = torch.argmax(obs_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            pred_superstructure = np.zeros_like(pred_segment)

        reference = _experiment_c(gt_seg_resized.astype(np.uint8), gt_obs_resized.astype(np.uint8), gsd, solar)

        exp_a = _experiment_a(pred_segment, len(seg_classes), gsd, solar)
        exp_b = _experiment_b(pred_segment, len(seg_classes), gsd, solar)
        exp_c = _experiment_c(pred_segment, pred_superstructure, gsd, solar)

        for key, value in (("A", exp_a), ("B", exp_b), ("C", exp_c)):
            row = dict(value)
            row["sample_id"] = raw_item["sample_id"]
            for metric in ("roof_area_m2", "usable_area_m2", "num_panels", "capacity_kwp"):
                row[f"{metric}_error_pct"] = _rel_error(value[metric], reference[metric])
            results_per_experiment[key].append(row)

    logger.info("Evaluating segmentation quality (IoU/Dice/precision/recall) on the test split...")
    from torch.utils.data import DataLoader

    from src.data.dataset import roof_collate_fn

    eval_dataset = RoofDataset(cfg.dataset.root, test_ids, transform=eval_tf, segment_scheme=cfg.dataset.segment_scheme)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=roof_collate_fn)
    seg_metrics = evaluate_model(segment_model, eval_loader, len(seg_classes), seg_classes, mask_key="segment_mask", device=device)
    obs_metrics = None
    if superstructure_model is not None:
        obs_metrics = evaluate_model(superstructure_model, eval_loader, len(obs_classes), obs_classes, mask_key="superstructure_mask", device=device)

    summary: dict[str, dict] = {}
    for key, rows in results_per_experiment.items():
        summary[key] = {
            metric: {
                "mean_absolute_pct_error": float(np.mean([r[f"{metric}_error_pct"] for r in rows if np.isfinite(r[f"{metric}_error_pct"])]))
            }
            for metric in ("roof_area_m2", "usable_area_m2", "num_panels", "capacity_kwp")
        }

    experiment_descriptions = {
        "A": "2D roof segmentation only (naive area/panel_area panel count, no geometric reasoning)",
        "B": "Roof segmentation + geometric reasoning (real placement, edge margin, no obstacle removal)",
        "C": "Roof segmentation + geometric reasoning + obstacle detection (full pipeline)",
    }

    report = {
        "num_test_samples": len(dataset),
        "segmentation_metrics_roof_segments": seg_metrics["overall"],
        "segmentation_metrics_superstructures": obs_metrics["overall"] if obs_metrics else "not available (no superstructure checkpoint)",
        "experiment_descriptions": experiment_descriptions,
        "mean_absolute_pct_error_vs_reference": summary,
        "note": (
            "The 'reference' is Experiment C computed on GROUND-TRUTH masks (oracle segmentation "
            "+ oracle obstacle detection), not an independently measured real-world ground truth. "
            "It isolates how much error each experiment's *modeling/geometric assumptions* "
            "contribute, holding the underlying annotation quality fixed."
        ),
    }

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "experiment_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key, rows in results_per_experiment.items():
        import csv
        with (metrics_dir / f"experiment_{key}_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    logger.info(f"Experiment comparison written to {metrics_dir / 'experiment_comparison.json'}")
    for key in ("A", "B", "C"):
        logger.info(f"Experiment {key} ({experiment_descriptions[key]}):")
        for metric, stats in summary[key].items():
            logger.info(f"    {metric}: mean abs % error vs reference = {stats['mean_absolute_pct_error']:.1f}%")

    _plot_comparison(summary, experiment_descriptions, out_dir / "figures" / "experiment_comparison.png")


def _plot_comparison(summary: dict, descriptions: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["roof_area_m2", "usable_area_m2", "num_panels", "capacity_kwp"]
    experiments = ["A", "B", "C"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, exp in enumerate(experiments):
        values = [summary[exp][m]["mean_absolute_pct_error"] for m in metrics]
        ax.bar(x + i * width, values, width, label=f"Exp {exp}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Roof area", "Usable area", "Panel count", "Capacity"])
    ax.set_ylabel("Mean absolute % error vs. oracle reference")
    ax.set_title("Experiment A/B/C comparison (lower is better)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
