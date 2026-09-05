#!/usr/bin/env python
"""Evaluate a trained checkpoint on the test split: metrics JSON + prediction/error figures.

Usage:
    python scripts/evaluate.py --config configs/unet.yaml --checkpoint models/unet_segment_best.pt --task segment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RoofDataset, roof_collate_fn
from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background
from src.data.splits import load_splits
from src.data.transforms import build_eval_transforms, denormalize
from src.evaluation.evaluator import evaluate_model
from src.models.factory import build_model
from src.training.trainer import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import get_logger
from src.visualization.plotting import (
    colorize_segment_mask,
    colorize_superstructure_mask,
    plot_confusion_matrix,
    save_figure,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="segment", choices=["segment", "superstructure"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-qualitative", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    class_names = segment_classes_with_background(cfg.dataset.segment_scheme) if args.task == "segment" else SUPERSTRUCTURE_CLASSES
    mask_key = "segment_mask" if args.task == "segment" else "superstructure_mask"
    cfg.model["num_classes"] = len(class_names)

    model = build_model(cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val mIoU={checkpoint['val_summary']['mean_iou']:.4f}).")

    splits = load_splits(cfg.dataset.splits_dir)
    eval_tf = build_eval_transforms(cfg.dataset.image_size)
    dataset = RoofDataset(cfg.dataset.root, splits[args.split], transform=eval_tf, segment_scheme=cfg.dataset.segment_scheme)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=roof_collate_fn)

    logger.info(f"Evaluating on '{args.split}' split ({len(dataset)} samples)...")
    metrics = evaluate_model(model, loader, len(class_names), class_names, mask_key=mask_key, device=device)

    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics" / f"{cfg.model.architecture}_{args.task}_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Overall: mIoU={metrics['overall']['mean_iou']:.4f} | Dice={metrics['overall']['mean_dice']:.4f} | "
                f"Precision={metrics['overall']['mean_precision']:.4f} | Recall={metrics['overall']['mean_recall']:.4f} | "
                f"PixelAcc={metrics['overall']['pixel_accuracy']:.4f}")
    logger.info(f"Metrics written to {metrics_path}")

    fig = plot_confusion_matrix(metrics["confusion_matrix"], class_names)
    save_figure(fig, out_dir / "figures" / f"{cfg.model.architecture}_{args.task}_confusion_matrix.png")

    model.eval()
    colorize = colorize_segment_mask if args.task == "segment" else lambda m, *_: colorize_superstructure_mask(m)
    n_shown = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch[mask_key]
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(images.shape[0]):
                if n_shown >= args.num_qualitative:
                    break
                img_disp = denormalize(images[i].cpu().numpy())
                gt = targets[i].numpy()
                pred = preds[i]

                gt_color = colorize(gt, cfg.dataset.segment_scheme) if args.task == "segment" else colorize_superstructure_mask(gt)
                pred_color = colorize(pred, cfg.dataset.segment_scheme) if args.task == "segment" else colorize_superstructure_mask(pred)

                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, im, name in zip(axes, [img_disp, gt_color, pred_color], ["Input", "Ground truth", "Prediction"]):
                    ax.imshow(im)
                    ax.set_title(name)
                    ax.axis("off")
                error = (gt != pred).astype(np.uint8)
                axes[3].imshow(error, cmap="Reds", vmin=0, vmax=1)
                axes[3].set_title(f"Error ({error.mean() * 100:.1f}%)")
                axes[3].axis("off")
                fig.tight_layout()
                save_figure(fig, out_dir / "predictions" / f"{cfg.model.architecture}_{args.task}_{batch['sample_id'][i]}.png")
                n_shown += 1
            if n_shown >= args.num_qualitative:
                break

    logger.info(f"Saved {n_shown} qualitative prediction figures to {out_dir / 'predictions'}")


if __name__ == "__main__":
    main()
