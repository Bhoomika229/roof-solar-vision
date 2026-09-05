#!/usr/bin/env python
"""Train a segmentation model (U-Net or SegFormer) for the roof-segment or superstructure task.

Usage:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/segformer.yaml --task superstructure
    python scripts/train.py --config configs/unet.yaml --epochs 2 --max-samples 8  # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader, Subset

from src.data.dataset import RoofDataset, roof_collate_fn
from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background
from src.data.splits import load_splits
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.models.factory import build_model
from src.models.losses import DiceCrossEntropyLoss
from src.training.trainer import Trainer
from src.utils.config import Config, load_config
from src.utils.device import get_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_dataloaders(cfg: Config, task: str, max_samples: int | None = None) -> tuple[DataLoader, DataLoader, int, list[str]]:
    splits = load_splits(cfg.dataset.splits_dir)
    train_tf = build_train_transforms(cfg.dataset.image_size, augment=cfg.augmentation.get("enabled", True))
    eval_tf = build_eval_transforms(cfg.dataset.image_size)

    train_ds = RoofDataset(cfg.dataset.root, splits["train"], transform=train_tf, segment_scheme=cfg.dataset.segment_scheme)
    val_ids = splits["val"] if splits["val"] else splits["train"][:1]
    val_ds = RoofDataset(cfg.dataset.root, val_ids, transform=eval_tf, segment_scheme=cfg.dataset.segment_scheme)

    if max_samples is not None:
        train_ds = Subset(train_ds, list(range(min(max_samples, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(max_samples, len(val_ds)))))

    batch_size = min(cfg.train.batch_size, len(train_ds)) or 1
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.train.get("num_workers", 0), collate_fn=roof_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, len(val_ds)) or 1, shuffle=False,
        num_workers=cfg.train.get("num_workers", 0), collate_fn=roof_collate_fn,
    )

    if task == "segment":
        class_names = segment_classes_with_background(cfg.dataset.segment_scheme)
    else:
        class_names = SUPERSTRUCTURE_CLASSES

    return train_loader, val_loader, len(class_names), class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, choices=["segment", "superstructure"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size (smoke tests).")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = args.task or cfg.train.get("task", "segment")
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    logger.info(f"Loaded config from {args.config} | task={task} | architecture={cfg.model.architecture}")

    train_loader, val_loader, num_classes, class_names = build_dataloaders(cfg, task, args.max_samples)
    logger.info(f"Task '{task}' has {num_classes} classes: {class_names}")

    cfg.model["num_classes"] = num_classes
    model = build_model(cfg)

    criterion = DiceCrossEntropyLoss(
        num_classes=num_classes,
        ce_weight=cfg.train.get("ce_weight", 0.5),
        dice_weight=cfg.train.get("dice_weight", 0.5),
    )

    mask_key = "segment_mask" if task == "segment" else "superstructure_mask"
    checkpoint_name = args.checkpoint_name or f"{cfg.model.architecture}_{task}_best.pt"

    trainer = Trainer(
        model=model,
        criterion=criterion,
        num_classes=num_classes,
        mask_key=mask_key,
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 1e-4),
        epochs=cfg.train.epochs,
        early_stopping_patience=cfg.train.get("early_stopping_patience", 8),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=checkpoint_name,
        device=get_device(),
        seed=cfg.get("seed", 42),
    )

    trainer.fit(train_loader, val_loader)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
