"""Generic segmentation trainer: device/CPU fallback, AMP, checkpointing, early stopping, LR scheduling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import SegmentationMetrics
from src.utils.device import get_device, mixed_precision_available
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_miou: list[float] = field(default_factory=list)
    val_dice: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_time_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best + self.min_delta)
            or (self.mode == "min" and value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class Trainer:
    """Trains a single-task (segment OR superstructure) segmentation model.

    ``mask_key`` selects which mask in each batch dict is the supervision target
    ("segment_mask" or "superstructure_mask"), so the same trainer serves both tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        mask_key: str = "segment_mask",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        early_stopping_patience: int = 10,
        checkpoint_dir: str | Path = "models",
        checkpoint_name: str = "best_model.pt",
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.criterion = criterion
        self.num_classes = num_classes
        self.mask_key = mask_key
        self.epochs = epochs
        self.seed = seed

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=max(2, early_stopping_patience // 3)
        )
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")

        self.use_amp = mixed_precision_available(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name

        self.history = TrainingHistory()

        logger.info(f"Trainer initialized on device={self.device}, amp={self.use_amp}, mask_key={mask_key}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, SegmentationMetrics]:
        self.model.train(mode=train)
        total_loss = 0.0
        n_batches = 0
        metrics = SegmentationMetrics(self.num_classes)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch[self.mask_key].to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                metrics.update(preds.detach().cpu(), targets.detach().cpu())

        return total_loss / max(n_batches, 1), metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        logger.info(f"Starting training for up to {self.epochs} epochs "
                    f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples).")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_metrics = self._run_epoch(val_loader, train=False)
            val_summary = val_metrics.compute()
            elapsed = time.time() - t0

            self.scheduler.step(val_summary["mean_iou"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_miou.append(val_summary["mean_iou"])
            self.history.val_dice.append(val_summary["mean_dice"])
            self.history.lr.append(current_lr)
            self.history.epoch_time_s.append(elapsed)

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_mIoU={val_summary['mean_iou']:.4f} | "
                f"val_Dice={val_summary['mean_dice']:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            improved = self.early_stopping.step(val_summary["mean_iou"])
            if improved:
                self.save_checkpoint(epoch, val_summary)
            if self.early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch} (best mIoU={self.early_stopping.best:.4f}).")
                break

        self._save_history()
        return self.history

    def save_checkpoint(self, epoch: int, val_summary: dict[str, Any]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_summary": val_summary,
                "num_classes": self.num_classes,
                "mask_key": self.mask_key,
                "seed": self.seed,
            },
            self.checkpoint_path,
        )
        logger.info(f"Saved best checkpoint to {self.checkpoint_path} (epoch {epoch}, mIoU={val_summary['mean_iou']:.4f}).")

    def _save_history(self) -> None:
        history_path = self.checkpoint_dir / f"{self.checkpoint_path.stem}_history.json"
        history_path.write_text(json.dumps(self.history.to_dict(), indent=2), encoding="utf-8")


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device | None = None) -> dict[str, Any]:
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
