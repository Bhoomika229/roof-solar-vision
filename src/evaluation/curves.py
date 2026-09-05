"""Training/validation curve plotting from a saved TrainingHistory JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history_path: str | Path, out_path: str | Path) -> None:
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_miou"], color="tab:green", label="val mIoU")
    axes[1].plot(epochs, history["val_dice"], color="tab:orange", label="val Dice")
    axes[1].set_title("Validation Segmentation Quality")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], color="tab:red")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
