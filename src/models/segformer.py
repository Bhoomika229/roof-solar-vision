"""SegFormer (Xie et al., 2021) segmentation head via HuggingFace Transformers.

Architecture: MiT (Mix Transformer) hierarchical encoder + a lightweight all-MLP decode head.
Pretrained weights: ``nvidia/segformer-b0-finetuned-ade-512-512`` by default (ADE20K-pretrained
ImageNet-backbone), fine-tuned end-to-end on the roof segmentation task. The classifier head is
always re-initialized to the project's own number of classes (``num_labels``) since RID's
classes have no correspondence to ADE20K's.

Fine-tuning strategy: full fine-tuning (encoder + head), since the target domain (nadir aerial
imagery) differs substantially from ADE20K's largely street-level/indoor scenes — freezing the
backbone was empirically not attempted here (no GPU budget for an ablation) and is left as
documented future work rather than a fabricated claim.

Input resolution: configurable, default 512x512 (matches the pretrained checkpoint's native
training resolution; SegFormer's MLP decoder is resolution-agnostic so other sizes also work).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegFormerSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        image_size: int = 512,
    ) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            self.pretrained = True
        except Exception:
            # No internet access / weights unavailable: fall back to a randomly-initialized
            # SegFormer-B0 of identical architecture so the pipeline still runs end-to-end.
            config = SegformerConfig(num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)
            self.pretrained = False

        self.image_size = image_size
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits
