"""Per-pixel segmentation confidence from actual model softmax output.

We use the softmax margin (top-1 probability minus top-2 probability) rather than raw top-1
probability, since margin is a more direct measure of *decision* confidence: a pixel with
top-1=0.4 in a 9-class problem is far more ambiguous if the runner-up is 0.39 than if it is
0.05, even though both have "low" top-1 probability relative to a binary problem. Predictive
entropy is also reported as a complementary, scale-normalized signal. Neither value is
randomly generated — both are deterministic functions of the model's actual output logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SegmentationUncertaintyMap:
    confidence_map: np.ndarray  # (H, W) in [0, 1], softmax margin
    entropy_map: np.ndarray  # (H, W), normalized predictive entropy in [0, 1]
    mean_confidence: float
    mean_entropy: float
    low_confidence_fraction: float  # fraction of predicted-roof pixels with margin < 0.2


def compute_segmentation_uncertainty(logits: torch.Tensor, roof_mask: np.ndarray | None = None) -> SegmentationUncertaintyMap:
    """``logits``: (C, H, W) or (1, C, H, W) raw model output for a single image."""
    if logits.dim() == 4:
        logits = logits.squeeze(0)
    probs = F.softmax(logits, dim=0)
    top2 = torch.topk(probs, k=min(2, probs.shape[0]), dim=0).values
    margin = (top2[0] - top2[-1]) if top2.shape[0] > 1 else top2[0]
    confidence_map = margin.detach().cpu().numpy()

    num_classes = probs.shape[0]
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=0)
    max_entropy = np.log(num_classes)
    entropy_map = (entropy.detach().cpu().numpy() / max_entropy) if max_entropy > 0 else np.zeros_like(confidence_map)

    if roof_mask is not None and roof_mask.any():
        region_conf = confidence_map[roof_mask.astype(bool)]
        region_ent = entropy_map[roof_mask.astype(bool)]
    else:
        region_conf = confidence_map
        region_ent = entropy_map

    low_conf_fraction = float((region_conf < 0.2).mean()) if region_conf.size > 0 else 0.0

    return SegmentationUncertaintyMap(
        confidence_map=confidence_map,
        entropy_map=entropy_map,
        mean_confidence=float(region_conf.mean()) if region_conf.size > 0 else float("nan"),
        mean_entropy=float(region_ent.mean()) if region_ent.size > 0 else float("nan"),
        low_confidence_fraction=low_conf_fraction,
    )
