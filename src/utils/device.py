"""Device selection with automatic CPU fallback.

This project was developed on a machine with an Intel integrated GPU and no CUDA-capable
device, so the default path is CPU. Mixed precision (``torch.autocast``) is only ever enabled
when CUDA is actually available, since it is unsupported/unhelpful on plain CPU.
"""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixed_precision_available(device: torch.device) -> bool:
    return device.type == "cuda"
