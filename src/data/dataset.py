"""PyTorch dataset for roof segmentation, with RID-compatible discovery and a synthetic backend.

Two on-disk layouts are supported, chosen automatically by :func:`discover_dataset_root`:

1. **RID layout** (after the user has run ``scripts/prepare_data.py`` on a real, manually
   downloaded copy of RID — see ``data/README.md``): a directory containing
   ``images_roof_centered_png/``, ``masks_segments/`` and ``masks_superstructures/``
   sub-folders of same-named raster files, mirroring RID's own ``definitions.py`` naming.
2. **Synthetic layout** (produced by ``scripts/generate_demo_data.py``): a directory with
   ``images/``, ``masks_segments/``, ``masks_superstructures/`` and ``metadata/`` sub-folders,
   plus an ``index.csv``.

Both expose the same ``RoofDataset`` interface so training/evaluation code never has to know
which backend it is reading from. Filenames are *discovered*, never hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background

IMAGE_DIR_CANDIDATES = ["images", "images_roof_centered_png", "images_roof_centered_geotiff"]
SEGMENT_MASK_DIR_CANDIDATES = ["masks_segments"]
SUPERSTRUCTURE_MASK_DIR_CANDIDATES = ["masks_superstructures", "masks_superstructures_reviewed", "masks_superstructures_initial"]


class DatasetDiscoveryError(RuntimeError):
    pass


def _find_existing(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def discover_dataset_root(root: str | Path) -> dict[str, Path]:
    """Locate the image/mask sub-directories under ``root`` without assuming exact names.

    Raises :class:`DatasetDiscoveryError` with an actionable message if the layout does not
    match either supported convention.
    """
    root = Path(root)
    if not root.exists():
        raise DatasetDiscoveryError(
            f"Dataset root '{root}' does not exist. See data/README.md for how to obtain or "
            f"generate a dataset (real RID or `python scripts/generate_demo_data.py`)."
        )

    images_dir = _find_existing(root, IMAGE_DIR_CANDIDATES)
    segments_dir = _find_existing(root, SEGMENT_MASK_DIR_CANDIDATES)
    superstructures_dir = _find_existing(root, SUPERSTRUCTURE_MASK_DIR_CANDIDATES)

    if images_dir is None:
        raise DatasetDiscoveryError(
            f"No image directory found under '{root}'. Expected one of {IMAGE_DIR_CANDIDATES}."
        )
    if segments_dir is None and superstructures_dir is None:
        raise DatasetDiscoveryError(
            f"No mask directories found under '{root}'. Expected at least one of "
            f"{SEGMENT_MASK_DIR_CANDIDATES + SUPERSTRUCTURE_MASK_DIR_CANDIDATES}."
        )

    metadata_dir = root / "metadata" if (root / "metadata").is_dir() else None
    return {
        "root": root,
        "images": images_dir,
        "segments": segments_dir,
        "superstructures": superstructures_dir,
        "metadata": metadata_dir,
    }


def _stem_index(directory: Path | None) -> dict[str, Path]:
    if directory is None:
        return {}
    return {p.stem: p for p in directory.iterdir() if p.is_file()}


def list_sample_ids(root: str | Path) -> list[str]:
    """Return sample ids present across image + (available) mask directories, sorted."""
    layout = discover_dataset_root(root)
    images_idx = _stem_index(layout["images"])
    ids = set(images_idx)
    if layout["segments"] is not None:
        ids &= set(_stem_index(layout["segments"]))
    if layout["superstructures"] is not None:
        ids &= set(_stem_index(layout["superstructures"]))
    if not ids:
        raise DatasetDiscoveryError(
            f"Images and masks under '{root}' do not share any common sample ids (filenames "
            f"without extension must match across images/segments/superstructures)."
        )
    return sorted(ids)


class RoofDataset(Dataset):
    """A single dataset item yields ``image``, ``segment_mask``, ``superstructure_mask``.

    Masks are returned as ``LongTensor`` class-id maps of shape (H, W). If a mask type is not
    available on disk for a given root, an all-zero (background) mask of the requested target
    task's class count is substituted so both single- and multi-task training loops can rely on
    consistent tensor shapes.
    """

    def __init__(
        self,
        root: str | Path,
        sample_ids: list[str] | None = None,
        transform: Callable | None = None,
        segment_scheme: int = 9,
    ) -> None:
        self.layout = discover_dataset_root(root)
        self.transform = transform
        self.segment_scheme = segment_scheme
        self.segment_classes = segment_classes_with_background(segment_scheme)
        self.superstructure_classes = SUPERSTRUCTURE_CLASSES

        self.images_idx = _stem_index(self.layout["images"])
        self.segments_idx = _stem_index(self.layout["segments"])
        self.superstructures_idx = _stem_index(self.layout["superstructures"])

        self.sample_ids = sample_ids if sample_ids is not None else list_sample_ids(root)
        missing = [sid for sid in self.sample_ids if sid not in self.images_idx]
        if missing:
            raise DatasetDiscoveryError(f"{len(missing)} requested sample ids have no image file, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise DatasetDiscoveryError(f"Failed to read image file: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: Path | None, index: dict[str, Path], sample_id: str, height: int, width: int) -> np.ndarray:
        if path is None or sample_id not in index:
            return np.zeros((height, width), dtype=np.uint8)
        mask_path = index[sample_id]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise DatasetDiscoveryError(f"Failed to read mask file: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_id = self.sample_ids[idx]
        image = self._load_image(self.images_idx[sample_id])
        h, w = image.shape[:2]

        segment_mask = self._load_mask(self.layout["segments"], self.segments_idx, sample_id, h, w)
        superstructure_mask = self._load_mask(self.layout["superstructures"], self.superstructures_idx, sample_id, h, w)

        metadata: dict[str, Any] = {}
        if self.layout["metadata"] is not None:
            meta_path = self.layout["metadata"] / f"{sample_id}.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=segment_mask, mask2=superstructure_mask)
            image_t = transformed["image"]
            segment_mask_t = torch.as_tensor(transformed["mask"], dtype=torch.long)
            superstructure_mask_t = torch.as_tensor(transformed["mask2"], dtype=torch.long)
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            segment_mask_t = torch.from_numpy(segment_mask).long()
            superstructure_mask_t = torch.from_numpy(superstructure_mask).long()

        return {
            "sample_id": sample_id,
            "image": image_t,
            "segment_mask": segment_mask_t,
            "superstructure_mask": superstructure_mask_t,
            "metadata": metadata,
        }


def roof_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate: stack tensor fields normally, keep ragged fields (id/metadata) as lists.

    ``metadata`` dicts have per-sample-varying structure (different numbers of obstacles etc.),
    which torch's default collate cannot stack — so it is left as a plain Python list.
    """
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "segment_mask": torch.stack([item["segment_mask"] for item in batch]),
        "superstructure_mask": torch.stack([item["superstructure_mask"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
