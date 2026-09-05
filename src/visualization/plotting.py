"""Professional visualization utilities shared by scripts, notebooks and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data.label_maps import (
    SEGMENT_PALETTE,
    SUPERSTRUCTURE_CLASSES,
    SUPERSTRUCTURE_PALETTE,
    segment_classes_with_background,
)
from src.optimization.panel_placement import PlacementResult


def _palette_array(class_names: list[str], palette) -> np.ndarray:
    return np.array(palette.as_list(class_names), dtype=np.uint8)


def colorize_mask(mask: np.ndarray, class_names: list[str], palette) -> np.ndarray:
    lut = _palette_array(class_names, palette)
    safe_mask = np.clip(mask, 0, len(class_names) - 1)
    return lut[safe_mask]


def colorize_segment_mask(mask: np.ndarray, segment_scheme: int = 9) -> np.ndarray:
    return colorize_mask(mask, segment_classes_with_background(segment_scheme), SEGMENT_PALETTE)


def colorize_superstructure_mask(mask: np.ndarray) -> np.ndarray:
    return colorize_mask(mask, SUPERSTRUCTURE_CLASSES, SUPERSTRUCTURE_PALETTE)


def overlay_mask(image: np.ndarray, color_mask: np.ndarray, background_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Alpha-blend ``color_mask`` onto ``image`` only where ``background_mask`` is nonzero."""
    out = image.copy().astype(np.float32)
    fg = background_mask.astype(bool)
    out[fg] = (1 - alpha) * out[fg] + alpha * color_mask[fg].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def plot_legend(ax: plt.Axes, class_names: list[str], palette) -> None:
    patches = [mpatches.Patch(color=np.array(c) / 255, label=n) for n, c in zip(class_names, palette.as_list(class_names)) if n != "background"]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=min(4, len(patches)), fontsize=8, frameon=False)


def plot_sample_overview(
    image: np.ndarray,
    segment_mask: np.ndarray,
    superstructure_mask: np.ndarray,
    segment_scheme: int = 9,
    title: str | None = None,
) -> Figure:
    """A 1x4 panel: original image, segment mask, superstructure mask, combined overlay."""
    seg_color = colorize_segment_mask(segment_mask, segment_scheme)
    obs_color = colorize_superstructure_mask(superstructure_mask)
    combined = overlay_mask(image, seg_color, segment_mask != 0, alpha=0.45)
    combined = overlay_mask(combined, obs_color, superstructure_mask != 0, alpha=0.75)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, img, name in zip(
        axes, [image, seg_color, obs_color, combined],
        ["Input image", "Roof segments (orientation)", "Superstructures (obstacles)", "Combined overlay"],
    ):
        ax.imshow(img)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_usable_area(image: np.ndarray, roof_polygon, usable_polygon, obstacle_polygons: list) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _draw_polygon(ax, roof_polygon, edgecolor="cyan", facecolor="none", linewidth=2, label="Roof boundary")
    for poly in obstacle_polygons:
        _draw_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.3, linewidth=1)
    _draw_polygon(ax, usable_polygon, edgecolor="lime", facecolor="lime", alpha=0.25, linewidth=2, label="Usable area")
    ax.set_title("Usable roof area (green) vs. obstacles (red)")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _draw_polygon(ax: plt.Axes, polygon, **kwargs) -> None:
    if polygon is None or polygon.is_empty:
        return
    geoms = polygon.geoms if hasattr(polygon, "geoms") else [polygon]
    for geom in geoms:
        if geom.is_empty:
            continue
        if geom.geom_type == "polygon" : 
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, **{k: v for k, v in kwargs.items() if k != "label"})
            ax.plot(xs, ys, color=kwargs.get("edgecolor", "black"), linewidth=kwargs.get("linewidth", 1),
                 label=kwargs.get("label"))
        elif geom.geom_type == "LineString":
            xs, ys = geom.xy
            ax.plot(
                xs,
                ys,
                color=kwargs.get("edgecolor", "black"),
                linewidth=kwargs.get("linewidth", 1),
                label=kwargs.get("label"),
            )
        elif hasattr(geom, "geoms"):
            _draw_polygon(ax, geom, **kwargs)


def plot_panel_placement(image: np.ndarray, usable_polygon, placement: PlacementResult) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _draw_polygon(ax, usable_polygon, edgecolor="lime", facecolor="lime", alpha=0.15, linewidth=1.5, label="Usable area")
    for panel in placement.panels:
        xs = [c[0] for c in panel.corners_px] + [panel.corners_px[0][0]]
        ys = [c[1] for c in panel.corners_px] + [panel.corners_px[0][1]]
        ax.fill(xs, ys, color="dodgerblue", alpha=0.6, edgecolor="navy", linewidth=0.8)
    ax.set_title(f"Panel placement: {len(placement.panels)} panels "
                 f"(packing efficiency {placement.packing_efficiency * 100:.1f}%)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_error_map(ground_truth: np.ndarray, prediction: np.ndarray) -> Figure:
    """Binary correct/incorrect map, useful for qualitative failure analysis."""
    error = (ground_truth != prediction).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(error, cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"Error map ({error.mean() * 100:.1f}% pixels differ)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix: list[list[int]], class_names: list[str], normalize: bool = True) -> Figure:
    cm = np.array(confusion_matrix, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.6)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
