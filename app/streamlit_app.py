"""Streamlit demo: upload an aerial roof image (or use synthetic demo mode) and run the full
roof-geometry + solar-potential pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import streamlit as st

from src.data.label_maps import SUPERSTRUCTURE_CLASSES, segment_classes_with_background
from src.data.synthetic import SyntheticRoofGenerator
from src.geometry.orientation import direction_to_compass_label
from src.models.factory import build_model
from src.pipeline import SolarConfig, run_pipeline
from src.training.trainer import load_checkpoint
from src.utils.config import Config
from src.utils.device import get_device
from src.visualization.plotting import plot_panel_placement, plot_sample_overview, plot_usable_area

st.set_page_config(page_title="Roof Solar Potential CV", layout="wide")

MODELS_DIR = Path("models")
DEFAULT_SEGMENT_CKPT = MODELS_DIR / "unet_segment_best.pt"
DEFAULT_SUPERSTRUCTURE_CKPT = MODELS_DIR / "unet_superstructure_best.pt"


@st.cache_resource
def _load_model(checkpoint_path: str, architecture: str, num_classes: int, image_size: int):
    cfg = Config({
        "model": {"architecture": architecture, "num_classes": num_classes, "base_channels": 32, "depth": 4},
        "dataset": {"image_size": image_size},
    })
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path, device=get_device())
    return model


def _discover_checkpoints() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.pt"))


def main() -> None:
    st.title("AI-Based Roof Geometry & Solar Potential Estimation")
    st.caption(
        "Aerial imagery -> deep-learning roof segmentation -> geometric reasoning -> "
        "obstacle-aware solar-panel placement -> capacity & uncertainty estimate. "
        "Portfolio research project; not a certified PV-yield tool."
    )

    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Input mode", ["Synthetic demo", "Upload image"], index=0)

        gsd_known = st.checkbox("I know the Ground Sampling Distance (GSD)", value=True)
        gsd = None
        if gsd_known:
            gsd = st.number_input("GSD (meters/pixel)", min_value=0.01, max_value=2.0, value=0.10, step=0.01)
        else:
            st.info("No GSD provided: physical areas will be computed with a documented "
                    "default assumption and clearly labeled as such.")

        st.subheader("Solar panel configuration")
        panel_width_m = st.number_input("Panel width (m)", 0.3, 3.0, 1.0, 0.05)
        panel_height_m = st.number_input("Panel height (m)", 0.3, 3.0, 1.7, 0.05)
        panel_power_w = st.number_input("Panel power rating (W)", 100, 800, 450, 10)
        spacing_m = st.number_input("Inter-panel spacing (m)", 0.0, 0.5, 0.02, 0.01)
        roof_edge_margin_m = st.number_input("Roof-edge safety margin (m)", 0.0, 2.0, 0.3, 0.05)
        obstacle_clearance_m = st.number_input("Obstacle clearance (m)", 0.0, 2.0, 0.2, 0.05)
        segment_scheme = st.selectbox("Roof-orientation class scheme", [5, 9, 17], index=1)

        st.subheader("Model checkpoints")
        available_ckpts = _discover_checkpoints()
        if available_ckpts:
            ckpt_names = [c.name for c in available_ckpts]
            seg_default_idx = ckpt_names.index(DEFAULT_SEGMENT_CKPT.name) if DEFAULT_SEGMENT_CKPT.name in ckpt_names else 0
            segment_ckpt_name = st.selectbox("Segment-task checkpoint", ckpt_names, index=seg_default_idx)
            obs_options = ["(none - obstacle detection disabled)"] + ckpt_names
            obs_default_idx = obs_options.index(DEFAULT_SUPERSTRUCTURE_CKPT.name) if DEFAULT_SUPERSTRUCTURE_CKPT.name in obs_options else 0
            superstructure_ckpt_name = st.selectbox("Superstructure-task checkpoint", obs_options, index=obs_default_idx)
        else:
            st.warning("No trained checkpoints found in models/. Run `python scripts/train.py "
                       "--config configs/unet.yaml` first, or continue in Synthetic demo mode "
                       "(which uses ground-truth masks and needs no model).")
            segment_ckpt_name = None
            superstructure_ckpt_name = "(none - obstacle detection disabled)"

    solar_config = SolarConfig(
        panel_width_m=panel_width_m, panel_height_m=panel_height_m, panel_power_w=panel_power_w,
        spacing_m=spacing_m, roof_edge_margin_m=roof_edge_margin_m, obstacle_clearance_m=obstacle_clearance_m,
        segment_scheme=segment_scheme,
    )

    image = None
    segment_mask = None
    superstructure_mask = None
    segment_model = None
    superstructure_model = None
    sample_label = "sample"

    if mode == "Synthetic demo":
        seed = st.sidebar.number_input("Synthetic sample seed", 0, 10_000, 0, 1)
        generator = SyntheticRoofGenerator(image_size=256, segment_scheme=segment_scheme, gsd_m_per_px=gsd or 0.10, seed=int(seed))
        sample = generator.generate(f"demo_{seed}")
        image = sample.image
        segment_mask = sample.segment_mask
        superstructure_mask = sample.superstructure_mask
        sample_label = sample.sample_id
        if gsd is None:
            gsd = sample.metadata["gsd_m_per_px"]
        st.info("Synthetic demo mode: ground-truth masks are used directly (no model inference) "
                "so the full pipeline is demonstrable without a trained checkpoint. "
                "See CLAUDE.md / data/README.md for why RID cannot be bundled here.")
    else:
        uploaded = st.file_uploader("Upload an aerial/orthophoto image (top-down view)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            image = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            sample_label = Path(uploaded.name).stem

            if segment_ckpt_name is None:
                st.error("No segment-task checkpoint available. Train one with scripts/train.py, "
                         "or switch to Synthetic demo mode.")
                st.stop()

            n_seg = len(segment_classes_with_background(segment_scheme))
            segment_model = _load_model(str(MODELS_DIR / segment_ckpt_name), "unet", n_seg, 256)

            if superstructure_ckpt_name != "(none - obstacle detection disabled)":
                n_obs = len(SUPERSTRUCTURE_CLASSES)
                superstructure_model = _load_model(str(MODELS_DIR / superstructure_ckpt_name), "unet", n_obs, 256)
        else:
            st.info("Upload an image to analyze, or switch to Synthetic demo mode in the sidebar.")
            st.stop()

    if image is None:
        st.stop()

    with st.spinner("Running roof analysis pipeline..."):
        result = run_pipeline(
            image, solar_config=solar_config, gsd_m_per_px=gsd,
            segment_mask=segment_mask, superstructure_mask=superstructure_mask,
            segment_model=segment_model, superstructure_model=superstructure_model,
            model_image_size=256,
        )

    st.subheader("Results")
    cols = st.columns(4)
    gsd_flag = " (assumed)" if result.gsd_is_assumed else ""
    cols[0].metric("Roof area", f"{result.usable.roof_area.area_m2:.1f} m^2{gsd_flag}")
    cols[1].metric("Usable area", f"{result.usable.usable_area.area_m2:.1f} m^2{gsd_flag}")
    cols[2].metric("Panels placed", f"{len(result.placement.panels)}")
    cols[3].metric("Capacity", f"{result.capacity.capacity_kwp:.1f} kWp")

    cols2 = st.columns(4)
    cols2[0].metric("Orientation", direction_to_compass_label(result.orientation.dominant_direction))
    cols2[1].metric("Estimated pitch", f"~{result.pitch.pitch_deg_typical:.0f} deg")
    cols2[2].metric("Uncertainty range", f"{result.uncertainty.panel_count_low}-{result.uncertainty.panel_count_high} panels")
    cols2[3].metric("Uncertainty level", result.uncertainty.uncertainty_level)

    st.caption(
        f"Pitch is an archetype-based heuristic estimate ({result.pitch.archetype}), NOT a "
        f"measurement -- a single nadir image has no direct geometric cue for out-of-plane tilt. "
        f"{result.uncertainty.explanation}"
    )

    tab1, tab2, tab3 = st.tabs(["Segmentation overview", "Usable area", "Panel placement"])
    with tab1:
        fig = plot_sample_overview(image, result.segment_mask, result.superstructure_mask, segment_scheme, title=sample_label)
        st.pyplot(fig)
        if result.seg_uncertainty is not None:
            st.caption(f"Mean segmentation confidence (softmax margin): {result.seg_uncertainty.mean_confidence:.2f} | "
                       f"Low-confidence pixel fraction: {result.seg_uncertainty.low_confidence_fraction:.2%}")
        else:
            st.caption("Segmentation confidence unavailable in ground-truth/synthetic-mask mode.")

    with tab2:
        obstacle_polys = sum(result.obstacle_polygons_by_class.values(), [])
        fig = plot_usable_area(image, None, result.usable.usable_polygon, obstacle_polys)
        st.pyplot(fig)
        st.write(f"Obstacle area removed: {result.usable.obstacle_area.area_m2:.1f} m^2")

    with tab3:
        fig = plot_panel_placement(image, result.usable.usable_polygon, result.placement)
        st.pyplot(fig)
        st.write(f"Packing efficiency: {result.placement.packing_efficiency * 100:.1f}% of usable area covered.")

    with st.expander("Methodology & limitations"):
        st.markdown(
            "- **Orientation** is read directly from the predicted/ground-truth roof-segment "
            "class (RID encodes azimuth as the segment's semantic class).\n"
            "- **Pitch** is a documented archetype-based heuristic (regional prior), never a "
            "measurement -- a single nadir photo has no stereo/elevation cue for tilt.\n"
            "- **Panel count** comes from real rectangle-packing geometry (obstacle-aware, "
            "orientation-aware), not `area / panel_area`.\n"
            "- **Capacity** is theoretical installed nameplate capacity, not an energy-yield "
            "simulation.\n"
            "- **Uncertainty** is a sensitivity-analysis-based estimate (perturbing safety "
            "margin / clearance / GSD, widened by segmentation confidence when available), "
            "not a formally calibrated statistical confidence interval."
        )


if __name__ == "__main__":
    main()
