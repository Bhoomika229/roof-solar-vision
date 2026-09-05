# Data

This project's primary dataset is the **Roof Information Dataset (RID)** from TUM's Institute
of Automotive Technology (TUMFTM), built for computer-vision-based photovoltaic potential
assessment.

- Code repository: https://github.com/TUMFTM/RID
- Raster data (images + masks) download: https://doi.org/10.14459/2022mp1655470
- Publication: Krapf et al., *"RID — Roof Information Dataset for Computer Vision-Based
  Photovoltaic Potential Assessment"*, Remote Sensing 14(10), 2022.
  https://doi.org/10.3390/rs14102299

## Why RID is not bundled in this repository

1. **Licensing.** RID's code is LGPL-licensed, and its aerial imagery is Google Satellite/aerial
   imagery usable only under Google's fair-use, non-commercial terms — it may not be
   redistributed.
2. **Size/access.** The raster data is hosted externally (TUM's data repository, ~GB scale) and
   requires a separate, direct download by each user.

This project therefore **never auto-downloads or redistributes RID**. What it does instead:

- Implements a complete, RID-*compatible* data pipeline (`src/data/dataset.py`) that
  **discovers** whatever dataset layout you place under `data/raw/` — it does not hard-code
  filenames.
- Ships a **synthetic data generator** (`src/data/synthetic.py`, `scripts/generate_demo_data.py`)
  so the entire non-training pipeline (geometry, obstacle handling, panel placement, capacity,
  uncertainty, visualization, the Streamlit demo, and the unit tests) is runnable with zero
  external downloads.

## How to obtain and place RID yourself

1. Download the raster data archive from https://doi.org/10.14459/2022mp1655470.
2. Extract it so you end up with (at minimum) these sub-folders somewhere under `data/raw/`:
   ```
   data/raw/RID/
     images_roof_centered_png/        (or images_roof_centered_geotiff/)
     masks_segments/
     masks_superstructures_reviewed/  (or masks_superstructures_initial/)
   ```
   These exact folder names come from RID's own `definitions.py`. `src/data/dataset.py`'s
   `discover_dataset_root()` looks for any of the name variants RID itself uses (reviewed vs.
   initial label quality, PNG vs. GeoTIFF) — it does not require you to rename anything.
3. Run:
   ```bash
   python scripts/prepare_data.py --root data/raw/RID --splits-dir data/splits
   ```
   This discovers matching image/mask ids and writes reproducible `train.txt` / `val.txt` /
   `test.txt` files under `data/splits/`.
4. Point a config's `dataset.root` at `data/raw/RID` (see `configs/unet.yaml`) and train:
   ```bash
   python scripts/train.py --config configs/unet.yaml
   ```

### Verified RID class taxonomy (not guessed)

Taken directly from RID's `definitions.py` (see `CLAUDE.md` for the full citation trail):

- **Roof superstructures** (obstacles): `pvmodule, dormer, window, ladder, chimney, shadow,
  tree, unknown` (+ a `background` class we add for semantic-segmentation masks).
- **Roof segments** (orientation): `N, NE, E, SE, S, SW, W, NW, flat` (9-class default; 5- and
  17-class variants also supported — see `src/data/label_maps.py`).

If your downloaded copy of RID encodes raw mask pixel values differently than assumed here,
adjust the mapping in `src/data/label_maps.py` — the discovery/loading code does not depend on
specific pixel values being hard-coded elsewhere.

## Synthetic data (default, no download required)

```bash
python scripts/generate_demo_data.py --num-samples 200 --image-size 256 --seed 42
python scripts/prepare_data.py --root data/processed/synthetic --splits-dir data/splits
```

This procedurally renders roofs (flat / gable / hip archetypes) with realistic obstacle
placements (chimneys, dormers, windows, ladders, existing PV modules, trees + cast shadows),
using RID's own class taxonomy. Because the generator *creates* each roof, it also knows the
true pitch/azimuth it used — useful for validating the geometry code against known ground truth,
but this is a property of the synthetic generator only, not evidence that pitch is measurable
from a real single aerial photo (see `CLAUDE.md`).

## Directory layout produced/expected here

```
data/
  raw/            # place a real, manually-downloaded RID copy here (gitignored)
  processed/      # synthetic dataset output and/or any preprocessed real data (gitignored)
  splits/         # train.txt / val.txt / test.txt (gitignored, regenerate via prepare_data.py)
```

## Optional second dataset (domain generalization)

Task 16 of the project spec (generalization to a second aerial-imagery distribution) is
supported at the code level — any dataset that can be discovered by
`src/data/dataset.py::discover_dataset_root()` (an images/ dir + at least one mask dir with
matching filenames) can be dropped in and evaluated with a model trained on RID/synthetic data,
via `scripts/evaluate.py --config <cfg> --checkpoint <ckpt>` pointed at the new root. No specific
second dataset is bundled or forced into RID's label scheme, since doing so without inspecting
its actual classes would violate the same "don't assume class names" principle applied to RID.
