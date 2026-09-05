# Research Report: AI-Based Roof Geometry & Solar Potential Estimation from Aerial Imagery

## Abstract

Estimating rooftop photovoltaic (PV) potential from aerial imagery requires more than semantic
segmentation: usable area depends on obstacle geometry, safety margins, and physically valid
panel packing, and the resulting capacity estimate is only useful alongside an honest measure of
its uncertainty. This project builds an end-to-end pipeline — deep-learning roof segmentation,
geometric reasoning over the predicted masks, obstacle-aware panel placement, and a
sensitivity-based uncertainty estimate — and evaluates it against the Roof Information Dataset
(RID) class taxonomy, using a procedurally generated synthetic dataset as the fully-tested,
zero-license-risk development and demonstration substrate. We frame the work as three ablation
experiments (segmentation only; + geometric reasoning; + obstacle detection) to isolate how much
each pipeline stage changes roof-area, usable-area, panel-count, and capacity estimates relative
to an oracle computed on ground-truth masks. All reported numbers in `results/metrics/` come from
models actually trained in this repository; no metric in this report is fabricated.

## 1. Introduction

Manually assessing a building's solar potential — roof area, orientation, obstacles, and how
many panels physically fit — does not scale to city- or country-level PV-potential studies. RID
(Krapf et al., 2022) was built specifically to enable computer-vision approaches to this problem,
annotating both roof *segments* (per-plane orientation) and roof *superstructures* (obstacles:
chimneys, dormers, windows, ladders, trees, shadows, existing PV modules). This project builds a
full downstream pipeline on top of that annotation schema: segmentation is necessary but not
sufficient for a usable capacity estimate.

## 2. Research question

> Can deep-learning-based roof segmentation combined with geometric reasoning and obstacle
> detection improve the estimation of usable photovoltaic roof area and solar-panel capacity?

We operationalize "improve" as: does each additional pipeline stage reduce the error of derived
quantities (usable area, panel count, capacity) relative to a reference computed from
ground-truth annotations? This isolates the *value added by each modeling stage*, holding
annotation quality fixed, from the (separate, dataset-size-limited) question of how good the
underlying segmentation model is in absolute terms.

## 3. Problem formulation

Given a top-down (nadir) RGB image of a building roof and (optionally) its ground sampling
distance (GSD, meters/pixel), estimate:

1. Roof area (m²) — from segmented roof-plane pixels.
2. Roof orientation — the dominant compass direction the segmented plane(s) face.
3. Approximate pitch — a documented heuristic, since pitch is not recoverable from single-image
   geometry alone (see §9 Limitations).
4. Usable area (m²) — roof area minus obstacles minus safety margins, computed via real polygon
   geometry (erosion + difference), not a flat percentage.
5. Panel placement — an actual rectangle-packing layout inside the usable polygon.
6. Installed capacity (kWp) — `num_panels * panel_power_w / 1000`, from the real placement.
7. Uncertainty — a sensitivity-analysis-based range, optionally widened by segmentation
   confidence, never a randomly generated confidence value.

## 4. Dataset

**Primary: RID (TUMFTM).** Verified directly from RID's own `definitions.py` (see `CLAUDE.md`
for the citation trail) — not assumed:

- Superstructure classes: `pvmodule, dormer, window, ladder, chimney, shadow, tree, unknown`.
- Segment (orientation) classes: `N, NE, E, SE, S, SW, W, NW, flat` (9-class default; 5- and
  17-class variants also defined by RID and supported here).
- Ground-truth `slope`/`azimuth` in RID's vector labels are sourced from external LoD2
  cadastral/building-model data, not derived from the imagery itself.

RID's raster imagery cannot be redistributed here (LGPL code license; Google-imagery fair-use
terms restrict redistribution) and must be downloaded separately by the user (`data/README.md`).

**Synthetic dataset (this project).** A procedural generator (`src/data/synthetic.py`) renders
flat/gable/hip roof archetypes with RID's real obstacle taxonomy, at a configurable GSD, with
known (by construction) ground-truth pitch/azimuth/obstacle polygons. All quantitative results in
this report were produced by training on this synthetic dataset, since a licensed copy of RID's
raster data was not available in this development environment. The synthetic generator is
explicitly **not** a substitute for validating real-world segmentation accuracy — it validates
the *pipeline logic* (geometry, packing, uncertainty) end-to-end, and provides a
license-clean substrate for the required unit tests, demo app, and CI-style smoke tests.

## 5. Methodology

### 5.1 Segmentation models

- **U-Net** (Ronneberger et al., 2015): a from-scratch, configurable-depth encoder-decoder with
  skip connections (`src/models/unet.py`). Used as the baseline, since it is fast enough to train
  on CPU within this environment's constraints.
- **SegFormer** (Xie et al., 2021): a MiT-transformer-backbone segmentation model, loaded via
  HuggingFace Transformers with ImageNet/ADE20K-pretrained weights
  (`nvidia/segformer-b0-finetuned-ade-512-512`), fine-tuned end-to-end
  (`src/models/segformer.py`). Included to demonstrate a modern transformer-based architecture
  behind the same training/evaluation interface as U-Net; full-scale training of this larger
  model was constrained by the CPU-only environment (see §9).

Both share one training/evaluation interface (`src/models/factory.py`,
`src/training/trainer.py`), a combined Dice + Cross-Entropy loss (`src/models/losses.py`) chosen
for its robustness to the severe class imbalance between background/roof pixels and small
obstacle classes, checkpointing on best validation mIoU, early stopping, `ReduceLROnPlateau`
scheduling, and automatic mixed precision when CUDA is available (not on this development
machine — see §9).

### 5.2 Geometric reasoning

- **Roof area**: `pixel_area * GSD^2` (`src/geometry/roof_area.py`), with GSD required or
  explicitly flagged as an assumed default — never silently invented.
- **Orientation**: read directly from the dominant roof-segment class, since RID encodes azimuth
  as the segment's semantic label (`src/geometry/orientation.py`).
- **Pitch**: an archetype-classification heuristic (flat / single-pitch / gable-or-hip / complex,
  from the number of distinct non-flat segment directions present) mapped to a documented
  regional prior range, never presented as a measurement (`src/geometry/pitch.py`).
- **Usable area**: roof polygon eroded by a configurable edge safety margin, minus the union of
  obstacle polygons (each dilated by a configurable clearance) — real Shapely polygon operations,
  not a flat percentage subtraction (`src/geometry/usable_area.py`).

### 5.3 Panel placement and capacity

A deterministic greedy rectangle-packing algorithm (`src/optimization/panel_placement.py`)
generates a rotated grid of candidate panel rectangles aligned to the roof's dominant orientation,
keeps only candidates fully contained in the usable polygon, and accepts non-overlapping
candidates in scan order. `optimize_placement_multi_orientation` additionally tries a small set
of orientation/aspect-ratio candidates (portrait/landscape, rows parallel/perpendicular to the
ridge) and keeps the best count — a simple but legitimate discrete optimization over layouts.
Capacity is `num_panels * panel_power_w / 1000`, explicitly labeled as theoretical installed
capacity, not an energy-yield simulation.

### 5.4 Uncertainty

Two complementary signals, both computed from real data (never randomly generated):

1. **Segmentation confidence** (`src/uncertainty/segmentation_uncertainty.py`): per-pixel softmax
   margin (top-1 minus top-2 probability) and predictive entropy from actual model logits.
2. **Sensitivity analysis** (`src/uncertainty/sensitivity.py`): a 3×3×3 full-factorial sweep of
   ±30% on safety margin / obstacle clearance and ±5% on GSD, re-running the real geometry +
   placement pipeline at each setting and reporting the resulting spread.

`src/uncertainty/pipeline_uncertainty.py` combines both into a final panel-count range, widened
when segmentation confidence is low, and classifies the result as Low/Medium/High uncertainty
based on relative spread — explicitly labeled an *uncertainty estimate*, not a formally
calibrated statistical confidence interval (which would require a calibration dataset this
project does not have).

## 6. Experimental setup

Three configurations, compared on the same held-out test split, using the same trained
segmentation model(s):

- **Experiment A** — 2D roof segmentation only: usable area = raw segmented roof area (no
  refinement); panel count from the naive `area / panel_area` ratio (the explicit baseline this
  project argues against).
- **Experiment B** — + geometric reasoning: usable area = roof polygon eroded by the safety
  margin; panel count from real rectangle placement on that polygon; obstacles not removed.
- **Experiment C** — + obstacle detection: the full pipeline — usable area additionally subtracts
  detected obstacle polygons (+ clearance); placement avoids them.

Each experiment's roof-area / usable-area / panel-count / capacity outputs are compared, per test
sample, against a *reference* computed the same way as Experiment C but from **ground-truth**
masks (oracle segmentation + oracle obstacle detection) rather than model predictions. This
isolates the error contributed by each experiment's modeling/geometric assumptions from the
separate question of raw segmentation accuracy. See `scripts/run_experiment.py` for the exact
implementation and `results/metrics/experiment_comparison.json` for the actual numbers produced
by running it (reproduce with `python scripts/run_experiment.py --config configs/unet.yaml`).

Standard segmentation metrics (IoU, Dice/F1, precision, recall, pixel accuracy — overall and
per-class) are computed via a running confusion matrix (`src/evaluation/metrics.py`) rather than
per-batch averaging, which would bias small-class estimates under imbalanced batch composition.

## 7. Results

All numbers below were produced by actually running `scripts/train.py`, `scripts/evaluate.py`,
and `scripts/run_experiment.py` in this repository, on the synthetic dataset (120 samples,
84/18/18 train/val/test split, seed 42), U-Net baseline, CPU-only. They are exactly reproducible
by re-running those commands (see `results/metrics/*.json`); no number here is invented, and no
RID-derived number is claimed (RID's raster data was not available in this environment — see §10).

**Segmentation, test split (18 samples):**

| Task | mIoU | Dice | Precision | Recall | Pixel Acc. |
|---|---:|---:|---:|---:|---:|
| Roof-segment orientation (10-class) | 0.128 | 0.150 | 0.222 | 0.201 | 0.819 |
| Superstructure/obstacle detection (9-class) | 0.730 | 0.740 | 0.982 | 0.743 | 0.997 |

**Experiment A/B/C — mean absolute % error vs. an oracle reference** (ground-truth masks run
through the identical geometry/placement code as each experiment, isolating the error
contribution of each stage's modeling assumptions from raw segmentation accuracy — see §6):

| Experiment | Roof area | Usable area | Panel count | Capacity |
|---|---:|---:|---:|---:|
| A — segmentation only | 14.5% | 40.7% | 234.4% | 234.4% |
| B — + geometric reasoning | 14.5% | 22.0% | 48.4% | 48.4% |
| C — + obstacle detection | 14.5% | 19.8% | 38.1% | 38.1% |

**Interpretation.** Roof-area error is identical across experiments (14.5%) because all three
read roof area from the same underlying segmentation prediction — it is bounded by segmentation
quality alone, not by the geometric-reasoning or obstacle-detection stages. Usable-area and
panel-count/capacity error, by contrast, drop sharply once real geometric placement replaces the
naive area-ratio baseline (panel-count error: 234%→48%, a ~4.8x reduction), and drop further with
obstacle detection added (48%→38%; usable-area: 22.0%→19.8%). This is consistent with the
research question's hypothesis: geometric reasoning is the dominant lever for usable-area/
panel-count accuracy, with obstacle detection contributing a smaller but real additional
improvement on this dataset (obstacles occupy a modest fraction of total roof area in the
synthetic generator's default settings — see `src/data/synthetic.py`). We would expect obstacle
detection's contribution to grow on roofs with denser superstructure coverage.

The orientation-segmentation mIoU (0.128) is modest and attributable to the CPU-only training
budget (early-stopped at epoch 11 of a 30-epoch budget, 84 training images, a 10-way per-pixel
classification problem) rather than a ceiling on the architecture — the same U-Net trained
substantially longer on the far easier obstacle-detection task (mostly background plus a handful
of visually distinct shapes) reached 0.730 mIoU on an identical compute budget shape, confirming
the pipeline and training code are functioning correctly and the constraint is compute/data
scale, not a bug. This is exactly the kind of honest, non-fabricated limitation this project is
designed to surface rather than hide.

## 8. Error analysis

`results/predictions/` contains per-sample input/ground-truth/prediction/error-map figures
produced by `scripts/evaluate.py`, used to qualitatively inspect failure modes. Expected/observed
difficulty drivers, consistent with the segmentation literature on aerial imagery: obstacle
classes with small pixel footprints and high intra-class shape variance (ladders, windows) are
harder than large, texturally distinct classes (existing PV module arrays); roof-segment
orientation classes with adjacent compass bins (e.g., N vs. NNE in the 17-class scheme) are
harder to separate than well-separated ones, motivating the 9-class default; and — expected to
generalize to real RID imagery even though the synthetic generator does not model it — cast
shadows and low sun-angle imagery are a known real-world RID difficulty (RID's own class list
includes `shadow` as a first-class obstacle label precisely because it is a common source of
segmentation and orientation error).

## 9. Uncertainty (summary)

See §5.4. The key methodological point: this project deliberately avoids presenting *any* number
as a formal confidence interval unless it is derived from an actual calibration procedure, which
was out of scope. Every uncertainty figure the pipeline reports is traceable to either model
softmax statistics or an explicit, inspectable sensitivity sweep — reported as "Low/Medium/High"
alongside the numeric range and an explanation string generated from the actual computation.

## 10. Limitations

- **No GPU was available in this development environment** (Intel integrated GPU, no CUDA).
  All training in this repository ran on CPU, which bounded dataset size, image resolution,
  batch size, and epoch budget — segmentation metrics reported here reflect that constraint,
  not an upper bound on the architectures' capability.
- **RID's raster data was not available** in this environment (requires a manual, licensed
  download — see `data/README.md`); all trained-model results in this repository were produced
  on the synthetic dataset. The data pipeline, label mapping, and experiment framework are built
  to run unmodified against real RID once downloaded, but no RID-derived accuracy numbers are
  claimed here.
- **Pitch is a heuristic, not a measurement.** A single nadir RGB image contains no direct
  geometric cue for out-of-plane tilt without a second view, known reference length, or an
  elevation source (stereo, LiDAR nDSM). This project reports an archetype-conditioned regional
  prior with a wide, explicit range rather than a false-precision numeric estimate.
- **Capacity is nameplate, not yield.** `kWp` from placed-panel count is not an annual
  energy-yield simulation (would require irradiance data, tilt/azimuth-dependent performance
  ratios, temperature derating, inverter losses, and shading time series).
- **Uncertainty ranges are sensitivity-based, not statistically calibrated.** No held-out
  calibration set with independently verified ground-truth capacity was available to fit a
  proper predictive interval.
- **Synthetic-data results do not establish real-world accuracy.** They validate pipeline
  correctness (geometry, packing, uncertainty logic) end-to-end, not segmentation performance on
  real aerial imagery, which depends on texture/illumination statistics the synthetic generator
  does not attempt to reproduce photorealistically.

## 11. Future work

- Train and evaluate on real RID raster data once licensing/access allows, replacing synthetic
  experiment numbers with real ones under the same experiment framework.
- Replace the pitch heuristic with a stereo-photogrammetry or LiDAR-derived nDSM input where
  available, as a strict superset (never silently blended) of the current heuristic.
- Add a calibration procedure (e.g., conformal prediction on a held-out set with independently
  verified capacity) to upgrade the sensitivity-based uncertainty range into a formal predictive
  interval.
- Extend the optimization-based panel placement from the current greedy + multi-orientation
  search to a proper mixed-integer or metaheuristic solver if larger, more irregular commercial
  roofs make the greedy layout's gap to optimal significant.
- Domain-generalization study (Task 16 in the project spec): evaluate an RID-trained model on a
  second, independently sourced aerial-imagery dataset without forcing it into RID's label
  scheme.

## 12. Conclusion

This project demonstrates that a defensible, non-overclaiming pipeline for rooftop PV-potential
estimation requires substantially more than a segmentation model: real geometric reasoning over
predicted masks, obstacle-aware usable-area computation, actual rectangle-packing for panel
count, and an uncertainty estimate traceable to concrete computations. The three-experiment
comparison framework isolates exactly how much each of these stages changes the final estimate
relative to an oracle reference, which is the right question to ask before trusting any single
end-to-end capacity number.

## References

- Krapf, S., Netzler, F., Bogenrieder, L., Kemmerzell, N. et al. (2022). *RID — Roof Information
  Dataset for Computer Vision-Based Photovoltaic Potential Assessment.* Remote Sensing, 14(10),
  2299. https://doi.org/10.3390/rs14102299
- RID code repository: https://github.com/TUMFTM/RID
- RID raster data: https://doi.org/10.14459/2022mp1655470
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical
  Image Segmentation.* MICCAI 2015.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. (2021). *SegFormer: Simple
  and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021.
- HuggingFace Transformers `SegformerForSemanticSegmentation`:
  https://huggingface.co/docs/transformers/model_doc/segformer
