# Task 5.1 — Classification Task Definition

**Status:** FINAL (locked); Track: Team C, Phase 5.

## Task
**Cloud / no-cloud** binary classification, from RGB and thermal processed
as separate branches and fused at the feature level (not pixel level -- see
Input spec and Architecture Update below). This is the inference task the
edge Digital Model will run.

## Architecture update (post Task 5.2 revision)

The original Task 5.1 input spec (a single pixel-fused RGB+thermal frame,
256x256x3) has been **superseded**. Two problems drove the change, both
measured directly against `model_architecture.py`, not assumed:

1. The single-branch, 256x256 model cost ~654,000,000 MACs/inference (TF
   FLOPs profiler) -- unworkable on ESP32-S3-class hardware despite a
   compact parameter count (187,650).
2. Pixel-level fusion required warping the native 32x24 thermal grid (768
   real pixels) onto RGB's full frame via homography, meaning nearly every
   "thermal" pixel the model saw was interpolated, not measured -- exactly
   the "silently up-sampled" failure mode the architecture document (Section
   2.2) says this project is designed to avoid.

The revised model (`model_architecture.py`) processes each modality at its
own native/appropriate resolution via two branches, and fuses **learned
feature vectors**, not raw pixels. Measured cost: 19,090 params, ~3.95M
MACs/inference (~166x cheaper than the original design).

## Inputs / Outputs

| Item | Spec |
|------|------|
| Input (RGB branch) | RGB crop corresponding to the current thermal footprint (registration provides the crop, not a full-frame warp) |
| Input size (RGB) | 128 x 128 x 3 |
| Input (thermal branch) | Native-resolution thermal grid -- **no upsampling** |
| Input size (thermal) | 24 x 32 x 1 (H x W = 24 rows x 32 columns, MLX90640 orientation) |
| Fusion | Feature-level: `Concatenate([f_rgb, f_thermal])` -> `Dense(32, ReLU)` -> classification head (see `model_architecture.py`) |
| Output | Binary label: `0` = no-cloud, `1` = cloud |
| Loss | Binary Cross-Entropy |
| Metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| Optimizer | Adam (default LR 1e-3, see Task 5.3 training pipeline) |

**Not in scope (deliberately):** a second output head for thermal-anomaly
detection was considered and explicitly not added. It would need its own
label definition (no anomaly ground-truth rule exists, unlike the cloud
label derived from Sentinel-2's SCL band below) and its own justification
for combining losses with the cloud/no-cloud task. Revisit as a separate,
deliberate scope decision if needed -- not a silent addition here.

**Consequence for `fusion.py` (Task 4.5):** its weighted-overlay pixel
fusion is no longer this classifier's input. It remains valid as a
standalone data product (Architecture doc Section 7.3, "fused output
frame") for logging/visualization, decoupled from inference.

**Consequence for `registration.py` (Task 4.1):** its role narrows from
"warp thermal onto RGB's full pixel grid" to "identify which RGB crop
corresponds to the current thermal footprint," so both branches observe the
same physical patch of ground. Per-frame registration is still used (not a
fixed calibrated offset), since a fixed offset would reintroduce the
degradation-under-vibration failure mode the architecture document (Section
10.2) already argues against.

## Provisional mission context (locked 2026-09)

- Altitude: 500 km (PROVISIONAL until final mission review)
- Camera baseline: nüSpace CMOS (Nüvü Caméras), 4.6 µm pixels, 320 mm focal
  length, GSD 7.2 m @ 500 km (Task 2.1, `sensor_specs.json`)
- Data source: Sentinel-2 L2A optical + Landsat-8 B10 thermal proxy tiles
  (Task 3.2 pilot, `data/manifest_raw.csv`)

## Labeling rule

Per Phase 3.5 methodology, per footprint:

- Cloud cover **< 30%** → `no-cloud` (0)
- Cloud cover **≥ 30%** → `cloud` (1)

Edge cases: scenes with no valid data in the footprint are excluded and
logged, not force-labeled.

## Evaluation protocol

- Stratified 70/15/15 train/val/test split, stratified on the cloud label
  (same cloud/no-cloud proportion in every split).
- Reported on the held-out test split: accuracy, precision, recall, F1,
  confusion matrix.

## Deliverable
This file (`task_definition.md`), referenced by:
- `model_architecture.py` (Task 5.2) for the model skeleton,
- Task 5.3 for the training pipeline,
- README / final report for the task specification.