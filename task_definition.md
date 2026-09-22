# Task 5.1 — Classification Task Definition

**Status:** FINAL (locked); Track: Team C, Phase 5.

## Task
**Cloud / no-cloud** binary classification of the fused optical+thermal
proxy imagery. This is the inference task the edge Digital Model will run.

## Inputs / Outputs

| Item | Spec |
|------|------|
| Input | Fused RGB+thermal frame (Task 4.5 output) at provisional GSD |
| Input size | 256 x 256 x 3 (configurable in `model_architecture.py`) |
| Output | Binary label: `0` = no-cloud, `1` = cloud |
| Loss | Binary Cross-Entropy |
| Metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| Optimizer | Adam (default LR 1e-3, see Task 5.3 training pipeline) |

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