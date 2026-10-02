# Aggregated Dataset: Cloud, Vegetation and Wildfire (Thermal Anomaly) Experiments

The patch datasets used to train and test the two-branch RGB + thermal model (`model_architecture.py`) in three experiments:

| Experiment | Task | Dataset folder |
|---|---|---|
| 1 | Cloud / no-cloud | `sentinel2_delhi/` |
| 2 | Vegetated / non-vegetated | `sentinel2_delhi/` (same patches, different label) |
| 3 | Fire (thermal anomaly) / no fire | `landsat9_siberia_fire/` |

Full methodology and results are in `provenance/reports/` (`Pipeline_Test_Run_Report.md` for experiments 1 and 2, `Thermal_Anomaly_Test_Run_Report.md` for experiment 3).

**Read section 7 (caveats) before using this data.** The labels are derived, not hand-annotated, and each dataset comes from a single scene.

---

## 1. Contents

| Path | What it is | Size |
|---|---|---|
| `sentinel2_delhi/patches.npz` | 1,764 patches: RGB, thermal, labels, class fractions, folds | 65 MB |
| `sentinel2_delhi/manifest.csv` | One row per patch: georeferencing, labels, class fractions, thermal statistics | 0.4 MB |
| `sentinel2_delhi/preview.png` | Sample patches with labels | 0.6 MB |
| `landsat9_siberia_fire/patches.npz` | 1,848 patches: RGB, thermal, labels, folds, model predictions | 67 MB |
| `landsat9_siberia_fire/manifest.csv` | One row per patch (same idea as above, plus predictions) | 0.4 MB |
| `landsat9_siberia_fire/preview.png` | Sample fire and non-fire patches | 0.5 MB |
| `manifest_all.csv` | Both datasets in one table with a common schema (3,612 rows) | 0.7 MB |
| `CHECKSUMS.sha256` | SHA-256 of every file (see section 9) | |
| `provenance/` | Build and training scripts, logs, result files, model snapshot, reports | |

Total about 129 MB. The **raw satellite scenes are not included** (about 1.3 GB); see section 8 for how to re-fetch them.

---

## 2. Quick start

```python
import numpy as np, csv

d = np.load("sentinel2_delhi/patches.npz")
rgb  = d["rgb"]          # (1764, 128, 128, 3) uint8, channel order R,G,B
thm  = d["thm"]          # (1764, 24, 32) float32, Kelvin; NaN where no thermal
cloud = d["y_cloud"]     # (1764,) int8, 0 = no-cloud, 1 = cloud
veg   = d["y_veg"]       # (1764,) int8, 0/1, and -1 where the label is not valid
fold  = d["fold"]        # (1764,) int8, 0..3 (the evaluation folds used in the experiments)

f = np.load("landsat9_siberia_fire/patches.npz")
frgb, fthm, fire, ffold = f["rgb"], f["thm"], f["y_fire"], f["fold"]

# model input convention used in the experiments
x_rgb = rgb.astype("float32") / 255.0          # (N,128,128,3)
x_thm = ((thm - mean) / std)[..., None]        # (N,24,32,1); mean/std from the TRAINING fold only
```

Do **not** split these datasets randomly. Neighbouring patches come from the same scene and are strongly correlated, so a random split leaks information between train and test. Use the provided `fold` column (section 5).

---

## 3. Dataset A: `sentinel2_delhi` (cloud and vegetation tasks)

### 3.1 Sources

| Role | Product | Identifier | Acquired |
|---|---|---|---|
| RGB and labels | Sentinel-2 L2A, tile 43RGM (UTM 43N), bands B02, B03, B04 at 10 m and SCL at 20 m | `S2C_43RGM_20260920_0_L2A` (Planetary Computer item `S2C_MSIL2A_20260920T052651_R105_T43RGM_20260920T102311`, baseline N05.12) | 2026-09-20 |
| Thermal | Landsat 9 Collection 2 Level-2 surface temperature `ST_B10` | `LC09_L2SP_147040_20260921_02_T1` | 2026-09-21, **about 24 hours after** the optical image |

The tile's catalog cloud cover is 27.37%. **The thermal comes from a different satellite about 24 hours later**, so it does not describe the same clouds as the optical patch (see section 7).

### 3.2 Construction

- The Sentinel-2 tile was read at **20 m** (5,490 × 5,490 px). The Landsat thermal was warped onto the same grid (bilinear).
- The grid is cut into non-overlapping **128 × 128 px patches (2.56 km × 2.56 km)**: 42 × 42 = 1,764 patches (5,376 of 5,490 px per axis are used). Patch centres span lon 77.05 to 78.14, lat 27.93 to 28.90.
- **RGB:** `uint8 = round(clip(DN / 6000, 0, 1) × 255)` on the raw L2A digital numbers, in R, G, B order. **No reflectance offset was applied**, because most clear-sky blue and green pixels fall below the L2A offset. Approximate recovery: `DN ≈ value / 255 × 6000`, saturating at 6000.
- **Thermal:** the patch's 128 × 128 thermal window is area-averaged to a **24 × 32** grid (rows × columns), in Kelvin. If less than 95% of the window is valid, the whole patch thermal is `NaN`. Only **285 of 1,764 patches** have thermal, all in a strip along the tile's western edge (`has_thermal = True`).

### 3.3 Labels

Both come from the Sentinel-2 scene classification layer (SCL), computed at native 20 m. Per-patch class fractions are provided (`scl_frac_0` to `scl_frac_11`).

| Task | Rule | Counts |
|---|---|---|
| Cloud / no-cloud (`y_cloud`) | Fraction of SCL classes 8, 9, 10 (cloud medium, cloud high, thin cirrus) **≥ 0.30 → 1 (cloud)**, else 0 | 735 cloud (41.7%), 1,029 no-cloud |
| Vegetation (`y_veg`) | A patch is valid only if SCL classes 4 (vegetation) + 5 (not vegetated) cover **≥ 50%**. Label = 1 if class 4 / (class 4 + class 5) **≥ 0.5** | 933 valid: 462 vegetated, 471 not; 831 patches have no vegetation label (`y_veg = -1`) |

Thermal subset (`has_thermal = True`, 285 patches): cloud 104 positive; vegetation 166 valid, 56 positive.

SCL legend: 0 no data, 1 saturated/defective, 2 dark area, 3 cloud shadow, 4 vegetation, 5 not vegetated, 6 water, 7 unclassified, 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus, 11 snow/ice.

### 3.4 `patches.npz` arrays

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `rgb` | (1764, 128, 128, 3) | uint8 | RGB patch |
| `thm` | (1764, 24, 32) | float32 | Thermal grid in K; `NaN` when `has_thermal` is False |
| `y_cloud` | (1764,) | int8 | Cloud label |
| `y_veg` | (1764,) | int8 | Vegetation label; **-1 = not valid** |
| `veg_valid` | (1764,) | bool | True where the vegetation label is defined |
| `has_thermal` | (1764,) | bool | True for the 285 patches with thermal |
| `cloud_frac`, `veg_frac` | (1764,) | float32 | Cloud fraction; vegetation share (NaN where invalid) |
| `scl_frac` | (1764, 12) | float32 | Fraction of each SCL class 0..11 |
| `fold` | (1764,) | int8 | Evaluation fold (section 5) |
| `rc` | (1764, 2) | int16 | Patch row, column in the 42 × 42 grid |
| `patch_id` | (1764,) | str | e.g. `s2delhi_r00_c00`; array index equals row order in the CSV |

`manifest.csv` carries the same information as columns, plus `crs` (EPSG:32643), projected bounds `x_min, y_min, x_max, y_max`, `center_lon`, `center_lat`, thermal statistics (`thermal_min_K`, `thermal_median_K`, `thermal_max_K`, `thermal_mean_K`, `thermal_std_K`, `thermal_contrast_K` = max minus median), and the two scene IDs.

---

## 4. Dataset B: `landsat9_siberia_fire` (fire / thermal-anomaly task)

### 4.1 Source

One scene supplies **both** inputs, so RGB and thermal are from the same overpass with no time gap:

| Property | Value |
|---|---|
| Scene | Landsat 9 `LC09_L2SP_158014_20260817_02_T1` (Collection 2, Level-2, Tier 1) |
| Path / row, date | 158 / 014, 2026-08-17, about 06:22 UTC |
| Cloud cover, size | 0.3%; 8371 × 8321 px at 30 m (about 251 km × 250 km) |
| Location | patch centres span lon 73.31 to 78.50, lat 64.51 to 66.66 (western Siberia), CRS EPSG:32643 |
| RGB bands | OLI surface reflectance B4, B3, B2 |
| Thermal band | TIRS surface temperature `ST_B10` (100 m native, distributed at 30 m) |
| Label bands | OLI SWIR SR_B7 and NIR SR_B5, **used only to make labels, not model inputs** |

### 4.2 Construction

- Non-overlapping **128 × 128 px patches at 30 m (3.84 km × 3.84 km)**: a 65 × 65 grid of 4,225 patches.
- **RGB:** `uint8 = round(clip((DN × 2.75e-5 − 0.2) / 0.3, 0, 1) × 255)`. Reflectance ≈ `value / 255 × 0.3`.
- **Thermal:** 24 × 32 grid, area-averaged from the 128 × 128 window, Kelvin (`K = DN × 0.00341802 + 149.0`). Up to 5% missing pixels are filled with the patch mean first.
- **Fire label:** a pixel is a fire pixel if ρ7 / ρ5 > 1.8 **and** ρ7 − ρ5 > 0.17 (SWIR-based rule in the style of published Landsat-8 active-fire detection; the constants were not independently validated). A patch with **at least 2** fire pixels is **positive (`y_fire = 1`)** and one with 0 is negative. Patches with exactly 1 were dropped as ambiguous.

### 4.3 What was kept

| Step | Patches |
|---|---|
| Total grid | 4,225 |
| Dropped: under 95% valid reflectance (scene edge) | 1,817 |
| Dropped: ambiguous (exactly 1 fire pixel) | 10 |
| Dropped: fire patches with under 95% valid thermal | 55 |
| Dropped: non-fire patches with under 95% valid thermal | 495 |
| **Kept** | **1,848** (**134 fire = 7.3%**, 1,714 non-fire) |

So **29% of valid fire patches and 22% of valid non-fire patches were removed for missing thermal**. This is a selection effect (section 7).

### 4.4 `patches.npz` arrays

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `rgb` | (1848, 128, 128, 3) | uint8 | RGB patch |
| `thm` | (1848, 24, 32) | float32 | Thermal grid in K (no NaN) |
| `y_fire` | (1848,) | int8 | Fire label |
| `n_fire_px` | (1848,) | int16 | Number of SWIR fire pixels in the patch (median 105, maximum 518 among fire patches) |
| `fold` | (1848,) | int8 | Evaluation fold (section 5) |
| `rc` | (1848, 2) | int16 | Patch row, column in the 65 × 65 grid |
| `patch_id` | (1848,) | str | e.g. `l9siberia_r01_c19` |
| `oof_rgb_mean`, `oof_thermal_mean`, `oof_both_mean` | (1848,) | float32 | Predicted fire probability from models that never saw the patch's fold, averaged over 3 seeds |
| `oof_{rgb,thermal,both}_seed{0,1,2}` | (1848,) | float32 | The same predictions per seed |

In `oof_*`, `rgb` = RGB-only model (thermal input zeroed), `thermal` = thermal-only model (RGB input zeroed), `both` = fused model. **These are model outputs, not features; do not train on them.**

---

## 5. How training and testing were done (the `fold` columns)

There was no fixed train/test split. Both datasets were evaluated with **4-fold spatial cross-validation**: each fold was tested by a model trained on the other three, and all out-of-fold predictions were pooled. Every patch is therefore a test patch exactly once.

| Dataset | Fold rule | Fold sizes | Fire patches per fold |
|---|---|---|---|
| `sentinel2_delhi` | Row band: `fold = floor(row × 4 / 42)` (rows 0-10, 11-20, 21-31, 32-41) | 462 / 420 / 462 / 420 | n/a |
| `landsat9_siberia_fire` | 10 × 10-patch blocks (38.4 km) assigned to folds to balance fire patches | 549 / 295 / 290 / 714 | 33 / 34 / 34 / 33 |

Subsets used in the experiments:
- **Cloud task:** all 1,764 patches, RGB-only (thermal input zeroed). This was experiment "E2".
- **Vegetation task:** the 933 patches with `veg_valid = True`, RGB-only.
- **Thermal ablation:** only the 285 patches with `has_thermal = True` ("E1"). It was inconclusive.
- **Fire task:** all 1,848 patches, three model variants, 3 seeds.

**Headline results** (details and uncertainty in the reports):

| Task | Result |
|---|---|
| Cloud / no-cloud (RGB-only) | 92.6% accuracy vs. 58.3% majority baseline (n = 1,764) |
| Vegetation (RGB-only) | 74.9% vs. 50.5% baseline; recall on vegetated only 0.52 (n = 933) |
| Fire (mean of 3 seeds, ROC-AUC) | RGB-only 0.751, thermal-only 0.918, RGB+thermal 0.889 |

Model predictions are provided **only** for the fire dataset. The Sentinel-2 run's per-patch predictions and results file were never written because the job was stopped by the system's low-memory protection; its partial log is in `provenance/logs_and_results/experiments_log.txt`.

---

## 6. Georeferencing

Every patch in `manifest.csv` and `manifest_all.csv` has `crs` (EPSG:32643 for both datasets), the patch's projected bounds (`x_min`, `y_min`, `x_max`, `y_max`, in metres) and its centre in WGS84 (`center_lon`, `center_lat`). The bounds are exact for the patch's pixel window in the source grid.

---

## 7. Caveats

1. **Labels are derived, not hand-annotated.**
   - Cloud and vegetation labels come from Sen2Cor's scene classification, computed from the same optical data the model sees. Accuracy against them means agreement with that algorithm.
   - Fire labels come from a rule-based SWIR detector that was not compared with MODIS/VIIRS or checked by eye at scale. A spot check of the smallest fire patch in `landsat9_siberia_fire/preview.png` (a river with bright sandbars, 2 fire pixels) suggests some 2 to 3 pixel positives may be non-fire hot or bright surfaces. Undetected smouldering fires may also sit among the negatives. Treat small-count positives with caution.
2. **The Delhi thermal is not simultaneous.** It is from a different satellite about 24 hours after the optical image, and covers only 285 patches. In this pairing cloud and clear pixels have the same mean surface temperature (310.3 K vs. 309.7 K), so it carries no cloud information.
3. **Single scene per dataset.** The folds are spatial, but everything comes from one scene and one date, so generalization to other scenes, seasons or regions is untested.
4. **Selection effect in the Siberian data.** Patches with missing thermal were dropped (section 4.3), and they may be the smokiest or most intense fires. The fires kept are large: 84 of 134 have at least 64 fire pixels (about 5.8 ha).
5. **Class imbalance.** Fire is 7.3% of patches. Use ROC-AUC, average precision or per-class metrics, not accuracy.
6. **Different radiometric scaling between the datasets.** Delhi RGB is scaled raw digital numbers; Siberia RGB is scaled surface reflectance. Do not pool the two without accounting for this. Resolution also differs (20 m vs. 30 m).
7. **Proxy data.** These are Sentinel-2 and Landsat images standing in for the payload's camera and thermal sensor. The intended payload has a 7.2 m RGB camera and a 32 × 24 thermal array.
8. **Level-2 surface temperature** is an atmospherically corrected retrieval; it can saturate at 373 K and can be masked over cloud or smoke.

---

## 8. Attribution, licensing and re-fetching the raw data

This package contains **derived, resampled and relabelled data**, not the original products.

- **Sentinel-2:** "Contains modified Copernicus Sentinel data (2026)". Copernicus Sentinel data is free and open under the Copernicus terms.
- **Landsat 9:** courtesy of the U.S. Geological Survey. Landsat data is in the public domain, and USGS asks for credit.
- Data was accessed through Microsoft Planetary Computer (Sentinel-2 B02 and B03 via the Earth Search catalog). Please check the current terms of each provider before redistributing.

Raw scenes (not included):

| Scene | Where | Size |
|---|---|---|
| `S2C_43RGM_20260920_0_L2A` (B02, B03, B04, SCL) | Earth Search / Planetary Computer, Sentinel-2 L2A | about 776 MB |
| `LC09_L2SP_147040_20260921_02_T1` (`ST_B10`) | Planetary Computer, `landsat-c2-l2` | read remotely, not stored locally |
| `LC09_L2SP_158014_20260817_02_T1` (bands `red`, `green`, `blue`, `nir08`, `swir22`, `lwir11`) | Planetary Computer, `landsat-c2-l2` | about 526 MB |

Catalogs can reprocess scenes; access tokens are short-lived.

---

## 9. Provenance and integrity

`provenance/code/` holds the exact scripts that built and evaluated these datasets:

| Script | Purpose |
|---|---|
| `fetch_scene.py`, `build_patches.py` | Delhi scene assembly and patch labels |
| `experiments.py` | Delhi cloud and vegetation training and cross-validation |
| `find_fires.py`, `screen_scene.py`, `screen_many.py`, `fetch_landsat.py` | Finding and screening candidate fire scenes |
| `build_fire_patches.py`, `train_fire.py`, `analyze_fire.py` | Siberian patches, training, and the uncertainty analysis |
| `make_export.py` | Built this package (and cross-checked every label) |
| `model_architecture.py`, `task_definition.md` | Snapshot of the model definition and task spec used |

The scripts refer to absolute paths on the original machine and will need those edited. `provenance/logs_and_results/` holds the logs, `fire_results.json` and `fire_candidates.json`.

**Integrity:** `CHECKSUMS.sha256` lists a SHA-256 for every file. Verify with `sha256sum -c CHECKSUMS.sha256` (Git Bash or Linux) or `Get-FileHash` in PowerShell.

**Model snapshot note:** the architecture used is the working-tree version with the corrected thermal input shape `(24, 32, 1)`, which differs from the last git commit (`fddb211`, `(32, 24, 1)`). That fix was still uncommitted when the experiments ran.

**Built and checked:** every label in the Delhi dataset was recomputed from the raw scene classification and matched the values used in training; counts match the reports (1,764 / 735 / 933 / 462 / 285 for Delhi; 1,848 / 134 for Siberia).
