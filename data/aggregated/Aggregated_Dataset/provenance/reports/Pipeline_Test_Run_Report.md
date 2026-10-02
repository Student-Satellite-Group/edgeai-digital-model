# Two-Branch Pipeline Test Run on a Real Sentinel-2 / Landsat 9 Scene

**Project:** edgeai-digital-model (Student Satellite Group)
**Run dates:** 2026-09-23 to 2026-09-24
**Model under test:** the two-branch, feature-level fusion CNN in `model_architecture.py` (Task 5.2, revised)
**Status of this run: PARTIAL.** The whole-tile experiments finished. The thermal-ablation experiment, the untrained-model reference and the single-image scoring did not (see §5.4 and §9).

---

## 1. Purpose and scope

**Request.** Run the newly designed pipeline on an image from the dataset and see how well it performs at cloud/no-cloud classification and at one further classification task of my choosing.

**What that turned out to require.** The repository could not support that request as stated:

| Needed for the run | State in the repo at the start |
|---|---|
| A trained model | None. Task 5.3 (issue #24) is open, and there is no `train_model.py` and no weights file. |
| An image from "the dataset" | None on disk. `data/raw/` is gitignored and was empty. |
| Labeled data | None. Phase 3 (issues #12–#15) is open, with no resampling pipeline, no `data/labeled/` and no labels. |

Running an untrained model on one image would return random numbers, and a single prediction cannot measure accuracy in any case. So the run was scoped as a **proof-of-life experiment**:

1. Acquire a real optical + thermal scene pair (thermal acquired about 24 hours after the optical pass; see §4.1).
2. Build a small labeled patch dataset from it.
3. Train the two-branch model from scratch.
4. Evaluate it with spatially blocked cross-validation against majority-class baselines.

This is **not** a substitute for Task 5.3 or for the Phase 3 dataset deliverables. It shows whether the architecture can learn from real data at all, and it exposed problems in the data pipeline (§8).

---

## 2. Executive summary

- **What ran end to end:** real Sentinel-2 L2A and Landsat 9 data → 20 m common grid → 1,764 labeled 128×128 patches (285 with thermal) → two-input model → trained from scratch → evaluated.
- **Cloud / no-cloud (RGB branch, whole tile, n = 1,764):** accuracy **92.6%** against a 58.3% majority baseline; F1 = 0.91. The labels come from Sentinel-2's Sen2Cor scene classification, which is itself computed from the same optical bands, so this measures agreement with that mask on one scene and date. It is not accuracy against independent ground truth.
- **Vegetated vs. non-vegetated (second task, RGB branch, whole tile, n = 933):** accuracy **74.9%** against a 50.5% baseline; F1 = 0.67. Precision is high (0.96) but recall is only 0.52, so it misses about half of the vegetated patches.
- **Thermal branch: no evidence either way.**
  - The thermal subset is only 285 patches, and neither the RGB-only nor the RGB+thermal cloud model learned anything on it. Both matched the majority baseline exactly (63.5%, F1 = 0).
  - The RGB+thermal vegetation runs never started.
  - The Landsat scene (acquired about 24 hours after the Sentinel-2 pass) has essentially no cloud signal in it: mean surface temperature is 310.3 K in cloud and 309.7 K in clear pixels.
- **Not obtained:** the untrained-model reference, the single-image scoring, and the RGB+thermal vegetation ablation. The background job was killed by the system's low-memory protection before it wrote its results file. It has not been restarted.
- **Repo pipeline problems found:** §8 lists seven. The most important is that the manifest pairs 2026 Sentinel-2 scenes with Landsat thermal from 2013–14.

---

## 3. Datasets

### 3.1 Datasets used

| Role | Product | Identifier | Acquisition | File and size | Access path |
|---|---|---|---|---|---|
| Blue band | Sentinel-2 L2A B02 (10 m) | tile 43RGM, `S2C_43RGM_20260920_0_L2A` (Earth Search ID) | 2026-09-20 | `B02.tif`, 239,672,237 B | Earth Search COG (AWS S3), downloaded manually by the user |
| Green band | Sentinel-2 L2A B03 (10 m) | same | 2026-09-20 | `B03.tif`, 240,992,763 B | same |
| Red band | Sentinel-2 L2A B04 (10 m) | Planetary Computer item `S2C_MSIL2A_20260920T052651_R105_T43RGM_20260920T102311` (baseline N05.12) | 2026-09-20 | `T43RGM_20260920T052651_B04_10m.tif`, 290,188,830 B | Planetary Computer (Azure), downloaded manually via signed links |
| Scene classification (labels) | Sentinel-2 L2A SCL (20 m) | same PC item | 2026-09-20 | `T43RGM_20260920T052651_SCL_20m.tif`, 4,609,455 B | Planetary Computer, downloaded manually |
| Thermal | Landsat 9 Collection 2 Level 2 surface temperature `ST_B10` (`lwir11`) | `LC09_L2SP_147040_20260921_02_T1` (path 147, row 040) | 2026-09-21, about 05:25 UTC | not stored; read remotely, windowed and warped | Planetary Computer (Azure), anonymous SAS token |

**Local location of the Sentinel-2 files:** `edgeai-digital-model\data\raw\sentinel2\S2C_43RGM_20260920_0_L2A\` (gitignored).

**Scene properties**
- Both catalogs report the same scene-level cloud cover for the Sentinel-2 tile, **27.37%**, which supports that they are the same ESA product.
- The Landsat 9 scene reports 5.28% cloud.
- The Landsat pass is **about 24 hours after** the Sentinel-2 pass (Sentinel-2 tile sensed roughly 05:27–05:41 UTC on 2026-09-20; Landsat 9 about 05:25 UTC on 2026-09-21). Both satellites are sun-synchronous, so local solar time is similar, but the calendar day differs: cloud and surface conditions are not the same.
- Thermal scale/offset: DN × 0.00341802 + 149.0 gives Kelvin, with 0 as nodata.

**Provenance caveat.** B02/B03 come from Earth Search and B04/SCL from Planetary Computer. The script asserts identical CRS (EPSG:32643) and identical tile bounds before combining them. Radiometric agreement between the two catalogs was not independently verified, although both derive from the same ESA product.

### 3.2 Datasets considered but not used

| Dataset | Why not used |
|---|---|
| Repo's `data/manifest_raw.csv` scenes | Sentinel-2 B04 (red only) from 2026-09-20, paired with Landsat-8 Collection 1 Tier-2 B10 from **2013-08-17, 2013-10-04, 2014-07-19 and 2014-09-05**. A 12-year gap means the thermal cannot describe the optical scene's cloud state, and there is no SCL and no green or blue band. None of these files existed on disk. |
| Landsat 9 `LC09_L2SP_146040_20260914_02_T2` | Full coverage of the area, but acquired 6 days before the Sentinel-2 pass and 39% cloudy. |
| Landsat Collection 2 via Earth Search (`s3://usgs-landsat`) | Requester-pays S3 bucket, which needs AWS credentials and bills the requester. |
| Sentinel-2 B08 (NIR) | Planned for an NDVI-based label, then dropped. Vegetation labels come from SCL classes instead (§4.3). |
| Public RGB+thermal datasets (e.g. FLIR ADAS) | Not needed for this run. |

### 3.3 Derived datasets

| Artifact | Contents |
|---|---|
| Scene cache (20 m common grid) | 5490 × 5490 px, EPSG:32643. RGB as raw uint16 DN, SCL as uint8, surface temperature as float32 Kelvin (NaN where no coverage). |
| Patch dataset | 1,764 non-overlapping 128×128 patches with labels (§4.2–4.3). |

---

## 4. Methodology

### 4.1 Scene assembly

1. **Grid.** The Sentinel-2 10 m tile (10980 × 10980) was read at **20 m** (5490 × 5490), using average resampling for the reflectance bands and nearest-neighbour for SCL, which is natively 20 m. An earlier build at 40 m was used only for the diagnostics in §5.1.
2. **Why 20 m.** The near-coincident thermal covers only about 17% of the tile. At 40 m that left roughly 74 patches; at 20 m it gives 285.
3. **Thermal.** The Landsat ST band was warped onto the same grid with a `WarpedVRT` (bilinear resampling, nodata = 0) and converted to Kelvin.
4. **New Delhi location.** Longitude 77.2 E, latitude 28.6 N maps to pixel (row 1719, col 758) on the 20 m grid, which falls in patch (r=13, c=5).
5. **Spatial and temporal matching between the two sensors.**
   - *Location:* matched by georeferencing only. Both products are map-projected, and the Landsat band is warped into the Sentinel-2 tile's CRS and pixel grid, so each patch's thermal window covers the same ground as its RGB patch. No image-based registration was run (`registration.py` was not used) and the Landsat-to-Sentinel-2 alignment error was not measured.
   - *Time:* **not matched.** The gap is about 24 hours (§3.1). Land cover is unchanged over that gap, but cloud fields and surface temperature are not. This is why the thermal channel carries no cloud information here (§5.1), and it weakens any thermal result, including the vegetation task.

### 4.2 Patch dataset

| Item | Specification |
|---|---|
| Patch size | 128 × 128 px at 20 m, 2.56 km × 2.56 km, non-overlapping. The grid is 42 × 42 = 1,764 patches, using 5,376 of the 5,490 px per axis. |
| RGB encoding | Channel order R, G, B. Each value is `clip(DN / 6000, 0, 1)`, stored as uint8, like an 8-bit camera. **No reflectance offset was applied**, because most clear-sky blue and green pixels fall below the L2A offset (§5.1). |
| Thermal encoding | The 128×128 thermal window is area-averaged (`cv2.INTER_AREA`) to a **24 × 32** grid (H × W), matching the MLX90640's 24 rows × 32 columns. If at least 95% of the window is valid, missing pixels are filled with the patch mean before averaging; otherwise the patch has no thermal. |
| Thermal ground footprint | Each thermal pixel covers about 107 m × 80 m, roughly the source's native TIRS resolution of about 100 m, so no fine detail is invented. |
| Thermal availability | 285 of 1,764 patches. |

### 4.3 Labels

Both label rules use Sentinel-2's Sen2Cor scene classification (SCL) at native 20 m.

| Task | Rule | Rationale |
|---|---|---|
| **Cloud / no-cloud** | Fraction of pixels with SCL class 8, 9 or 10 (cloud medium probability, cloud high probability, thin cirrus) is **≥ 0.30 → cloud**. | Same 30% threshold as `task_definition.md`. The class IDs differ from issue #14's text (§8). |
| **Vegetated / non-vegetated** (my choice) | A patch is valid only if SCL classes 4 (vegetation) plus 5 (not vegetated) cover **≥ 50%** of it. Label = 1 if class 4 / (class 4 + class 5) is **≥ 0.5**. | Land surface temperature is physically related to vegetation, so it is a fair test of whether the thermal branch adds information. It needs no NIR band and is non-trivial from RGB alone. Cloudy patches automatically drop out. |

**Class balance**

| Set | Cloud task (n, positives) | Vegetation task (n, positives) |
|---|---|---|
| Whole tile | 1,764, 735 (41.7%) | 933, 462 (49.5%) |
| Thermal subset | 285, 104 (36.5%) | 166, 56 (33.7%) |

### 4.4 Model under test

`model_architecture.py`, two inputs, **19,090 parameters** and **3,949,014 MACs per inference** (TensorFlow FLOPs profiler, measured earlier; unchanged by this run).

| Component | Layers |
|---|---|
| RGB branch (128 × 128 × 3) | Conv 16 (3×3, stride 2) → three blocks of [DepthwiseConv 3×3 stride 2 → 1×1 Conv to 32 / 64 / 128 filters, each with BatchNorm and ReLU] → GlobalAveragePooling → 128-dim vector |
| Thermal branch (24 × 32 × 1) | Conv 8 (3×3) → DepthwiseConv 3×3 stride 2 → 1×1 Conv 16, each with BatchNorm and ReLU → GlobalAveragePooling → 16-dim vector |
| Fusion head | Concatenate (144) → Dense 32 ReLU → Dropout 0.3 → Dense 2 softmax |

**Correction made before the run:** `THERMAL_INPUT_SHAPE` was `(32, 24, 1)`, which is 32 rows × 24 columns. The MLX90640 is 32 columns × 24 rows, so it was changed to `(24, 32, 1)` in `model_architecture.py` and `task_definition.md`. The parameter and MAC counts are unchanged.

### 4.5 Training protocol

| Setting | Value |
|---|---|
| Initialization | From scratch (random weights, no pretraining) |
| Optimizer / loss | Adam, learning rate 1e-3; sparse categorical cross-entropy on the 2-way softmax |
| Batch size / epochs | 32 / 25 (whole tile), 30 (thermal subset) |
| Class imbalance | Balanced class weights computed from the training fold |
| Augmentation | Random horizontal and vertical flips, re-drawn every epoch, applied identically to the RGB and thermal arrays |
| Thermal normalization | (K − mean) / std, with mean and std from the training fold only |
| "RGB-only" variant | Thermal input set to zeros, so the thermal branch sees a constant |
| Model selection | **None.** Fixed epochs, no validation set, no early stopping, so test folds were never used to pick a model. |

### 4.6 Evaluation protocol

- **Spatially blocked 4-fold cross-validation** by patch-row band, fold = ⌊row × 4 / 42⌋. Adjacent patches from the same area never straddle train and test, except at the band boundaries.
- Each fold's model trains on the other three bands. **Pooled out-of-fold predictions** give one confusion matrix per experiment.
- **Metrics:** accuracy, majority-class baseline (accuracy of always predicting the commoner class), balanced accuracy, precision, recall and F1 for the positive class, and confusion counts.

### 4.7 Experiments planned

| ID | Description | Outcome |
|---|---|---|
| E0 | Untrained (random-init) reference on all patches | Computed but **not captured** (§5.4) |
| E2 | Whole tile, RGB-only, both tasks, 1 seed, 25 epochs | **Completed** |
| E1 | Thermal subset, RGB-only vs. RGB+thermal, both tasks, 3 seeds, 30 epochs | **Partially completed** |
| Single image | Score the New Delhi patch and the nearest both-label patch with out-of-fold models | **Not done** |

---

## 5. Results

### 5.1 Data diagnostics

Diagnostics were run on the earlier 40 m build of the scene.

**Raw digital numbers by scene-classification group (percentiles 2 / 25 / 50 / 75 / 98):**

| Band | Cloud pixels | Clear pixels |
|---|---|---|
| Blue (B02) | 673 / 1914 / **4204** / 10026 / 21525 | 45 / 423 / **663** / 989 / 1869 |
| Green (B03) | 877 / 1900 / **3757** / 8519 / 17978 | 301 / 741 / **963** / 1245 / 1953 |
| Red (B04) | 1530 / 2591 / **4209** / 8620 / 17043 | 1175 / 1520 / **1761** / 2177 / 3197 |

- Clouds are about 4× brighter than clear surfaces in every band, and the bands are not mixed up.
- Median clear-sky blue and green DN are **below 1000**, so applying the L2A −1000 offset would give negative reflectance for most clear pixels. The cause is probably atmospheric-correction behaviour in hazy air (unverified). That is why plain DN scaling was used.

**Thermal coverage and content**

| Quantity | Value |
|---|---|
| Thermal-valid fraction of the tile | **16.7%**, a strip along the western edge (path 147 only partly overlaps the tile) |
| New Delhi pixel inside coverage | Yes |
| Surface temperature range / mean | 268.8–326.8 K / 309.9 K |
| Mean surface temperature in cloud vs. clear pixels | **310.3 K vs. 309.7 K** (no cloud signal) |
| Cloud fraction inside the thermal strip vs. outside | 25.6% vs. 28.5% |

**Tile composition by SCL:** cloud 27.4%, vegetation 21.8%, not vegetated 31.9%, no-data 0%.

### 5.2 E2: whole tile, RGB-only, spatial 4-fold CV, pooled out-of-fold

| Task | n | Accuracy | Majority baseline | Balanced acc. | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cloud / no-cloud | 1,764 | **0.926** | 0.583 | 0.927 | 0.891 | 0.936 | 0.913 |
| Vegetated / non-vegetated | 933 | **0.749** | 0.505 | 0.747 | 0.956 | 0.517 | 0.671 |

**Confusion counts**

| Task | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Cloud | 688 | 84 | 47 | 945 |
| Vegetation | 239 | 11 | 223 | 460 |

**Reading the results**
- **Cloud.** The RGB branch learns this well from scratch on about 1,300 training patches per fold, beating the baseline by 34 percentage points. Clouds are bright and white, so this is an easy pattern, and the labels are a rule-based function of the same optical bands. The result mostly shows the network can reproduce Sen2Cor's cloud mask.
- **Vegetation.** The gain over baseline is real (+24 points) but modest. The model rarely calls a patch vegetated wrongly (11 false alarms) but misses about half of the truly vegetated patches (223 misses). Red, green and blue alone are a weak proxy for a label derived from NDVI-type indices.
- **Sampling error.** Ignoring spatial correlation and training variance, the standard error is about ±0.6 points for cloud and ±1.4 points for vegetation, which understates the true uncertainty. This was a single seed.
- **Runtime.** About 395 s for the cloud task and 234 s for the vegetation task.

### 5.3 E1: thermal subset (285 patches), 3 seeds

| Task | Variant | Seeds finished | Accuracy | Majority baseline | F1 |
|---|---|---|---:|---:|---:|
| Cloud | RGB-only | 3 of 3 | 0.635 in every seed | 0.635 | 0.000 |
| Cloud | RGB + thermal | 3 of 3 | 0.635 in every seed | 0.635 | 0.000 |
| Vegetation | RGB-only | 1 of 3 (seed 0) | 0.530 | 0.663 | 0.536 |
| Vegetation | RGB + thermal | **0 of 3** | not run | 0.663 | not run |

- **Cloud, both variants:** accuracy equals the majority baseline exactly (181 of 285 patches are no-cloud) with F1 = 0, which means they never predicted "cloud". With about 210 training patches per fold, nothing was learned.
- **Cloud, thermal:** this is also the outcome to expect physically. In this pairing, cloud and clear pixels have the same mean surface temperature.
- **Vegetation, RGB-only (seed 0):** below the majority baseline, so it did not learn either.
- **Bottom line:** **whether the thermal branch adds information is unanswered.**

### 5.4 What was not obtained

| Item | Reason |
|---|---|
| Untrained (random-init) reference numbers | Computed inside the script but held in memory and written only at the end, so lost when the job was killed. No untrained figure is quoted in this report. |
| RGB+thermal vegetation runs, RGB-only seeds 1–2 | Job killed before reaching them. |
| Out-of-fold predictions, `results.json`, `oof.npz` | Written only at the end of the script, so never created. |
| Single-image scoring | Depends on the out-of-fold predictions. |

**Images prepared for the single-image step:** the New Delhi patch, indexed as patch (r=13, c=5), index 551. It has valid thermal, cloud fraction 0.45 so its cloud label is 1, and no vegetation label because the patch is mostly cloud.

---

## 6. What can and cannot be concluded

**Supported by this run**
- The revised two-branch architecture is trainable and runs end to end on real satellite data.
- The RGB branch alone reproduces the Sentinel-2 cloud mask at 92.6% accuracy on this scene (baseline 58.3%).
- It reaches 74.9% on a vegetation label (baseline 50.5%), with weak recall.

**Not supported by this run**
- Any claim about the thermal branch helping or not helping.
- Any claim about performance on other scenes, dates or seasons.
- Any claim about performance at the payload's actual imaging scale (design GSD 7.2 m against 20 m here).
- Any on-device timing. No latency was measured, and only the compute cost (3.95M MACs) is known.
- Any statement about the untrained baseline, beyond the fact that random weights carry no information.

---

## 7. Limitations and threats to validity

1. **One scene, one date.** Spatial CV reduces leakage between neighbouring patches, but it cannot show generalization to other scenes.
2. **Labels are not independent ground truth.** SCL is computed from the same optical data the model sees, so accuracy means agreement with Sen2Cor. Its known error modes (e.g. bright surfaces flagged as cloud) are inherited.
3. **Weak thermal pairing.** Thermal covers 17% of the tile, comes from a different day, and shows no cloud signal. The thermal experiment is under-powered and mismatched to the cloud task.
4. **Geometry mismatch.** The RGB crop is square (128 × 128) while the thermal grid is 4:3 (24 × 32), so the thermal grid samples the square footprint anisotropically (107 m × 80 m per pixel). This was a design simplification.
5. **Proxy sensors.** Sentinel-2 and Landsat stand in for the payload's camera and MLX90640. Resolution, spectral response and noise differ.
6. **Single seed for E2**, and no confidence intervals beyond the rough standard errors above.
7. **The loss deviates slightly from the spec.** `task_definition.md` names binary cross-entropy, while the 2-way softmax output was trained with sparse categorical cross-entropy. For two classes these are equivalent formulations, but it is a documented deviation.

---

## 8. Issues found in the repo pipeline

| # | Finding | Evidence | Suggested action | Related issue |
|---|---|---|---|---|
| 1 | Manifest pairs 2026 Sentinel-2 with 2013–14 Landsat thermal (12 years apart) | `data/manifest_raw.csv` acquisition dates | Pair each optical scene with a Landsat scene within about a day, e.g. via Planetary Computer's anonymous access | #11, #12, #13 |
| 2 | Downloader fetches only Sentinel-2 **B04** (red) | `manifest_raw.csv` lists only `B04.tif`; run needed B02, B03 and SCL | Also download B02, B03 and SCL (SCL is only about 5 MB) | #11, #14 |
| 3 | Issue #14 says "SCL=1 → cloud", but class 1 is "saturated/defective" | Sentinel-2 SCL class table | Use classes 8, 9 and 10 (optionally 3 for cloud shadow) | #14 |
| 4 | `THERMAL_INPUT_SHAPE` was transposed (32×24 instead of 24×32) | MLX90640 is 32 columns × 24 rows | **Fixed in the working tree** (uncommitted) | #23 |
| 5 | Landsat Collection 2 on Earth Search is on a requester-pays bucket | `storage:requester_pays` in the asset | Use Planetary Computer's anonymous SAS route | #11 |
| 6 | Only one near-coincident (about 24 h) Landsat scene overlaps the tile, covering 17% | Coverage strip; path 147 vs. tile 43RGM | Choose a Sentinel-2 tile that path 147 overlaps fully (e.g. a tile west of 43RGM), or accept a larger time offset | #11 |
| 7 | `registration.py`'s self-test fails in a fresh environment (translation drift 1.357, 0.756 vs. an assertion of < 1.0), and the repo has no `requirements.txt` | Failed after installing `opencv-python-headless` and `scikit-image`; not caused by this run's edits (docstring-only diff) | Pin dependencies; re-tune the tolerance under Task 4.2 | #16, #17 |

---

## 9. Run log and incidents

| Step | What happened |
|---|---|
| Discovery | No image, weights or labels existed (§1). |
| First fetch | GDAL failed the TLS handshake to the Sentinel-2 bucket after about 10 minutes of retries. A direct Python probe timed out to the same host (`sentinel-cogs.s3.us-west-2.amazonaws.com`), so the block is host-specific, not GDAL's fault. |
| Workaround | The user downloaded B02 and B03 manually, then B04 and SCL through signed Planetary Computer links. |
| Data diagnostics | 40 m build; found the radiometry and thermal-coverage issues in §5.1. |
| Rebuild | 20 m build, patch dataset, and a speed benchmark of about 2.5 s per epoch on 1,344 patches. |
| Experiments | E2 finished at about 630 s. E1 cloud finished at about 1,406 s. E1 vegetation RGB-only seed 0 finished at about 1,509 s. |
| Termination | The background job was **stopped by Claude Code because the system was critically low on memory**. It was not a code error, and it was not restarted. The script keeps the whole patch array as float32 (about 347 MB) and makes per-epoch augmented copies; the exact peak memory was not measured. |
| Environment | `opencv-python-headless` and `scikit-image` were missing and were installed to run the existing repo scripts. |

---

## 10. Reproducibility and next steps

### 10.1 Environment

| Package | Version |
|---|---|
| Python | 3.12.10 |
| NumPy | 2.5.2 |
| TensorFlow | 2.20.0 |
| rasterio / GDAL | 1.5.0 / 3.12.1 |
| OpenCV | 5.0.0 |
| scikit-image | 0.26.0 |

### 10.2 Artifacts

The scripts and caches live in the session scratchpad, **not** in the repo:
`C:\Users\Pranjal\AppData\Local\Temp\claude\C--Users-Pranjal-Desktop-SSG\8234bb2b-5a15-4ccd-8bea-8b00b6179bc6\scratchpad\`

| File | Purpose |
|---|---|
| `fetch_scene.py` | Builds the 20 m scene cache from the local Sentinel-2 files plus the Landsat thermal read |
| `build_patches.py` | Cuts patches and computes labels |
| `experiments.py` | Training and cross-validation (`python experiments.py <E1 epochs> <E2 epochs> <E1 seeds>`, run as `30 25 3`) |
| `single_image.py` | Scores and plots individual patches; needs `oof.npz` |
| `experiments_log.txt` | Log of the partial run |
| `scene_cache.npz`, `patches.npz` | 167 MB and 65 MB caches |

These are temporary. Copy them into the repo if they should be kept.

### 10.3 Suggested next steps

1. **Re-run with a smaller memory footprint** (store patches as uint8 and cast per batch). Save out-of-fold predictions and the untrained baseline as they are computed, not at the end. This would finish the single-image scoring.
2. **Skip or redo the thermal ablation.** 285 patches from a narrow strip cannot settle it. A proper test needs thermal that overlaps a Sentinel-2 tile fully and is within about a day.
3. **Fix the data pipeline items in §8**, particularly the manifest's date pairing and the missing bands and SCL, before Phase 3 builds a dataset on them.
4. **Add an independent check on the labels** (e.g. a second cloud mask or manual review of a sample) before quoting cloud accuracy as anything beyond agreement with Sen2Cor.
5. **Commit the thermal-shape fix**, which is currently only in the working tree of `model_architecture.py` and `task_definition.md`.
