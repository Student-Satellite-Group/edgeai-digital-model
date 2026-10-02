# Thermal-Anomaly Test Run: Does the Thermal Branch Help?

**Experiment:** fire (thermal anomaly) detection on a single Landsat 9 wildfire scene, comparing RGB-only, thermal-only and RGB+thermal versions of the two-branch model
**Project:** edgeai-digital-model (Student Satellite Group)
**Run date:** 2026-09-24
**Model under test:** `model_architecture.py`, the two-branch feature-level fusion CNN (19,090 parameters, unchanged from the first run)
**Companion report:** `Pipeline_Test_Run_Report.md` (the earlier Sentinel-2 cloud and vegetation run, whose thermal question this experiment follows up)
**Status:** the planned experiment completed. Items that were not done are listed in §8 and §12.

---

## 1. Purpose and relationship to the first run

The first test run (`Pipeline_Test_Run_Report.md`) could not answer whether the thermal branch adds anything, for three reasons:

- Its thermal data came from a different satellite, about 24 hours after the optical image.
- In that pairing, cloud and clear pixels had the same mean surface temperature (310.3 K vs. 309.7 K), so thermal carried no cloud information.
- The thermal subset (285 patches) was too small for either model to learn anything.

This experiment asks the question again on a task where the thermal signal should be **clear**, with two requirements set by the requester:

1. **A classification task with a clear thermal anomaly.**
2. **Negligible time drift between the RGB and thermal inputs.**

The task is **active-fire detection** on a single Landsat 9 scene of a large wildfire complex. RGB and thermal come from the same scene, so there is no time gap between them. The label comes from neither input.

---

## 2. Executive summary

**Data and design**
- **Scene:** Landsat 9 `LC09_L2SP_158014_20260817_02_T1` (western Siberia, 2026-08-17, 0.3% cloud).
- **Inputs:** RGB is the OLI visible bands (B4, B3, B2) and thermal is the TIRS surface-temperature band (ST_B10). Both come from one scene, acquired in the same overpass.
- **Label:** active fire, detected from OLI SWIR bands (B7, B5), which are in neither model input.
- **Patches:** 1,848 patches of 128 × 128 pixels (3.84 km × 3.84 km): **134 fire (7.3%)** and 1,714 non-fire.
- **Validation:** spatial-block 4-fold cross-validation, three variants (RGB-only, thermal-only, RGB+thermal), three seeds each.
- **Scene choice:** this scene was the only one of four screened that passed a pre-agreed gate, with a thermal-contrast AUC of 0.964. The other three scored 0.43 to 0.59.

**Headline results** (pooled out-of-fold, mean ± std over 3 seeds; a random guess gets average precision 0.073)

| Variant | ROC-AUC | Average precision | Best F1 | Recall at 90% specificity |
|---|---|---|---|---|
| RGB-only | 0.751 ± 0.028 | 0.234 ± 0.032 | 0.325 | 0.400 |
| Thermal-only | **0.918 ± 0.011** | **0.683 ± 0.037** | **0.673** | **0.813** |
| RGB + thermal | 0.889 ± 0.007 | 0.605 ± 0.029 | 0.579 | 0.736 |
| *Untrained statistic: hottest thermal pixel minus median* | *0.964* | *0.857* | *n/a* | *n/a* |

**What the results support**
- **The thermal channel helps a lot.**
  - Adding thermal to RGB raises ROC-AUC by +0.110 (95% interval +0.064 to +0.157) and average precision by +0.359 (+0.278 to +0.436).
  - Both intervals exclude zero.
- **Fusion did not beat thermal alone.**
  - Thermal-only was ahead of the fused model by +0.018 ROC-AUC (interval −0.005 to +0.046) and +0.050 average precision (+0.003 to +0.104).
  - The AUC interval includes zero and the AP interval only just excludes it, so this gap is suggestive, not established.
- **A simple untrained statistic beat every trained variant.** "Hottest thermal pixel minus median" scores ROC-AUC 0.964, which is 0.036 above the thermal CNN (interval +0.013 to +0.063).
- **The result is about large fires.** The median fire patch contains 105 SWIR fire pixels (about 9.5 ha) and 63% contain at least 64 (about 5.8 ha). The earlier Punjab scene, where fires were small, showed no thermal contrast at all.

**What the results do not support:** any claim about other scenes, dates, regions or small fires; the mission area (New Delhi); on-device performance; or *why* fusion did not help.

---

## 3. Design decisions

### 3.1 Removing time drift: one scene, both inputs

The requester's constraint was that RGB and thermal must not be separated in time. Pairing separate satellites can't guarantee that: in the first run the closest Landsat pass was 24 hours after Sentinel-2.

Landsat 8 and 9 carry both a visible/SWIR imager (OLI) and a thermal sensor (TIRS) on **one platform, imaged in the same pass**. Using a single Landsat scene for both inputs removes the calendar-day gap, the time-of-day gap and any difference in cloud or smoke state between the two inputs. This also mirrors the payload concept: two co-mounted sensors viewing the same ground at the same time.

The intra-scene OLI-to-TIRS timing offset and sub-pixel alignment were not measured in this run. Alignment relies on the Tier-1 (T1) geometric processing of the product.

### 3.2 Making the test meaningful: a label from neither input

If "anomaly" were defined from the thermal pixels, the thermal branch would win trivially and the test would show nothing about whether thermal adds information. So the label comes from OLI's **SWIR bands (B7, B5)**:

- The RGB input has only B2, B3 and B4.
- The thermal input is the TIRS band.

The thermal branch therefore only helps if TIRS heat carries real information about SWIR-detected fire. The RGB branch can only use indirect cues such as smoke or burn scars.

### 3.3 Pre-committed go/no-go gate before any training

Before training on any scene, I measured how well one untrained number, the thermal contrast (hottest cell minus median of the 24 × 32 thermal grid), separates fire patches from non-fire patches. The gate was an AUC of about 0.8 or higher. The purpose was to avoid training on a scene where the thermal anomaly isn't visible, which is exactly what happened with the first candidate (§6.1).

---

## 4. Data

### 4.1 Primary dataset

| Property | Value |
|---|---|
| Scene ID | `LC09_L2SP_158014_20260817_02_T1` |
| Platform / instruments | Landsat 9, OLI and TIRS-2 |
| Product | Collection 2, Level-2 Science Product (L2SP), Tier 1 |
| WRS-2 path / row | 158 / 014 |
| Acquisition | 2026-08-17, about 06:22 UTC |
| Catalog cloud cover | 0.3% |
| Sun elevation | about 37° |
| Raster size | 8371 × 8321 px at 30 m (about 251 km × 250 km) |
| Valid surface-reflectance pixels | 40.9 M of about 69.7 M (59%); the rest is fill outside the tilted scene footprint |
| Location | around 65.5°N, 76.2°E: the centre of the MODIS fire cluster that led to it, which lies within the scene footprint |
| Source | Microsoft Planetary Computer, collection `landsat-c2-l2`, anonymous access |

### 4.2 Bands used

| Role | Band / STAC asset | Native resolution | Conversion | Used for |
|---|---|---|---|---|
| Red | SR_B4 / `red` | 30 m | reflectance = DN × 2.75e-5 − 0.2 | RGB input |
| Green | SR_B3 / `green` | 30 m | same | RGB input |
| Blue | SR_B2 / `blue` | 30 m | same | RGB input |
| NIR | SR_B5 / `nir08` | 30 m | same | label only |
| SWIR2 | SR_B7 / `swir22` | 30 m | same | label only |
| Thermal | ST_B10 / `lwir11` | 100 m (TIRS), distributed at 30 m | K = DN × 0.00341802 + 149.0; DN 0 = fill | thermal input |

The six band files are 77–96 MB each, about 526 MB in total.

**Note on the thermal product.** ST_B10 is an atmospherically corrected *surface temperature*, not the raw brightness temperature. It can saturate (373 K cap) and can be masked where retrieval fails, for example under smoke or cloud (§8, item 3). A raw-radiance product would preserve more of the hottest pixels, but Landsat 8/9 Level-1 is not hosted on Planetary Computer (§9).

### 4.3 Scenes screened and rejected

All four scenes were screened with the same script and the same pre-training gate.

| Scene | Region / date | Cloud | SWIR fire px | Clusters | Positive patches (with thermal) | Thermal contrast, median K (fire vs. no-fire) | Fire-pixel ST vs. scene median | **AUC** | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| `LC09_L2SP_147039_20251105` | Punjab/Haryana, 2025-11-05 (crop-residue burning) | 0.05% | 772 | 389 | 167 (163) | 5.06 vs. 4.77 | 304.5 K vs. 301.8 K | 0.547 | Rejected |
| `LC09_L2SP_202032_20260720` | Central Spain, 2026-07-20 | 0.1% | 1,380 | 450 | 163 (158) | 5.89 vs. 6.62 | 322.2 K vs. 318.8 K | 0.432 | Rejected |
| `LC09_L2SP_045029_20260724` | Oregon, 2026-07-24 | 0.2% | 4,314 | 604 | 121 (119) | 9.14 vs. 8.21 | 324.8 K vs. 312.6 K | 0.590 | Rejected |
| **`LC09_L2SP_158014_20260817`** | **W. Siberia, 2026-08-17** | **0.3%** | **21,021** | **1,485** | **194 (134)** | **16.27 vs. 3.02** | **313.3 K vs. 299.1 K** | **0.964** | **Selected** |
| `LC09_L2SP_046025_20260731` | British Columbia, 2026-07-31 | 16.4% | n/a | n/a | n/a | n/a | n/a | n/a | Download produced no files; not screened |

- **Level-2 thermal mostly does not mask fires:** 0.0% of fire pixels have missing thermal in Punjab, Spain and Oregon, and 6.6% in Siberia.
- **Very hot pixels are rarer than saturation:** 1.4% of Oregon fire pixels and 0.6% of Siberian ones sit at the top of the range.
- **"Positive patches" here** counts every 128 × 128 patch with at least two SWIR fire pixels, before any thermal-coverage filter. The final dataset (§5.1) uses a stricter filter.

### 4.4 How the candidate scenes were found

1. **Locate large fires** using MODIS Terra 8-day fire composites (`modis-14A2-061`), June 20 to August 31 2026, over four regions:
   - North America west / Canada: 48 composites.
   - Siberia: 44.
   - Mediterranean / southern Europe: 50.
   - South America: 44.

   Fire pixels are mask class 8 or 9 (nominal and high confidence). Each fire cluster is an 8-connected group of 1 km pixels. MODIS was used **only to find events**, never as labels.

2. **Rank** the tile-composites by largest cluster. The top entries:

| Region | Composite start | MODIS tile | Fire px | Largest cluster (px) | Centre (lon, lat) |
|---|---|---|---|---|---|
| NA west / Canada | 2026-07-20 | h10v03 | 1,481 | 696 | −121.711, 51.158 |
| Mediterranean | 2026-07-20 | h17v04 | 1,479 | 675 | −4.543, 40.361 |
| NA west / Canada | 2026-07-28 | h10v04 | 1,931 | 654 | −119.639, 48.722 |
| NA west / Canada | 2026-07-20 | h09v04 | 4,586 | 581 | −120.625, 44.513 |
| Siberia | 2026-07-28 | h22v02 | 4,793 | 566 | 117.847, 66.014 |
| Siberia | 2026-08-13 | h21v02 | 5,516 | 457 | 76.183, 65.491 |

3. **Find Landsat 8/9 scenes** over those centres near the composite dates. Candidates included `LC09_L2SP_202032_20260720` (Spain, 0.1% cloud), `LC09_L2SP_045029_20260724` (Oregon, 0.2%), `LC09_L2SP_158014_20260817` (Siberia, 0.3%), `LC09_L2SP_046025_20260731` (British Columbia, 16.4%), `LC08_L2SP_045026_20260801` (Washington / BC border, 16.4%) and `LC09_L2SP_130014_20260728` (Yakutia, 10.4%). The last two were not screened.

### 4.5 Data considered but not used

| Dataset | Reason |
|---|---|
| Sentinel-2 tile 43RGM and its Landsat 9 thermal pass (first run) | Different satellites, 24 hours apart; no thermal-cloud correlation |
| Landsat 8/9 Level-1 (TOA radiance) | Not on Planetary Computer; that collection is 1972–2013 MSS only |
| Landsat Collection 2 on Earth Search | Requester-pays S3 bucket (needs AWS credentials and bills the requester) |
| MODIS composites as labels | 1 km resolution, 8-day accumulation; too coarse for 30 m patch labels |
| Repo's `download_tiles.py` scenes | Not used in either run |

---

## 5. Methodology

### 5.1 From scene to patches

Patches are built with `build_fire_patches.py`, reading the scene one patch-row (128 rows) at a time to keep memory low.

1. **Grid:** non-overlapping **128 × 128 px patches** (3.84 km × 3.84 km): a 65 × 65 grid, 4,225 patches in all.
2. **Validity filter:** a patch needs at least 95% valid surface-reflectance pixels, which removes patches that fall on the fill areas at the scene's tilted edges.
3. **Fire label** (§5.2): at least 2 fire pixels = positive, 0 = negative, exactly 1 = ambiguous and dropped.
4. **Thermal filter:** a patch is kept only if at least 95% of its ST pixels are valid. This is applied to **both** classes, and to every variant, so all three see exactly the same patches.
5. **Thermal grid:** valid gaps (up to 5%) are filled with the patch mean, then the 128 × 128 window is area-averaged (`cv2.INTER_AREA`) to a **24 × 32** grid (rows × columns).
6. **RGB:** `clip((DN × 2.75e-5 − 0.2) / 0.3, 0, 1)`, stored as uint8, channel order R, G, B.

**Patch accounting**

| Step | Patches |
|---|---|
| Total grid | 4,225 |
| Dropped: less than 95% valid reflectance (scene edge) | 1,817 |
| Dropped: ambiguous (exactly 1 fire pixel) | 10 |
| Dropped: fire patches with less than 95% valid thermal | 55 |
| Dropped: non-fire patches with less than 95% valid thermal | 495 |
| **Kept** | **1,848** (134 fire, 1,714 non-fire) |

Before the thermal filter there were 189 valid fire patches and 2,209 valid non-fire ones. So **29.1% of fire patches** and **22.4% of non-fire patches** were removed for thermal coverage; this is a selection effect (§8, item 3).

### 5.2 The fire label

A pixel is a fire pixel if both hold, using surface reflectance ρ:
- ρ7 / ρ5 > 1.8, and
- ρ7 − ρ5 > 0.17.

These are SWIR fire-detection thresholds in the style of published Landsat-8 active-fire work; **the constants were not independently validated in this run.** Only pixels with valid B5 and B7 count.

**Patch label:** positive if the patch has at least 2 fire pixels, negative if it has 0, and patches with exactly 1 are dropped as ambiguous (10 patches).

### 5.3 Model and variants

The model is `build_model()` from `model_architecture.py`, **19,090 parameters**, with default inputs:
- RGB (128, 128, 3).
- Thermal (24, 32, 1).
- Two branches, each ending in **GlobalAveragePooling**.
- Concatenation, then Dense 32, Dropout 0.3, Dense 2 (softmax).

The file was last modified on 2026-09-23, before this run, so the run used the current working-tree version with the corrected thermal shape.

Three variants use the same architecture with different inputs:

| Variant | RGB input | Thermal input |
|---|---|---|
| RGB-only | real | zeros |
| Thermal-only | zeros | real |
| RGB + thermal | real | real |

### 5.4 Training protocol

| Setting | Value |
|---|---|
| Initialization | From scratch |
| Optimizer / loss | Adam, learning rate 1e-3; sparse categorical cross-entropy |
| Epochs / batch | 20 / 32, no early stopping |
| Class imbalance | Balanced class weights from the training fold (about 1:12.8) |
| Augmentation | Random horizontal and vertical flips, redrawn each batch, applied to RGB and thermal together |
| Thermal normalization | (K − mean) / std, with mean and std from the training fold only |
| RGB scaling | 0–1 (uint8 / 255) |
| Model selection | None: no validation set, so test folds never influence training |
| Seeds | 0, 1, 2 (`tf.keras.utils.set_random_seed` plus a seeded NumPy generator) |

### 5.5 Validation: spatial-block cross-validation

- The 65 × 65 grid is divided into **10 × 10-patch blocks (38.4 km on a side)**. 37 blocks contain data.
- Blocks are assigned to **4 folds**, balancing fire patches greedily. Labels are used only for this assignment, never for training.
- Each fold is tested by a model trained on the other three, and all out-of-fold predictions are pooled into one score per patch.

| Fold | Patches | Fire patches |
|---|---|---|
| 0 | 549 | 33 |
| 1 | 295 | 34 |
| 2 | 290 | 34 |
| 3 | 714 | 33 |

Neighbouring patches from the same fire complex are strongly correlated, so blocking is essential. It reduces but does not eliminate leakage at block boundaries.

### 5.6 Metrics

Accuracy is meaningless at a 7.3% fire rate, so I report rank and precision-recall metrics:

- **ROC-AUC:** probability that a random fire patch scores higher than a random non-fire patch (0.5 = chance).
- **Average precision (AP):** mean precision at the rank of each fire patch (chance = the base rate, 0.0725).
- **Best F1:** the highest F1 over all thresholds (optimistic, since the threshold is chosen on the same predictions).
- **Recall at 90% specificity:** recall when the threshold lets through 10% of non-fire patches.

### 5.7 Uncertainty and baselines

- **Block bootstrap:** resample the 37 blocks with replacement (1,000 replicates, percentile intervals) and recompute each metric on the seed-averaged predictions. Differences between variants use the *same* resamples, so they are paired. Replicates with fewer than 5 fire patches were skipped.
- **Seed-averaged predictions:** the three seeds' probabilities are averaged per patch. These act as a small ensemble, so their scores are higher than any single seed's.
- **Untrained statistics:** several single numbers computed from the raw patches with no fitting, as baselines. Because nothing is fitted, they carry no leakage.

---

## 6. Results

### 6.1 Per-seed results (single models, pooled out-of-fold)

| Seed | Variant | ROC-AUC | AP | Best F1 | Recall at 90% spec. |
|---|---|---|---|---|---|
| 0 | RGB-only | 0.731 | 0.266 | 0.331 | 0.410 |
| 0 | Thermal-only | 0.933 | 0.731 | 0.717 | 0.843 |
| 0 | RGB + thermal | 0.882 | 0.576 | 0.561 | 0.739 |
| 1 | RGB-only | 0.732 | 0.190 | 0.306 | 0.358 |
| 1 | Thermal-only | 0.906 | 0.640 | 0.635 | 0.806 |
| 1 | RGB + thermal | 0.898 | 0.645 | 0.618 | 0.754 |
| 2 | RGB-only | 0.791 | 0.245 | 0.337 | 0.433 |
| 2 | Thermal-only | 0.914 | 0.678 | 0.667 | 0.791 |
| 2 | RGB + thermal | 0.886 | 0.595 | 0.557 | 0.716 |

The mean ± std over seeds is the headline table in §2 (population standard deviation, n = 3).

### 6.2 Seed-averaged predictions with 95% block-bootstrap intervals

| Variant | ROC-AUC (95% interval) | AP (95% interval) |
|---|---|---|
| RGB-only | 0.802 (0.731, 0.859) | 0.285 (0.135, 0.460) |
| Thermal-only | **0.928 (0.890, 0.954)** | **0.706 (0.516, 0.821)** |
| RGB + thermal | 0.911 (0.865, 0.939) | 0.658 (0.475, 0.775) |
| *Untrained: hottest pixel minus median* | *0.964 (0.940, 0.979)* | *0.857 (0.738, 0.920)* |

**Paired differences** (same resamples; a positive value favours the first name):

| Comparison | Δ ROC-AUC | Share of resamples above 0 | Δ AP | Share above 0 |
|---|---|---|---|---|
| Thermal-only − RGB-only | +0.128 (+0.064, +0.193) | 100% | +0.409 (+0.300, +0.499) | 100% |
| RGB+thermal − RGB-only | +0.110 (+0.064, +0.157) | 100% | +0.359 (+0.278, +0.436) | 100% |
| Thermal-only − RGB+thermal | +0.018 (−0.005, +0.046) | 94% | +0.050 (+0.003, +0.104) | 98% |
| Untrained statistic − thermal CNN | +0.036 (+0.013, +0.063) | 100% | n/a | n/a |

- **Thermal vs. RGB is decisive:** every interval excludes zero.
- **Thermal-only vs. fused is not decisive:** the AUC interval includes zero, and the AP interval only just excludes it. The ranking is consistent (thermal-only ahead in 94–98% of resamples) but weak evidence.

### 6.3 Operating point at 90% specificity (seed-averaged)

The threshold is set so that 10% of non-fire patches (172 of 1,714) are flagged as fire:

| Variant | Fires caught (of 134) | Missed | False alarms | Recall | Precision |
|---|---|---|---|---|---|
| RGB-only | 66 | 68 | 172 | 0.493 | 0.277 |
| Thermal-only | 108 | 26 | 172 | 0.806 | 0.386 |
| RGB + thermal | 107 | 27 | 172 | 0.799 | 0.384 |

At this setting, thermal-only and the fused model both catch about 80% of fires, with roughly 1.6 false alarms for every true detection. RGB-only catches under half.

### 6.4 Per-fold results (mean over the 3 seeds; ROC-AUC / AP)

| Fold | Patches / fire | RGB-only | Thermal-only | RGB + thermal |
|---|---|---|---|---|
| 0 | 549 / 33 | 0.770 / 0.284 | 0.916 / 0.752 | 0.910 / 0.629 |
| 1 | 295 / 34 | 0.811 / 0.392 | 0.940 / 0.839 | 0.914 / 0.793 |
| 2 | 290 / 34 | 0.770 / 0.406 | 0.958 / 0.869 | 0.876 / 0.711 |
| 3 | 714 / 33 | 0.776 / 0.136 | 0.912 / 0.499 | 0.892 / 0.455 |

- **Thermal beats RGB in every fold.**
- **Thermal-only is at or above the fused model in every fold**, on both metrics.
- **AP varies widely between folds** (thermal-only ranges from 0.499 to 0.869), which is why the block-bootstrap intervals are wide.

### 6.5 Paired per-seed comparison: thermal-only minus fused

| Seed | Δ ROC-AUC | Δ AP |
|---|---|---|
| 0 | +0.051 | +0.155 |
| 1 | +0.008 | −0.005 |
| 2 | +0.028 | +0.083 |

Thermal-only is ahead on ROC-AUC in all three seeds, but on AP the gap is large in two seeds and essentially zero in one.

### 6.6 Untrained baselines (same 1,848 patches, no fitting)

| Statistic | ROC-AUC | AP |
|---|---|---|
| Thermal contrast (max − median of the 24 × 32 grid) | **0.964** | 0.857 |
| Thermal max | 0.963 | **0.883** |
| Thermal 95th percentile − median | 0.873 | 0.488 |
| Thermal standard deviation | 0.855 | 0.371 |
| Thermal mean | 0.767 | 0.361 |
| RGB standard deviation (texture) | 0.747 | 0.239 |
| RGB blue mean | 0.701 | 0.123 |
| RGB red mean | 0.652 | 0.116 |
| RGB mean brightness | 0.644 | 0.112 |
| RGB green mean | 0.587 | 0.098 |

- **One-number thermal statistics beat every trained model**, on both AUC and AP.
- **The RGB CNN (seed-averaged 0.802) is only modestly above a trivial RGB texture statistic (0.747).**
- **At 90% specificity, the contrast statistic finds 123 of 134 fires; the thermal CNN finds 108.** They share 107. The CNN finds 1 that the statistic misses; the statistic finds 16 that the CNN misses.

### 6.7 Dose-response: thermal contrast against fire size

| SWIR fire pixels in patch | Patches | Contrast, median K (p25, p75) | Hottest-cell median |
|---|---|---|---|
| 0 (non-fire) | 1,714 | 3.02 (2.41, 4.08) | 302.4 K |
| 2–3 | 6 | 6.91 (6.48, 7.52) | 307.0 K |
| 4–7 | 13 | 4.38 (3.31, 6.17) | 305.2 K |
| 8–15 | 7 | 7.80 (4.35, 8.97) | 307.7 K |
| 16–63 | 24 | 11.95 (7.73, 15.01) | 311.6 K |
| 64 or more | 84 | 20.52 (16.29, 30.59) | 320.7 K |

- Thermal contrast **rises with fire size**. The bins below 16 pixels hold only 26 patches, so their medians are noisy.
- The fire patches are large: **median 105 fire pixels (about 9.5 ha), 90th percentile 286, maximum 518.** At 0.09 ha per 30 m pixel, 84 of 134 patches (63%) are at least 5.8 ha.

---

## 7. Interpretation

**Does the thermal pipeline help at all?** Yes, for this task. Adding thermal to RGB raises ROC-AUC from 0.80 to 0.91 and average precision from 0.29 to 0.66 (seed-averaged), with intervals that exclude zero. The improvement holds in every fold and every seed. The result follows from the dose-response: bigger fires produce much higher thermal contrast, and non-fire patches sit at about 3 K.

**Does fusion help beyond thermal alone?** Not shown. The fused model matched or trailed thermal-only in every fold, but the gap (+0.018 AUC, +0.050 AP) is inside the bootstrap uncertainty for AUC. I have not tested why it trails. Plausible explanations are that the RGB branch adds noise with only 134 fire patches, or that the fusion head or training setup is not tuned. Neither is verified.

**Is the CNN thermal branch efficient?** No. A one-line statistic outperforms it significantly. My untested hypothesis is that global average pooling dilutes a localized hot spot, and that global max pooling would suit this task better. That is a hypothesis about `model_architecture.py`, not a finding.

**What are the RGB cues?** The RGB-only model reaches AUC 0.80 versus 0.75 for a trivial texture statistic. Smoke and burn scars are plausible sources but were not investigated.

**How far does this extend?** The earlier Punjab scene had small crop-residue fires and no thermal contrast (AUC 0.547). Together with the dose-response above, the consistent reading is that **this thermal sensor at this resolution helps for large, intense fires and not for small ones.** That is inferred from two scenes and is not a tested boundary.

---

## 8. Limitations and threats to validity

1. **One scene, one date, one region.** The block bootstrap resamples blocks within this scene only. It captures spatial variation, not variation between scenes, seasons or biomes.
2. **Small positive class.** 134 fire patches, mostly large fires, spread over 37 blocks. Fold AP ranges from 0.499 to 0.869 for thermal-only.
3. **Selection effect from the thermal filter.** 55 of 189 valid fire patches (29%) and 495 of 2,209 valid non-fire patches (22%) were dropped because more than 5% of their surface-temperature pixels were missing. Across the scene, 11.2% of valid pixels have missing surface temperature and 6.6% of fire pixels do. The cause was not investigated; smoke, cloud masks and retrieval failure on the hottest pixels are candidates. The dropped fire patches may be the smokiest or most intense, which would bias the measured thermal advantage in an unknown direction.
4. **Label reliability was not verified.** The SWIR rule was not compared with MODIS or VIIRS detections at the same time, and no patches were inspected by eye. False positives (for example bright hot ground) and false negatives (smouldering fire under smoke) are possible. MODIS composites were used only to locate events.
5. **Label and thermal share a physical cause.** Fire produces both the SWIR and thermal signals. The sensors and bands are independent, but a strong relationship is expected, so the size of the thermal gain partly reflects that.
6. **No hyperparameter tuning, and only 3 seeds.** Fixed 20 epochs, learning rate 1e-3, no validation-based stopping. Three seeds give only a rough picture of run-to-run variation.
7. **Ensemble versus single-model metrics.** The intervals in §6.2 are on the 3-seed ensemble; the headline table in §2 is per-seed. Both are reported and they differ (thermal-only AUC 0.918 vs. 0.928).
8. **Pooled metrics.** Out-of-fold scores from four differently trained models are pooled; calibration differences between them can affect pooled ROC-AUC and AP. Per-fold results are shown in §6.4.
9. **Resolution mismatch with the payload.** Landsat gives 30 m visible and 100 m thermal; the payload design is a 7.2 m RGB camera and a 32 × 24 MLX90640. The thermal grid here has cells of about 120 m × 160 m, roughly the TIRS native scale. Optics, noise and spectral response all differ.
10. **Surface-temperature product, not brightness temperature.** Level-2 ST is an atmospherically corrected retrieval that can saturate at 373 K and can be masked.
11. **Not the mission area.** A Siberian scene tests the thermal branch, not the New Delhi payload use case.
12. **Best-F1 is optimistic** (threshold picked on the evaluated predictions); the 90%-specificity operating point uses the same predictions to set its threshold.
13. **Fold construction used labels** (to balance fire patches across folds), though only for assignment.

---

## 9. Pipeline and infrastructure findings

| # | Finding | Detail | Suggested action |
|---|---|---|---|
| 1 | Landsat 8/9 Level-1 is not on Planetary Computer | The `landsat-c2-l1` collection covers 1972–2013 (MSS only). Only Level-2 exists for Landsat 8/9. | Accept Level-2 ST, or source Level-1 elsewhere |
| 2 | Landsat Collection 2 on Earth Search is requester-pays | `s3://usgs-landsat`; needs AWS credentials and bills the requester | Use Planetary Computer's anonymous access |
| 3 | MODIS SAS tokens are scoped to storage account and container | `/sas/v1/token/modis-14A2-061` returned 403 on the files; `/sas/v1/token/modiseuwest/modis-061-cogs` worked | Use account/container-scoped tokens |
| 4 | MODIS date filter returned no 2025 results | Searches for late 2025 returned nothing; 2026 dates work. Cause unknown. | Verify coverage before relying on older dates |
| 5 | GDAL's HTTP reads were unreliable on this network | Windowed COG reads hung, or failed with corrupt tile decodes; one 4096 × 4096 band read took 20 s when it worked | Download whole bands with `requests` |
| 6 | Streaming downloads can hang | One stalled at exactly 45 MiB | Use short fixed-size range requests with resume |
| 7 | Scratch disk is nearly full | 97% used, 8.2 GB free, with about 1.36 GB of Landsat bands in scratch | Clean up (see §11) |
| 8 | Uncommitted thermal-shape fix | `THERMAL_INPUT_SHAPE = (24, 32, 1)` is in the working tree of `model_architecture.py` and `task_definition.md` but not committed. This run used it. | Commit it |

No repository files were changed by this experiment.

---

## 10. Incident log

| Step | What happened |
|---|---|
| First scene | The Punjab scene screened at AUC 0.547 and was rejected. This led to the search for a scene with a large anomaly. |
| Level-1 search | Empty: that collection is MSS only. |
| MODIS search | Date filter returned nothing for 2025; 2026 dates worked. |
| Windowed reads | The first attempt hit a 500-second timeout with no output. A larger read chunk size made one band take 20 s. Later reads failed with corrupt-tile errors and could not be recovered. |
| File-like reads | Reading through a `requests`-backed file object hung for 240 s and was abandoned. |
| Whole-band downloads | Using `requests`. A streaming download stalled at 45 MiB and was killed; it was rewritten with 8 MB range requests. Per-band times then ranged widely: 24 s each for the first scene, 211–834 s for the Siberian screening bands, and 97–153 s for its RGB bands. |
| MODIS files | Returned 403 XML errors until the account/container-scoped token was used. |
| Four-scene screening | The job reported a failure status although three scenes' results were written and the fourth (British Columbia) produced no files; cause undetermined. |
| Session | The Claude Code session restarted mid-run, marking two waiting monitors stopped; no effect on results. |
| Training | Ran for about 40 minutes without interruption; no memory kill this time. |

---

## 11. Reproducibility

### 11.1 Environment

| Package | Version |
|---|---|
| Python | 3.12.10 |
| NumPy | 2.5.2 |
| SciPy | 1.16.2 |
| TensorFlow | 2.20.0 |
| rasterio / GDAL | 1.5.0 / 3.12.1 |
| OpenCV | 5.0.0 |
| scikit-image | 0.26.0 |

### 11.2 Artifacts

Everything is in the session scratchpad, **not in the repo**:
`C:\Users\Pranjal\AppData\Local\Temp\claude\C--Users-Pranjal-Desktop-SSG\8234bb2b-5a15-4ccd-8bea-8b00b6179bc6\scratchpad\`

| File | Purpose |
|---|---|
| `find_fires.py`, `find_fires.log`, `fire_candidates.json` | MODIS fire-cluster ranking |
| `fetch_landsat.py` | Resumable band downloader (Planetary Computer) |
| `screen_scene.py`, `screen_many.py`, `screen_many.log` | Pre-training gate: fire counts and thermal-contrast AUC |
| `build_fire_patches.py` | Patch dataset and labels |
| `train_fire.py`, `train_fire.log` | Training and cross-validation |
| `fire_results.json`, `fire_oof.npz` | Per-seed metrics and out-of-fold predictions |
| `analyze_fire.py`, `analyze_fire.log` | Bootstrap intervals, per-fold results, baselines, dose-response |
| `landsat/<scene>/` | Downloaded bands, `screen.npz`, and (for the Siberian scene) `fire_patches.npz` |
| `probe_scene.py`, `rangefile.py` | Abandoned approaches (windowed GDAL reads, file-like reads); kept for reference |

### 11.3 Commands

```
python fetch_landsat.py LC09_L2SP_158014_20260817_02_T1 swir22,nir08,lwir11,red,green,blue
python screen_scene.py LC09_L2SP_158014_20260817_02_T1
python build_fire_patches.py LC09_L2SP_158014_20260817_02_T1
python train_fire.py LC09_L2SP_158014_20260817_02_T1 20 3
python analyze_fire.py
```

Seeds are fixed, but CPU TensorFlow (oneDNN) is not bit-for-bit deterministic, so expect small differences on re-runs. Planetary Computer scenes can be reprocessed, and access tokens are short-lived and are fetched fresh on each call.

### 11.4 Scratch cleanup

The scratchpad holds about 1.36 GB of Landsat bands. The three rejected scenes' bands (about 800 MB together) are safe to delete; keep `screen.npz` in each if you want the screening statistics. The Siberian scene folder (about 566 MB including derived files; the six raw bands are 526 MB) is needed to rebuild patches, though `fire_patches.npz` alone is enough to retrain.

---

## 12. Recommended next steps

1. **Test global max pooling in the thermal branch** (or average plus max concatenated). About 10 minutes to run. It directly targets the gap between the CNN (0.918) and the untrained statistic (0.964).
2. **Add independent scenes and switch to leave-one-scene-out validation.** Candidates: Yakutia `LC09_L2SP_130014_20260728` (not yet screened), retry British Columbia `LC09_L2SP_046025_20260731`, and Oregon as a negative control (its thermal AUC was only 0.59). Every added scene should pass the same pre-training gate.
3. **Verify labels** against MODIS or VIIRS detections on the scene's date, and inspect a random sample of fire and non-fire patches by eye.
4. **Investigate the missing-thermal patches:** what fraction is smoke, cloud, water or hot-pixel masking, and how much the dropped fire patches would change the result.
5. **Stratify by fire size** with more small-fire patches, to find the size at which thermal stops helping.
6. **Revisit the fusion design only if fusion is wanted:** for example modality dropout during training, or a gated fusion head. Neither is warranted until the fusion gap is shown to be real (step 2).
7. **Commit the thermal-shape fix** (§9, item 8), and **clean up scratch storage** (§11.4).
