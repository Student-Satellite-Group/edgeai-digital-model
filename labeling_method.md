# Dataset Labeling Methodology

**Issue:** [#14 — [Team B] Task 3.5 - Label the Dataset](https://github.com/SohamB-42/edgeai-digital-model/issues/14)
**Script:** `label_dataset.py`

---

## 1. Primary Task: Cloud / No-Cloud (`label_cloud`)

### Source
Sentinel-2 L2A **SCL (Scene Classification Layer)** band (native 20 m resolution).
The SCL is produced by the ESA Sen2Cor processor and assigns each pixel one of
12 surface-type classes.

### Locked-in Rule

| Condition | label_cloud |
|-----------|-------------|
| Fraction of pixels with SCL ∈ {8, 9, 10} **< 30 %** | **0** (no-cloud) |
| Fraction of pixels with SCL ∈ {8, 9, 10} **≥ 30 %** | **1** (cloud) |

**SCL classes used:**
- **8** Medium cloud probability
- **9** High cloud probability
- **10** Thin cirrus

Cloud fraction is computed over all valid (non-nodata) pixels within the
pass footprint of the patch.

> ⚠ **Correction vs. issue #14 original wording:** The original issue stated
> "SCL = 1 → cloud", but SCL class 1 is "saturated or defective" — not cloud.
> The correct classes are 8, 9, and 10 (optionally 3 for cloud shadow).
> This fix is documented in `Pipeline_Test_Run_Report.md §8, finding #3`.

---

## 2. Secondary Task: Vegetation / Non-Vegetation (`label_vegetation`)

### Source
Same Sentinel-2 SCL band.

### Locked-in Rule

```
veg_fraction = class_4_pixels / (class_4_pixels + class_5_pixels)

veg_fraction ≥ 50 %  →  label_vegetation = 1  (vegetated)
veg_fraction  < 50 %  →  label_vegetation = 0  (non-vegetated)
```

**SCL classes used:**
- **4** Vegetation
- **5** Not vegetated

If neither class 4 nor class 5 pixels are present in the patch, the label is
set to **-1** (undetermined).

---

## 3. Fallback: Synthetic Labels

When no SCL band is available alongside an optical tile (e.g., the raw tiles
were downloaded before SCL was added to the downloader), `label_dataset.py`
assigns **deterministic synthetic labels** derived from a SHA-256 hash of the
`sample_id`.

Synthetic labels are *not* suitable for scientific evaluation; they exist
solely to keep the pipeline end-to-end testable without blocking on data
availability. Samples with synthetic labels can be identified by running
`qc_report.py` which records the labeling method for every sample.

---

## 4. Reproducibility

- SCL-derived labels are deterministic given the same SCL file and footprint.
- Synthetic labels are deterministic: the same `sample_id` always produces the
  same label across machines and runs.
- The labeling logic is fully contained in `label_dataset.py:_compute_labels_from_scl()`
  and `label_dataset.py:_synthetic_labels()`.
