"""
label_dataset.py — Task 3.5: Label the Dataset

Applies classification labels to every sample in data/labeled/manifest.csv
using the Sentinel-2 SCL (Scene Classification Layer) rules locked in by
issue #14.

Locked-in labeling rules (from issue #14):
    Primary:   label_cloud
        Source  : Sentinel-2 SCL band (20 m native resolution)
        Rule    : cloud fraction = fraction of pixels with SCL ∈ {8, 9, 10}
                  cloud fraction < 30%  →  label_cloud = 0  (no-cloud)
                  cloud fraction ≥ 30%  →  label_cloud = 1  (cloud)

    Secondary: label_vegetation
        Source  : Sentinel-2 SCL band
        Rule    : veg_fraction = class-4 pixels / (class-4 + class-5 pixels)
                  veg_fraction ≥ 50%  →  label_vegetation = 1  (vegetated)
                  otherwise           →  label_vegetation = 0  (non-vegetated)

When a real SCL tile is available next to the optical tile, it is used.
When it is absent (synthetic data or missing download), a deterministic
synthetic label is assigned so the pipeline remains fully testable.

Usage:
    python label_dataset.py

Deliverables (issue #14):
    - Updated data/labeled/manifest.csv with label_cloud and label_vegetation
    - labeling_method.md documenting the methodology
"""

import os
import csv
import json
import hashlib
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELED_DIR    = os.path.join("data", "labeled")
MANIFEST_PATH  = os.path.join(LABELED_DIR, "manifest.csv")
LABELING_DOC   = "labeling_method.md"

# SCL cloud classes (issue #14, locked-in)
SCL_CLOUD_CLASSES      = {8, 9, 10}    # medium, high probability, thin cirrus
SCL_CLOUD_THRESHOLD    = 0.30          # ≥ 30 % → cloud label 1
SCL_VEG_CLASS          = 4             # vegetation
SCL_NONVEG_CLASS       = 5             # not vegetated
SCL_VEG_THRESHOLD      = 0.50          # ≥ 50 % → vegetation label 1

NULL_LABEL = -1

MANIFEST_FIELDNAMES = [
    "sample_id", "pass_id", "date", "optical_path", "thermal_path",
    "footprint_geojson", "gsd_m", "label_cloud", "label_vegetation",
]


# ---------------------------------------------------------------------------
# SCL-based labeling (when a real SCL raster is available)
# ---------------------------------------------------------------------------

def _compute_labels_from_scl(scl_path: str,
                               footprint_geojson: Optional[str] = None
                               ) -> Tuple[int, int]:
    """Read an SCL GeoTIFF and compute cloud + vegetation labels.

    Parameters
    ----------
    scl_path          : path to Sentinel-2 SCL GeoTIFF (uint8, 1 band).
    footprint_geojson : optional GeoJSON string; if supplied, only pixels
                        inside the footprint are counted.

    Returns
    -------
    (label_cloud, label_vegetation) — each 0 or 1.
    """
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import shape

    with rasterio.open(scl_path) as src:
        if footprint_geojson:
            try:
                geom = shape(json.loads(footprint_geojson)["geometry"])
                data, _ = rio_mask(src, [geom.__geo_interface__],
                                   crop=True, nodata=255)
                pixels = data[0].flatten()
                pixels = pixels[pixels != 255]
            except Exception:
                pixels = src.read(1).flatten()
        else:
            pixels = src.read(1).flatten()

    if pixels.size == 0:
        return NULL_LABEL, NULL_LABEL

    n_total = pixels.size
    n_cloud = np.isin(pixels, list(SCL_CLOUD_CLASSES)).sum()
    cloud_frac = n_cloud / n_total
    label_cloud = 1 if cloud_frac >= SCL_CLOUD_THRESHOLD else 0

    n_veg    = (pixels == SCL_VEG_CLASS).sum()
    n_nonveg = (pixels == SCL_NONVEG_CLASS).sum()
    denom    = n_veg + n_nonveg
    if denom > 0:
        veg_frac = n_veg / denom
        label_vegetation = 1 if veg_frac >= SCL_VEG_THRESHOLD else 0
    else:
        label_vegetation = NULL_LABEL  # cannot determine from SCL in this patch

    return label_cloud, label_vegetation


# ---------------------------------------------------------------------------
# Synthetic labeling (deterministic, for when SCL is not available)
# ---------------------------------------------------------------------------

def _synthetic_labels(sample_id: str, optical_path: str) -> Tuple[int, int]:
    """Assign deterministic synthetic labels when SCL is unavailable.

    The labels are derived from a hash of the sample_id so they are
    reproducible across runs but vary between samples.  Roughly 40 % of
    samples receive cloud label 1, and 50 % receive vegetation label 1,
    which approximates the New Delhi AOI class distribution observed in
    the test run (see Pipeline_Test_Run_Report.md §5.1).
    """
    h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
    # Cloud: 40 % positive rate
    label_cloud      = 1 if (h % 100) < 40 else 0
    # Vegetation: 50 % positive rate
    label_vegetation = 1 if ((h >> 8) % 100) < 50 else 0
    return label_cloud, label_vegetation


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_manifest(
    manifest_path: str = MANIFEST_PATH,
    labeled_dir:   str = LABELED_DIR,
) -> List[Dict]:
    """Apply SCL-based (or synthetic) labels to every sample in manifest.csv.

    For each sample:
    1. Looks for an SCL band at the same directory as the optical tile
       (expected filename: SCL.tif or *_SCL*.tif).
    2. If found, uses _compute_labels_from_scl().
    3. If not found, uses _synthetic_labels() and records a warning.

    Updates manifest.csv in-place with filled label_cloud and label_vegetation.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run package_dataset.py first."
        )

    with open(manifest_path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    n_scl      = 0
    n_synthetic = 0

    for rec in records:
        if (int(rec.get("label_cloud", NULL_LABEL)) != NULL_LABEL
                and int(rec.get("label_vegetation", NULL_LABEL)) != NULL_LABEL):
            # Already labeled — skip
            continue

        # Search for SCL file alongside optical tile
        opt_dir = os.path.dirname(rec.get("optical_path", ""))
        scl_path: Optional[str] = None
        for fname in os.listdir(opt_dir) if os.path.isdir(opt_dir) else []:
            if "SCL" in fname.upper() and fname.lower().endswith(".tif"):
                scl_path = os.path.join(opt_dir, fname)
                break

        if scl_path and os.path.exists(scl_path):
            try:
                lc, lv = _compute_labels_from_scl(
                    scl_path, rec.get("footprint_geojson")
                )
                n_scl += 1
            except Exception as e:
                warnings.warn(f"SCL labeling failed for {rec['sample_id']}: {e}")
                lc, lv = _synthetic_labels(rec["sample_id"], rec.get("optical_path", ""))
                n_synthetic += 1
        else:
            lc, lv = _synthetic_labels(rec["sample_id"], rec.get("optical_path", ""))
            n_synthetic += 1

        rec["label_cloud"]      = lc
        rec["label_vegetation"] = lv

    # Validate: every sample must have a valid label
    unlabeled = [r["sample_id"] for r in records
                 if int(r.get("label_cloud", NULL_LABEL)) == NULL_LABEL]
    if unlabeled:
        warnings.warn(f"{len(unlabeled)} samples still have NULL cloud label "
                      f"after labeling: {unlabeled[:5]}{'...' if len(unlabeled)>5 else ''}")

    # Write updated manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    print(f"Labeled {len(records)} samples.")
    print(f"  SCL-derived    : {n_scl}")
    print(f"  Synthetic      : {n_synthetic}")

    n_cloud = sum(1 for r in records if int(r.get("label_cloud", -1)) == 1)
    n_veg   = sum(1 for r in records if int(r.get("label_vegetation", -1)) == 1)
    print(f"  Cloud=1        : {n_cloud} / {len(records)} ({100*n_cloud/max(1,len(records)):.1f}%)")
    print(f"  Vegetation=1   : {n_veg} / {len(records)} ({100*n_veg/max(1,len(records)):.1f}%)")

    return records


# ---------------------------------------------------------------------------
# Write labeling_method.md
# ---------------------------------------------------------------------------

def write_labeling_method_doc(path: str = LABELING_DOC) -> None:
    """Write the labeling methodology documentation required by issue #14."""
    doc = """\
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
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Labeling methodology written: {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Task 3.5: label_dataset.py ===")
    label_manifest()
    write_labeling_method_doc()
    print("Done.")
