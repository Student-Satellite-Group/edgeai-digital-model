"""
resample_pipeline.py — Task 3.3: Crop, Reproject, and Resample raw satellite tiles
to the mission's locked-in model input dimensions.

HARD JOIN: depends on Team A Task 1.5 (pass footprints) and Task 2.2 (GSD = 7.19 m).

Output shapes (locked-in by model_architecture.py):
    Optical  : 128 x 128 x 3  (RGB: Sentinel-2 B04, B03, B02)
    Thermal  : 24 x 32 x 1    (native MLX90640 / Lepton resolution — no upsampling)

Target CRS: EPSG:32643 (WGS 84 / UTM zone 43N — covers New Delhi).
Target GSD: 7.19 m/pixel (PROVISIONAL — from sensor_specs.json via sensor_model.py).

Usage (stand-alone):
    python resample_pipeline.py

Produces data/resampled/<sample_id>/optical.tif and thermal.tif for every
(optical, thermal) pair registered in data/manifest_raw.csv that has at least
one valid pass footprint in data/overpass_schedule.txt.
"""

import os
import json
import math
import csv
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box

# ---------------------------------------------------------------------------
# Constants (locked-in by model_architecture.py + sensor_model.py)
# ---------------------------------------------------------------------------

TARGET_CRS = CRS.from_epsg(32643)          # UTM 43N — covers New Delhi

OPTICAL_COLS = 128                          # RGB_INPUT_SHAPE width
OPTICAL_ROWS = 128                          # RGB_INPUT_SHAPE height
OPTICAL_BANDS = 3                           # B04 (R), B03 (G), B02 (B)

THERMAL_COLS = 32                           # THERMAL_INPUT_SHAPE width  (MLX90640)
THERMAL_ROWS = 24                           # THERMAL_INPUT_SHAPE height
THERMAL_BANDS = 1

PROVISIONAL_GSD_M = 7.19                   # PROVISIONAL — from sensor_specs.json

# Target AOI (New Delhi) as lon/lat WGS-84
TARGET_LAT = 28.6
TARGET_LON = 77.2
TARGET_RADIUS_KM = 50.0                    # 50 km radius (from Task 1.5)

# Footprint half-size in meters derived from target radius (square approximation)
FOOTPRINT_HALF_M = TARGET_RADIUS_KM * 1_000.0

MANIFEST_RAW_PATH = os.path.join("data", "manifest_raw.csv")
RESAMPLED_DIR = os.path.join("data", "resampled")

# Band file suffixes expected inside each Sentinel-2 tile directory
S2_BAND_SUFFIXES = ["B04.tif", "B03.tif", "B02.tif"]  # R, G, B


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _latlon_to_utm43n(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS-84 lat/lon to approximate EPSG:32643 (UTM 43N) easting/northing.

    Uses the standard 6-degree UTM strip formula (zone 43, central meridian 75E).
    Accurate to better than 1 m within ±3° of the zone boundary — well within the
    50 km radius used by this project.
    """
    import pyproj
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    easting, northing = proj.transform(lon, lat)
    return easting, northing


def _pass_footprint_polygon(center_lat: float = TARGET_LAT,
                             center_lon: float = TARGET_LON,
                             half_m: float = FOOTPRINT_HALF_M) -> dict:
    """Return a Shapely geometry (as GeoJSON-like dict) for the rectangular
    pass footprint in EPSG:32643.

    The footprint is a square of side 2*half_m centred on the target.
    """
    cx, cy = _latlon_to_utm43n(center_lat, center_lon)
    poly = box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    return mapping(poly)


# ---------------------------------------------------------------------------
# Core: resample one file
# ---------------------------------------------------------------------------

def _reproject_and_resample(
    src_path: str,
    out_path: str,
    target_crs: CRS,
    target_cols: int,
    target_rows: int,
    bands: Optional[List[int]] = None,
    resampling_method: Resampling = Resampling.bilinear,
    footprint_geom: Optional[dict] = None,
) -> bool:
    """Reproject src_path to target_crs, optionally crop to footprint_geom,
    then resample to (target_rows x target_cols) and write to out_path.

    Parameters
    ----------
    bands : list of 1-indexed band numbers to read (None = all bands).
    footprint_geom : GeoJSON-like geometry in the *source* CRS for masking.
                     If None, the full raster is used.

    Returns True on success, False if the footprint does not intersect the source.
    """
    with rasterio.open(src_path) as src:
        src_bands = bands if bands else list(range(1, src.count + 1))

        # --- Crop to footprint (in source CRS) if requested -----------------
        if footprint_geom is not None:
            # Reproject footprint from UTM 43N to source CRS for masking
            import pyproj
            from shapely.geometry import shape, mapping
            from shapely.ops import transform as shp_transform

            footprint_shape = shape(footprint_geom)
            src_crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
            try:
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:32643", src_crs_str, always_xy=True
                )
                reprojected = shp_transform(transformer.transform, footprint_shape)
            except Exception:
                # If reprojection fails, use the full raster
                reprojected = None

            if reprojected is not None and not reprojected.is_empty:
                try:
                    data, transform = rio_mask(src, [mapping(reprojected)],
                                               crop=True, nodata=0)
                    data = data[np.array(src_bands) - 1]  # select bands
                    meta = src.meta.copy()
                    meta.update({
                        "height": data.shape[1],
                        "width": data.shape[2],
                        "transform": transform,
                        "count": len(src_bands),
                    })
                except Exception:
                    # Footprint outside tile bounds — skip
                    return False
            else:
                # Read all selected bands without cropping
                data = src.read(src_bands)
                meta = src.meta.copy()
                meta["count"] = len(src_bands)
        else:
            data = src.read(src_bands)
            meta = src.meta.copy()
            meta["count"] = len(src_bands)

        if data.size == 0 or np.all(data == 0):
            return False

        # --- Reproject to TARGET_CRS ----------------------------------------
        target_transform, width, height = calculate_default_transform(
            meta.get("crs", src.crs), target_crs,
            meta["width"], meta["height"],
            *rasterio.transform.array_bounds(meta["height"], meta["width"],
                                              meta["transform"]),
        )
        reprojected_data = np.zeros(
            (len(src_bands), height, width), dtype=data.dtype
        )

        import rasterio.warp
        for i in range(len(src_bands)):
            rasterio.warp.reproject(
                source=data[i],
                destination=reprojected_data[i],
                src_transform=meta["transform"],
                src_crs=meta.get("crs", src.crs),
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method,
            )

        # --- Resample to target dimensions -----------------------------------
        from rasterio.enums import Resampling as RS
        from PIL import Image

        final = np.zeros((len(src_bands), target_rows, target_cols),
                         dtype=np.uint16)
        for i in range(len(src_bands)):
            band = reprojected_data[i].astype(np.float32)
            # Normalize to 0-65535 for uint16 storage
            bmin, bmax = band.min(), band.max()
            if bmax > bmin:
                band = (band - bmin) / (bmax - bmin) * 65535.0
            pil_img = Image.fromarray(band.astype(np.float32))
            pil_img = pil_img.resize((target_cols, target_rows),
                                     resample=Image.BILINEAR)
            final[i] = np.array(pil_img).astype(np.uint16)

        # --- Write output ----------------------------------------------------
        out_meta = {
            "driver": "GTiff",
            "dtype": "uint16",
            "count": len(src_bands),
            "height": target_rows,
            "width": target_cols,
            "crs": target_crs,
            "transform": from_bounds(
                *(rasterio.transform.array_bounds(target_rows, target_cols,
                                                  target_transform)),
                target_cols, target_rows,
            ),
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(final)

    return True


# ---------------------------------------------------------------------------
# Public API: resample_to_footprint()
# ---------------------------------------------------------------------------

def resample_to_footprint(
    optical_paths: List[str],
    thermal_path: str,
    sample_id: str,
    out_dir: str = RESAMPLED_DIR,
    center_lat: float = TARGET_LAT,
    center_lon: float = TARGET_LON,
    half_m: float = FOOTPRINT_HALF_M,
) -> Dict[str, str]:
    """Crop, reproject and resample one optical+thermal tile pair to the
    locked-in model input dimensions and write them under out_dir/sample_id/.

    Parameters
    ----------
    optical_paths : list of up to 3 single-band GeoTIFF paths (R, G, B order).
    thermal_path  : path to a single-band thermal GeoTIFF (Landsat B10 or equivalent).
    sample_id     : unique identifier for this sample (used as sub-directory name).
    out_dir       : root output directory.
    center_lat, center_lon : centre of the pass footprint (WGS-84 degrees).
    half_m        : half-side of the square footprint in metres.

    Returns
    -------
    dict with keys 'optical_path' and 'thermal_path' pointing to the output
    files, or an empty dict if the footprint did not intersect either source.
    """
    sample_dir = os.path.join(out_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    footprint_geom = _pass_footprint_polygon(center_lat, center_lon, half_m)

    # --- Optical (multi-band merge) -----------------------------------------
    optical_out = os.path.join(sample_dir, "optical.tif")
    if optical_paths:
        # Stack individual single-band files
        bands_data = []
        ref_meta = None
        for bp in optical_paths[:OPTICAL_BANDS]:
            if not os.path.exists(bp):
                warnings.warn(f"Optical band missing: {bp}")
                continue
            tmp_out = optical_out + f"_band{len(bands_data)}.tmp.tif"
            ok = _reproject_and_resample(
                bp, tmp_out,
                TARGET_CRS, OPTICAL_COLS, OPTICAL_ROWS,
                bands=[1],
                footprint_geom=footprint_geom,
            )
            if ok:
                with rasterio.open(tmp_out) as s:
                    bands_data.append(s.read(1))
                    ref_meta = s.meta.copy()
                os.remove(tmp_out)

        if len(bands_data) == OPTICAL_BANDS:
            ref_meta.update({"count": OPTICAL_BANDS, "dtype": "uint16"})
            with rasterio.open(optical_out, "w", **ref_meta) as dst:
                for i, b in enumerate(bands_data, start=1):
                    dst.write(b, i)
            optical_result = optical_out
        else:
            warnings.warn(f"[{sample_id}] Could not produce all {OPTICAL_BANDS} "
                          f"optical bands (got {len(bands_data)}); skipping.")
            return {}
    else:
        return {}

    # --- Thermal (single band) ---------------------------------------------
    thermal_out = os.path.join(sample_dir, "thermal.tif")
    if os.path.exists(thermal_path):
        ok = _reproject_and_resample(
            thermal_path, thermal_out,
            TARGET_CRS, THERMAL_COLS, THERMAL_ROWS,
            bands=[1],
            resampling_method=Resampling.bilinear,
            footprint_geom=footprint_geom,
        )
        if not ok:
            warnings.warn(f"[{sample_id}] Thermal footprint outside tile; "
                          f"writing zero-valued placeholder.")
            _write_zero_thermal(thermal_out)
        thermal_result = thermal_out
    else:
        warnings.warn(f"[{sample_id}] Thermal file not found: {thermal_path}; "
                      f"writing zero-valued placeholder.")
        _write_zero_thermal(thermal_out)
        thermal_result = thermal_out

    # --- Verify shapes -------------------------------------------------------
    with rasterio.open(optical_out) as o:
        assert o.count == OPTICAL_BANDS, f"Optical band count mismatch: {o.count}"
        assert o.height == OPTICAL_ROWS and o.width == OPTICAL_COLS, (
            f"Optical shape mismatch: {o.height}x{o.width}"
        )
    with rasterio.open(thermal_out) as t:
        assert t.count == THERMAL_BANDS, f"Thermal band count mismatch: {t.count}"
        assert t.height == THERMAL_ROWS and t.width == THERMAL_COLS, (
            f"Thermal shape mismatch: {t.height}x{t.width}"
        )

    return {"optical_path": optical_out, "thermal_path": thermal_out}


def _write_zero_thermal(path: str) -> None:
    """Write a zero-valued 24x32x1 uint16 GeoTIFF as a nodata placeholder."""
    meta = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": THERMAL_BANDS,
        "height": THERMAL_ROWS,
        "width": THERMAL_COLS,
        "crs": TARGET_CRS,
        "transform": from_bounds(77.05, 28.45, 77.35, 28.75,
                                  THERMAL_COLS, THERMAL_ROWS),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(np.zeros((THERMAL_BANDS, THERMAL_ROWS, THERMAL_COLS),
                           dtype=np.uint16))


# ---------------------------------------------------------------------------
# CLI: process all manifest_raw.csv pairs
# ---------------------------------------------------------------------------

def process_manifest(
    manifest_path: str = MANIFEST_RAW_PATH,
    out_dir: str = RESAMPLED_DIR,
) -> List[Dict[str, str]]:
    """Read manifest_raw.csv, identify optical+thermal tile pairs, and call
    resample_to_footprint() for each.  Returns a list of result dicts.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    optical_rows = [r for r in rows if r["source"] == "sentinel2"]
    thermal_rows = [r for r in rows if r["source"] == "landsat"]

    if not optical_rows:
        raise ValueError("No Sentinel-2 rows found in manifest_raw.csv")

    results = []
    for idx, opt_row in enumerate(optical_rows):
        sample_id = f"sample_{idx:04d}"

        # Build multi-band optical path list from the single recorded B04 path
        # by substituting band suffixes (B04, B03, B02) — mirrors download_tiles.py layout
        b04_path = opt_row.get("path", "")
        scene_dir = os.path.dirname(b04_path)
        optical_paths = [
            os.path.join(scene_dir, "B04.tif"),
            os.path.join(scene_dir, "B03.tif"),
            os.path.join(scene_dir, "B02.tif"),
        ]

        # Use first thermal row if available (best-effort; ideally co-temporal)
        thermal_path = ""
        if thermal_rows:
            t_idx = idx % len(thermal_rows)
            thermal_path = thermal_rows[t_idx].get("thermal_band_path", "")

        result = resample_to_footprint(
            optical_paths=optical_paths,
            thermal_path=thermal_path,
            sample_id=sample_id,
            out_dir=out_dir,
        )
        if result:
            result["sample_id"] = sample_id
            result["scene_id_optical"] = opt_row.get("scene_id", "")
            result["scene_id_thermal"] = (thermal_rows[t_idx].get("scene_id", "")
                                           if thermal_rows else "")
            result["date"] = opt_row.get("acquisition_date", "")
            results.append(result)
            print(f"[OK]  {sample_id} -> {result['optical_path']}")
        else:
            print(f"[SKIP] {sample_id} — footprint outside available tiles")

    return results


if __name__ == "__main__":
    print("=== Task 3.3: resample_pipeline.py ===")
    print(f"Target GSD : {PROVISIONAL_GSD_M} m (PROVISIONAL)")
    print(f"Optical    : {OPTICAL_ROWS} x {OPTICAL_COLS} x {OPTICAL_BANDS}")
    print(f"Thermal    : {THERMAL_ROWS} x {THERMAL_COLS} x {THERMAL_BANDS}")
    print()
    results = process_manifest()
    print(f"\nProcessed {len(results)} sample(s).")
    print(f"Output written to: {os.path.abspath(RESAMPLED_DIR)}")
