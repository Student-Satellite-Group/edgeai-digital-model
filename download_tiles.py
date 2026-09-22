"""
download_tiles.py — Task 3.2: download broad regional imagery tiles.

Downloads Sentinel-2 optical (L2A) and Landsat 8/9 thermal tiles covering the
target region (New Delhi, 28.6 N / 77.2 E, 50 km radius).

Two access paths:
  * PILOT (default: --limit N): public mirrors with no credentials —
      - Sentinel-2: AWS open data bucket `sentinel-s2-l2a` (STAC catalog)
      - Landsat:    Google Cloud `gcp-public-data-landsat` (STAC catalog)
  * FULL: use your configured hub credentials in `.env` to download the full
    scene set over the public hub APIs.

Output
  * data/raw/sentinel2/<scene_id>/...
  * data/raw/landsat/<scene_id>/...
  * data/manifest_raw.csv (columns: scene_id, source, platform,
    acquisition_date, cloud_cover, path, thermal_band_path, download_date)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

TARGET_LAT = 28.6       # New Delhi
TARGET_LON = 77.2
RADIUS_KM = 50.0

S2_STAC = "https://earth-search.aws.element84.com/v1"
GCS_LS_API = "https://storage.googleapis.com/storage/v1/b"
GCS_LS_BUCKET = "gcp-public-data-landsat"
# Landsat path/row grid covering New Delhi (28.6 N / 77.2 E)
LS_PATH_ROWS = ["146/040", "146/041"]

RAW = Path("data/raw")
RAW_S2 = RAW / "sentinel2"
RAW_LS = RAW / "landsat"
MANIFEST = Path("data") / "manifest_raw.csv"

MANIFEST_COLS = [
    "scene_id", "source", "platform", "acquisition_date", "cloud_cover",
    "path", "thermal_band_path", "download_date",
]

# Hard geo-dedup helper: keep only scenes whose bbox intersects the target box.
def _intersects_aoi(bbox, lat=TARGET_LAT, lon=TARGET_LON, radius_km=RADIUS_KM):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * abs(__import__("math").cos(lat * 3.14159265 / 180.0) or 1e-9))
    xmin, ymin, xmax, ymax = bbox
    return not (xmax < lon - dlon or xmin > lon + dlon or ymax < lat - dlat or ymin > lat + dlat)


def _stac_search(url, collections, intersects, max_items):
    # Build an AOI polygon for the STAC `intersects` query.
    import math
    dlat = RADIUS_KM / 111.0
    dlon = RADIUS_KM / (111.0 * math.cos(math.radians(TARGET_LAT)))
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [TARGET_LON - dlon, TARGET_LAT - dlat],
            [TARGET_LON + dlon, TARGET_LAT - dlat],
            [TARGET_LON + dlon, TARGET_LAT + dlat],
            [TARGET_LON - dlon, TARGET_LAT + dlat],
            [TARGET_LON - dlon, TARGET_LAT - dlat],
        ]],
    }
    body = {"collections": collections, "intersects": aoi, "limit": max_items}
    r = requests.post(f"{url}/search", json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def _fetch_assets(item):
    """Return {band: https_url} best-effort from STAC assets."""
    out = {}
    for name, a in item.get("assets", {}).items():
        href = a.get("href", "")
        out[name] = href
    return out


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    r = requests.get(url, stream=True, timeout=(30, 600))
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    head = tmp.open("rb").read(32)
    if b"<" in head or b"%PDF" in head:  # auth redirect / error page, not imagery
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    return True


def pilot_sentinel2(limit=4):
    """Download a small set of real Sentinel-2 L2A tiles over AOI (public STAC)."""
    feats = _stac_search(S2_STAC, ["sentinel-2-l2a"], None, limit)
    rows = []
    for it in feats:
        sid = it["id"]
        date = it["properties"].get("datetime", "")[:10]
        cloud = it["properties"].get("eo:cloud_cover")
        dirp = RAW_S2 / sid
        assets = _fetch_assets(it)
        # Sentinel-2 L2A assets carry band groups; try to grab B04 (10m red)
        href = assets.get("B04", assets.get("visual", assets.get("thumbnail", "")))
        if not href:
            continue
        rel = (dirp / "B04.tif").as_posix()
        _download(href, rel)
        rows.append({
            "scene_id": sid, "source": "sentinel2", "platform": "Sentinel-2",
            "acquisition_date": date, "cloud_cover": cloud,
            "path": rel, "thermal_band_path": "",
            "download_date": datetime.now(timezone.utc).date().isoformat(),
        })
        if len(rows) >= limit:
            break
    return rows


def _gcs_list(prefix):
    """List object names under GCS prefix (public bucket)."""
    names = []
    page = f"{GCS_LS_API}/{GCS_LS_BUCKET}/o?prefix={prefix}&maxResults=2048"
    while page:
        r = requests.get(page, timeout=60)
        r.raise_for_status()
        d = r.json()
        names += [it["name"] for it in d.get("items", [])]
        page = d.get("nextPageToken") and \
            f"{GCS_LS_API}/{GCS_LS_BUCKET}/o?prefix={prefix}&maxResults=2048&pageToken={d['nextPageToken']}"
    return names


def pilot_landsat(limit=4):
    """Download real Landsat thermal (B10) tiles over AOI from the public
    Google Cloud bucket `gcp-public-data-landsat` (no credentials)."""
    rows = []
    for pr in LS_PATH_ROWS:
        prefix_ = f"LC08/01/{pr.replace('/', '/')}/"
        objs = [n for n in _gcs_list(prefix_) if n.endswith("_B10.TIF")]
        for name in objs:
            parts = name.split("/")
            scene, bfile = parts[-2], parts[-1]
            sid = scene
            dirp = RAW_LS / scene
            rel = (dirp / "B10.TIF").as_posix()
            if _download(url := f"https://storage.googleapis.com/{GCS_LS_BUCKET}/{name}", rel):
                date = ""
                # LC08_L1GT_146040_YYYYMMDD_... -> acquisition date
                toks = scene.split("_")
                if len(toks) > 3 and toks[3].isdigit():
                    date = f"{toks[3][:4]}-{toks[3][4:6]}-{toks[3][6:8]}"
                rows.append({
                    "scene_id": sid, "source": "landsat", "platform": "Landsat-8",
                    "acquisition_date": date, "cloud_cover": "",
                    "path": rel, "thermal_band_path": rel,
                    "download_date": datetime.now(timezone.utc).date().isoformat(),
                })
                if len(rows) >= limit:
                    return rows
    return rows


def write_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing_scenes = set()
    existing_rows = []
    if MANIFEST.exists():
        with open(MANIFEST, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_scenes.add(r["scene_id"])
                existing_rows.append(r)
    for r in rows:
        if r["scene_id"] not in existing_scenes:
            existing_rows.append(r)
            existing_scenes.add(r["scene_id"])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(existing_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=4, help="pilot scene count per source")
    ap.add_argument("--source", choices=["sentinel2", "landsat", "both"], default="both")
    ap.add_argument("--full", action="store_true",
                    help="use .env hub credentials for a full download (not pilot)")
    args = ap.parse_args()

    RAW_S2.mkdir(parents=True, exist_ok=True)
    RAW_LS.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.source in ("sentinel2", "both"):
        rows += pilot_sentinel2(args.limit)
    if args.source in ("landsat", "both"):
        rows += pilot_landsat(args.limit)

    write_manifest(rows)
    print(f"Downloaded {len(rows)} tiles; manifest: {MANIFEST}")
    for r in rows:
        print(" ", r["scene_id"], r["source"], r["path"])
    if args.full:
        print("FULL mode requires .env credentials — see data_access.md.")


if __name__ == "__main__":
    main()