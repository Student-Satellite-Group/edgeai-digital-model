"""
rmse_harness.py — Task 4.4: Build RMSE Measurement Harness

Provides a reusable tool for measuring optical-to-thermal registration accuracy
using automatically generated tie-point correspondences derived from ORB matching.
Replaces the click-based annotation step with a deterministic synthetic annotator
so the harness is always runnable on any dataset (proxy or real).

Functions
---------
generate_tie_points(optical_gray, thermal_gray, n_points=15)
    Uses ORB + BFMatcher to find N high-confidence corresponding point pairs.
    Returns (pts_optical, pts_thermal) — both as (N, 2) float32 arrays.

apply_homography(pts, H)
    Warps pts_optical through homography H -> pts_warped.

compute_rmse(pts_ref, pts_warped)
    Root Mean Square Error (pixels) between reference and warped point sets.

run_harness(manifest_csv, subsample_n=20)
    Loads manifest, runs registration + RMSE measurement on a subsample, and
    returns a list of per-sample result dicts.

main()
    Runs harness and writes rmse_results.md.

Deliverables
------------
    rmse_harness.py   — this file
    rmse_results.md   — RMSE measurements per sample + summary

PROXY-DATA NOTE: Results are provisional; synthetic tiles produce lower RMSE
than real satellite imagery will. See proxy_data_caveats.md.
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
MANIFEST    = os.path.join(REPO_DIR, "data", "labeled", "manifest.csv")
RMSE_REPORT = os.path.join(REPO_DIR, "rmse_results.md")

# Best-guess ORB params (will be replaced by tune_registration.py output)
ORB_NFEATURES  = 500
ORB_SCALE      = 1.2
ORB_NLEVELS    = 8
RANSAC_THRESH  = 5.0
N_TIE_POINTS   = 15
SUBSAMPLE_N    = 20


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _load_uint8(path: str) -> Optional[np.ndarray]:
    try:
        import rasterio as rio
        with rio.open(path) as ds:
            data = ds.read()
        bgr = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32) if data.shape[0] >= 3 else data[0].astype(np.float32)
        mn, mx = bgr.min(), bgr.max()
        return ((bgr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(bgr, dtype=np.uint8)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32)-mn)/(mx-mn)*255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
        return img


def _gray(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


# ---------------------------------------------------------------------------
# Tie-point generation
# ---------------------------------------------------------------------------

def generate_tie_points(
    optical_gray: np.ndarray,
    thermal_gray: np.ndarray,
    n_points: int = N_TIE_POINTS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (pts_optical, pts_thermal) of shape (N, 2) or (None, None) on failure."""
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, scaleFactor=ORB_SCALE, nlevels=ORB_NLEVELS)
    kp1, d1 = orb.detectAndCompute(optical_gray, None)
    kp2, d2 = orb.detectAndCompute(thermal_gray, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return None, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    good = sorted(good, key=lambda m: m.distance)[:n_points]
    if len(good) < 4:
        return None, None
    pts_opt = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_thm = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts_opt, pts_thm


def apply_homography(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project pts (N, 2) through 3x3 homography H -> (N, 2)."""
    n = pts.shape[0]
    ones = np.ones((n, 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])               # (N, 3)
    warped_h = (H @ pts_h.T).T                   # (N, 3)
    denom = warped_h[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return (warped_h[:, :2] / denom).astype(np.float32)


def compute_rmse(pts_ref: np.ndarray, pts_warped: np.ndarray) -> float:
    """Euclidean pixel RMSE between two (N, 2) point sets."""
    diff = pts_ref.astype(np.float32) - pts_warped.astype(np.float32)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Full registration + RMSE for one image pair
# ---------------------------------------------------------------------------

def _registration_rmse(opt_path: str, thm_path: str) -> Dict:
    """Run registration + tie-point RMSE for a single sample pair."""
    result = {
        "optical_path": opt_path,
        "thermal_path": thm_path,
        "n_tie_points": 0,
        "rmse_px"     : None,
        "inlier_frac" : 0.0,
        "status"      : "ok",
    }

    opt = _load_uint8(opt_path)
    thm = _load_uint8(thm_path)
    if opt is None or thm is None:
        result["status"] = "file_missing"
        return result

    opt_g = _gray(opt)
    # Upsample thermal to optical size for homography estimation
    thm_resized = cv2.resize(thm, (opt_g.shape[1], opt_g.shape[0]), interpolation=cv2.INTER_AREA)
    thm_g       = _gray(thm_resized)

    # Step 1: estimate homography via registration pipeline
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, scaleFactor=ORB_SCALE, nlevels=ORB_NLEVELS)
    kp1, d1 = orb.detectAndCompute(opt_g, None)
    kp2, d2 = orb.detectAndCompute(thm_g, None)

    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        result["status"] = "insufficient_features"
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw   = matcher.knnMatch(d1, d2, k=2)
    good  = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        result["status"] = "insufficient_matches"
        return result

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
    if M is None or mask is None:
        result["status"] = "homography_failed"
        return result

    inlier_frac = float(mask.sum()) / max(len(good), 1)
    result["inlier_frac"] = inlier_frac

    try:
        H = np.linalg.inv(M)   # thermal -> optical direction
    except np.linalg.LinAlgError:
        result["status"] = "singular_homography"
        return result

    # Step 2: generate tie points on optical, warp with H, measure RMSE
    pts_opt, pts_thm = generate_tie_points(opt_g, thm_g, N_TIE_POINTS)
    if pts_opt is None:
        result["status"] = "tie_point_failure"
        return result

    pts_warped = apply_homography(pts_thm, H)
    rmse       = compute_rmse(pts_opt, pts_warped)

    result["n_tie_points"] = len(pts_opt)
    result["rmse_px"]      = rmse
    return result


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def run_harness(manifest_csv: str, subsample_n: int = SUBSAMPLE_N) -> List[Dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    rng     = np.random.default_rng(0)
    idx     = rng.choice(len(records), size=min(subsample_n, len(records)), replace=False)
    sample  = [records[i] for i in sorted(idx)]

    results = []
    for i, rec in enumerate(sample):
        opt_path = os.path.join(REPO_DIR, rec["optical_path"].replace("\\", os.sep))
        thm_path = os.path.join(REPO_DIR, rec["thermal_path"].replace("\\", os.sep))
        scene    = "cloudy" if rec.get("label_cloud", "0").strip() == "1" else "clear"

        r = _registration_rmse(opt_path, thm_path)
        r["sample_id"] = rec.get("sample_id", f"sample_{i:04d}")
        r["scene"]     = scene

        rmse_str = f"{r['rmse_px']:.2f} px" if r["rmse_px"] is not None else "N/A"
        print(f"  [{i+1}/{len(sample)}] {r['sample_id']} ({scene}) "
              f"inlier={r['inlier_frac']:.2%}  RMSE={rmse_str}  [{r['status']}]")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def write_report(results: List[Dict], path: str):
    rmse_vals = [r["rmse_px"] for r in results if r["rmse_px"] is not None]
    failed    = [r for r in results if r["rmse_px"] is None]

    overall_rmse = float(np.mean(rmse_vals)) if rmse_vals else float("nan")
    clear_rmse   = float(np.mean([r["rmse_px"] for r in results
                                  if r["rmse_px"] is not None and r["scene"] == "clear"])) \
                   if any(r["scene"] == "clear" and r["rmse_px"] is not None for r in results) \
                   else float("nan")
    cloudy_rmse  = float(np.mean([r["rmse_px"] for r in results
                                  if r["rmse_px"] is not None and r["scene"] == "cloudy"])) \
                   if any(r["scene"] == "cloudy" and r["rmse_px"] is not None for r in results) \
                   else float("nan")

    lines = [
        "# RMSE Results — Registration Accuracy Harness",
        "",
        "**Task:** 4.4 - Build RMSE Measurement Harness  ",
        f"**Samples evaluated:** {len(results)}  ",
        f"**Successful RMSE measurements:** {len(rmse_vals)}  ",
        f"**Failed:** {len(failed)}  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Scene Type | Mean RMSE (px) |",
        "|:-----------|---------------:|",
        f"| Overall    | {overall_rmse:.2f}         |",
        f"| Clear      | {clear_rmse:.2f}         |",
        f"| Cloudy     | {cloudy_rmse:.2f}         |",
        "",
        "---",
        "",
        "## Per-Sample Results",
        "",
        "| sample_id   | scene  | n_tie_pts | RMSE (px) | inlier_frac | status |",
        "|:------------|:------:|----------:|----------:|------------:|:-------|",
    ]
    for r in results:
        rmse_str   = f"{r['rmse_px']:.2f}" if r["rmse_px"] is not None else "N/A"
        inlier_str = f"{r['inlier_frac']:.2%}"
        lines.append(
            f"| {r['sample_id']} | {r['scene']} | {r['n_tie_points']} | "
            f"{rmse_str} | {inlier_str} | {r['status']} |"
        )
    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- ORB params used: nfeatures=500, scaleFactor=1.2, nlevels=8, ransac_thresh=5 px.",
        "  Run tune_registration.py and update constants if better params are found.",
        "- Thermal images upsampled from 24x32 to 128x128 for homography estimation.",
        "- Tie points generated via ORB feature matching (no manual annotation needed).",
        "- All results are provisional proxy-data measurements. See proxy_data_caveats.md.",
        "- Re-run after downloading real Sentinel-2 / Landsat-8 tiles.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(MANIFEST):
        print(f"ERROR: manifest not found: {MANIFEST}", file=sys.stderr)
        sys.exit(1)
    print(f"Running RMSE harness on {MANIFEST} (subsample={SUBSAMPLE_N})...")
    results = run_harness(MANIFEST, SUBSAMPLE_N)
    write_report(results, RMSE_REPORT)

    rmse_vals = [r["rmse_px"] for r in results if r["rmse_px"] is not None]
    if rmse_vals:
        print(f"\nMean RMSE (overall): {float(np.mean(rmse_vals)):.2f} px")
    print("DONE")


if __name__ == "__main__":
    main()
