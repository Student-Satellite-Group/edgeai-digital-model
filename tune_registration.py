import csv, itertools, os, sys
from typing import Dict, List, Tuple
import cv2
import numpy as np

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
MANIFEST   = os.path.join(REPO_DIR, "data", "labeled", "manifest.csv")
TUNING_LOG = os.path.join(REPO_DIR, "registration_tuning_log.md")
SUBSAMPLE_N = 20
PARAM_GRID = {
    "nfeatures"    : [200, 500, 1000],
    "scaleFactor"  : [1.1, 1.2, 1.3],
    "nlevels"      : [6, 8, 10],
    "ransac_thresh": [3, 5, 10],
}
INTERIOR_FRAC = 0.15


def _load_gray_uint8(path):
    try:
        import rasterio as rio
        with rio.open(path) as ds:
            data = ds.read()
        bgr = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32) if data.shape[0] >= 3 else data[0].astype(np.float32)
        mn, mx = bgr.min(), bgr.max()
        bgr = (bgr - mn) / (mx - mn) * 255.0 if mx > mn else bgr
        return bgr.astype(np.uint8)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32)-mn)/(mx-mn)*255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
        return img


def _to_gray(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


def _compute_rmse(ref, reg):
    h, w = ref.shape
    f = INTERIOR_FRAC
    r0, r1, c0, c1 = int(h*f), int(h*(1-f)), int(w*f), int(w*(1-f))
    diff = ref[r0:r1, c0:c1].astype(np.float32) - reg[r0:r1, c0:c1].astype(np.float32)
    return float(np.sqrt(np.mean(diff**2)))


def _register(ref, mov, nfeatures, scaleFactor, nlevels, ransac_thresh):
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels)
    kp1, d1 = orb.detectAndCompute(ref, None)
    kp2, d2 = orb.detectAndCompute(mov, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return 0.0, 255.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0.0, 255.0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if M is None or mask is None:
        return 0.0, 255.0
    inlier_frac = float(mask.sum()) / max(len(good), 1)
    try:
        H = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return inlier_frac, 255.0
    h, w = ref.shape
    registered = cv2.warpPerspective(mov, H, (w, h))
    return inlier_frac, _compute_rmse(ref, registered)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_sweep(records):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(records), size=min(SUBSAMPLE_N, len(records)), replace=False)
    sub    = [records[i] for i in sorted(idx)]
    print(f"Grid search: {total} combinations x {len(sub)} samples")
    all_results = []
    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        rmse_s   = {"clear": [], "cloudy": []}
        inlier_s = {"clear": [], "cloudy": []}
        for rec in sub:
            opt_path = os.path.join(REPO_DIR, rec["optical_path"].replace("\\", os.sep))
            thm_path = os.path.join(REPO_DIR, rec["thermal_path"].replace("\\", os.sep))
            scene    = "cloudy" if rec.get("label_cloud", "0").strip() == "1" else "clear"
            opt = _to_gray(_load_gray_uint8(opt_path))
            thm = _to_gray(_load_gray_uint8(thm_path))
            if opt is None or thm is None:
                continue
            mov = cv2.resize(thm, (opt.shape[1], opt.shape[0]), interpolation=cv2.INTER_AREA)
            inlier, rmse = _register(opt, mov, **params)
            rmse_s[scene].append(rmse)
            inlier_s[scene].append(inlier)

        def mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        r = {**params,
             "rmse_clear"    : mean(rmse_s["clear"]),
             "rmse_cloudy"   : mean(rmse_s["cloudy"]),
             "rmse_overall"  : mean(rmse_s["clear"] + rmse_s["cloudy"]),
             "inlier_clear"  : mean(inlier_s["clear"]),
             "inlier_cloudy" : mean(inlier_s["cloudy"]),
             "inlier_overall": mean(inlier_s["clear"] + inlier_s["cloudy"]),
             "n_clear"       : len(rmse_s["clear"]),
             "n_cloudy"      : len(rmse_s["cloudy"])}
        all_results.append(r)
        if (ci + 1) % 10 == 0 or ci == total - 1:
            print(f"  [{ci+1}/{total}] nf={params['nfeatures']} sf={params['scaleFactor']} "
                  f"nl={params['nlevels']} rt={params['ransac_thresh']} "
                  f"-> RMSE={r['rmse_overall']:.2f} inlier={r['inlier_overall']:.2%}")
    return all_results


def find_best(results):
    valid = [r for r in results if not (isinstance(r["rmse_overall"], float) and r["rmse_overall"] != r["rmse_overall"])]
    return min(valid, key=lambda r: r["rmse_overall"]) if valid else results[0]


def flag_outlier_scenes(results, best):
    avg   = best["rmse_overall"]
    flags = []
    for scene in ("clear", "cloudy"):
        v = best[f"rmse_{scene}"]
        if v == v and v > 2 * avg:   # NaN-safe check
            flags.append(f"Scene '{scene}' RMSE ({v:.2f}) > 2x overall ({avg:.2f}).")
    return flags


def write_log(results, best, flags, path):
    lines = [
        "# Registration Tuning Log", "",
        "**Task:** 4.2 - Tune Registration on Proxy Data  ",
        "**Dataset:** data/labeled/manifest.csv (Phase 3 synthetic proxy)  ",
        f"**Subsample per combo:** {SUBSAMPLE_N}  ",
        "**Grid:** nfeatures x scaleFactor x nlevels x ransac_thresh  ",
        "", "---", "", "## Best Parameter Set", "",
        "| Parameter        | Value |",
        "|:-----------------|:------|",
        f"| nfeatures        | {best['nfeatures']} |",
        f"| scaleFactor      | {best['scaleFactor']} |",
        f"| nlevels          | {best['nlevels']} |",
        f"| ransac_thresh    | {best['ransac_thresh']} px |",
        f"| RMSE (overall)   | {best['rmse_overall']:.4f} |",
        f"| RMSE (clear)     | {best['rmse_clear']:.4f} |",
        f"| RMSE (cloudy)    | {best['rmse_cloudy']:.4f} |",
        f"| Inlier (overall) | {best['inlier_overall']:.2%} |",
        f"| Inlier (clear)   | {best['inlier_clear']:.2%} |",
        f"| Inlier (cloudy)  | {best['inlier_cloudy']:.2%} |",
        "", "---", "", "## Outlier Scene Flags", "",
    ]
    if flags:
        for fl in flags:
            lines.append(f"- **WARNING:** {fl}")
    else:
        lines.append("No scene types flagged (all RMSE within 2x overall average).")
    lines += [
        "", "---", "", "## Full Results Table (sorted by RMSE overall)", "",
        "| nfeatures | scaleFactor | nlevels | ransac_thresh | RMSE_overall | RMSE_clear | RMSE_cloudy | inlier_overall | inlier_clear | inlier_cloudy |",
        "|----------:|------------:|--------:|--------------:|-------------:|-----------:|------------:|---------------:|-------------:|--------------:|",
    ]
    key = lambda x: x["rmse_overall"] if x["rmse_overall"] == x["rmse_overall"] else 9999
    for r in sorted(results, key=key):
        lines.append(
            f"| {r['nfeatures']} | {r['scaleFactor']} | {r['nlevels']} | "
            f"{r['ransac_thresh']} | {r['rmse_overall']:.2f} | {r['rmse_clear']:.2f} | "
            f"{r['rmse_cloudy']:.2f} | {r['inlier_overall']:.2%} | "
            f"{r['inlier_clear']:.2%} | {r['inlier_cloudy']:.2%} |"
        )
    lines += [
        "", "---", "", "## Caveats", "",
        "- Dataset is **synthetic proxy**. Re-run after downloading real Sentinel-2/Landsat tiles.",
        "- Thermal (24x32) upsampled to optical resolution (128x128) for ORB feature detection.",
        "- RMSE computed over central 70% of pixels (15% border excluded).",
        "- 22-min temporal parallax (Sentinel-2 vs Landsat-8) causes systematic misalignment.",
        "  See proxy_data_caveats.md.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


def main():
    if not os.path.isfile(MANIFEST):
        print(f"ERROR: manifest not found: {MANIFEST}", file=sys.stderr)
        sys.exit(1)
    records = load_manifest(MANIFEST)
    print(f"Loaded {len(records)} samples.")
    results = run_sweep(records)
    best    = find_best(results)
    flags   = flag_outlier_scenes(results, best)
    write_log(results, best, flags, TUNING_LOG)
    print("\n=== Best Params ===")
    for k in ["nfeatures", "scaleFactor", "nlevels", "ransac_thresh"]:
        print(f"  {k} = {best[k]}")
    print(f"  RMSE overall   = {best['rmse_overall']:.4f}")
    print(f"  Inlier overall = {best['inlier_overall']:.2%}")
    if flags:
        for fl in flags:
            print(f"  WARNING: {fl}")
    print("\nDONE")


if __name__ == "__main__":
    main()
