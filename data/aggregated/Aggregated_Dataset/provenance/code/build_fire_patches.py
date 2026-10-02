"""Cut one Landsat 9 L2 scene into 128x128 patches: RGB (OLI) + thermal (TIRS ST) from the SAME scene, fire label from OLI SWIR.

RGB     : uint8, clip((DN*2.75e-5 - 0.2)/0.3, 0, 1)*255, order R,G,B
Thermal : 24x32 (H x W) grid, area-averaged from the 128x128 window, Kelvin; patch kept only if >= 95% thermal-valid
Label   : positive = >= 2 SWIR fire pixels (R75 = rho7/rho5 > 1.8 and rho7 - rho5 > 0.17); negative = 0; 1 px = ambiguous (dropped)
usage: python build_fire_patches.py <scene_id>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

P = 128
SID = sys.argv[1]
D = Path(__file__).parent / "landsat" / SID
ds = {k: rasterio.open(D / f"{k}.tif") for k in ("swir22", "nir08", "lwir11", "red", "green", "blue")}
H, W = ds["swir22"].height, ds["swir22"].width
assert all((d.height, d.width) == (H, W) for d in ds.values()), "band shapes differ"
nr, nc = H // P, W // P
print(SID, "scene", H, "x", W, "| patch grid", nr, "x", nc, flush=True)

rgb, thm, y, nfire, rc = [], [], [], [], []
stats = dict(total=0, edge_invalid=0, ambiguous=0, no_thermal_pos=0, no_thermal_neg=0)
for r in range(nr):
    win = Window(0, r * P, nc * P, P)
    band = {k: d.read(1, window=win) for k, d in ds.items()}
    sr = lambda a: a.astype(np.float32) * 2.75e-5 - 0.2
    valid = (band["swir22"] > 0) & (band["nir08"] > 0)
    p7, p5 = sr(band["swir22"]), sr(band["nir08"])
    with np.errstate(divide="ignore", invalid="ignore"):
        fire = (p7 / p5 > 1.8) & ((p7 - p5) > 0.17) & valid
    Kst = np.where(band["lwir11"] > 0, band["lwir11"].astype(np.float32) * 0.00341802 + 149.0, np.nan)
    img = np.stack([np.clip(sr(band[k]) / 0.3, 0, 1) for k in ("red", "green", "blue")], axis=-1)
    for c in range(nc):
        sl = (slice(None), slice(c * P, (c + 1) * P))
        stats["total"] += 1
        if valid[sl].mean() < 0.95:
            stats["edge_invalid"] += 1
            continue
        n = int(fire[sl].sum())
        if n == 1:
            stats["ambiguous"] += 1
            continue
        t = Kst[sl]
        if np.isfinite(t).mean() < 0.95:
            stats["no_thermal_pos" if n >= 2 else "no_thermal_neg"] += 1
            continue
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        thm.append(cv2.resize(t, (32, 24), interpolation=cv2.INTER_AREA))
        rgb.append((img[sl] * 255).round().astype(np.uint8))
        y.append(int(n >= 2)); nfire.append(n); rc.append((r, c))
    if r % 10 == 0:
        print(f"  row {r}/{nr} kept {len(y)}", flush=True)

rgb, thm, y, nfire, rc = np.stack(rgb), np.stack(thm).astype(np.float32), np.array(y), np.array(nfire), np.array(rc)
np.savez_compressed(D / "fire_patches.npz", rgb=rgb, thm=thm, y=y, nfire=nfire, rc=rc)
print("\npatch accounting:", stats)
print("KEPT %d patches | positive %d (%.1f%%) | negative %d" % (len(y), y.sum(), 100 * y.mean(), (1 - y).sum()))
print("thermal K: min %.1f  median %.1f  max %.1f" % (thm.min(), np.median(thm), thm.max()))
