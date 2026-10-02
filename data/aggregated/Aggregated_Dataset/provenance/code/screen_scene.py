"""Screen a downloaded Landsat L2 scene for SWIR-detected fires and check how they look in the ST band.
usage: python screen_scene.py <scene_id>   (needs swir22.tif, nir08.tif, lwir11.tif in landsat/<scene_id>/)"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage

P = 128
SID = sys.argv[1]
D = Path(__file__).parent / "landsat" / SID


def load(name):
    with rasterio.open(D / f"{name}.tif") as ds:
        return ds.read(1)


b7, b5, st = load("swir22"), load("nir08"), load("lwir11")
H, W = b7.shape
print(SID, "| scene", H, "x", W)

valid = (b7 > 0) & (b5 > 0)
p7 = np.where(valid, b7.astype(np.float32) * 2.75e-5 - 0.2, np.nan)
p5 = np.where(valid, b5.astype(np.float32) * 2.75e-5 - 0.2, np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    fire = (p7 / p5 > 1.8) & ((p7 - p5) > 0.17) & valid
del p7, p5
print("valid SR px: %.1f M | SWIR fire pixels: %d" % (valid.sum() / 1e6, fire.sum()))
lab, ncomp = ndimage.label(fire, structure=np.ones((3, 3)))
print("connected fire clusters:", ncomp)

st_fill = (st == 0) & valid
print("ST fill (DN=0) inside valid SR area: %.4f%% of px" % (100 * st_fill.sum() / valid.sum()))
print("ST fill at SWIR-fire pixels: %.1f%%" % (100 * (st[fire] == 0).mean() if fire.any() else 0))
print("ST saturated-high (>=65000) at fire pixels: %.1f%%" % (100 * (st[fire] >= 65000).mean() if fire.any() else 0))
K = np.where(st > 0, st.astype(np.float32) * 0.00341802 + 149.0, np.nan)
if fire.any():
    fk = K[fire]
    print("ST at fire px (valid only): median %.1f K, p90 %.1f K, max %.1f K | scene median %.1f K"
          % (np.nanmedian(fk), np.nanpercentile(fk, 90), np.nanmax(fk), np.nanmedian(K)))

# patch-level view at the payload's coarse thermal resolution (128x128 px -> 24x32)
nr, nc = H // P, W // P
n_fire = np.zeros((nr, nc), int)
contrast = np.full((nr, nc), np.nan, np.float32)
frac_missing = np.zeros((nr, nc), np.float32)
for r in range(nr):
    for c in range(nc):
        sl = (slice(r * P, (r + 1) * P), slice(c * P, (c + 1) * P))
        n_fire[r, c] = fire[sl].sum()
        t = K[sl]
        frac_missing[r, c] = np.isnan(t).mean()
        if np.isfinite(t).mean() >= 0.95:
            g = cv2.resize(np.where(np.isfinite(t), t, np.nanmean(t)), (32, 24), interpolation=cv2.INTER_AREA)
            contrast[r, c] = g.max() - np.median(g)
pos, neg = n_fire >= 2, n_fire == 0
print("\npatches: %d total | positive (>=2 fire px): %d | ambiguous (1 px): %d | negative (0): %d"
      % (n_fire.size, pos.sum(), (n_fire == 1).sum(), neg.sum()))
for nm, m in (("positive", pos), ("negative", neg)):
    v = contrast[m & np.isfinite(contrast)]
    if len(v):
        print("  thermal contrast (max - median of 24x32 grid, K) %-8s n=%4d | median %.2f | p25 %.2f | p75 %.2f | p90 %.2f"
              % (nm, len(v), np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.percentile(v, 90)))
print("  mean fraction of thermal missing in positive patches: %.3f | negative: %.3f"
      % (frac_missing[pos].mean() if pos.any() else 0, frac_missing[neg].mean()))
np.savez_compressed(D / "screen.npz", n_fire=n_fire, contrast=contrast, frac_missing=frac_missing)

pv, nv = contrast[pos & np.isfinite(contrast)], contrast[neg & np.isfinite(contrast)]
auc = float("nan")
if len(pv) >= 5 and len(nv) >= 5:
    ranks = np.concatenate([pv, nv]).argsort().argsort()
    auc = (ranks[:len(pv)].sum() - len(pv) * (len(pv) - 1) / 2) / (len(pv) * len(nv))
print("SUMMARY %s | positive patches %d (with thermal %d) | negative %d | fire px %d | thermal-contrast AUC %.3f | ST fill at fire px %.1f%%"
      % (SID, pos.sum(), len(pv), neg.sum(), fire.sum(), auc, 100 * (st[fire] == 0).mean() if fire.any() else 0))
