"""Cut the cached 20 m scene into non-overlapping 128x128 patches (2.56 km) with labels.

RGB   : uint8, DN/6000 clipped (like an 8-bit camera; no reflectance offset assumed)
Thermal: 24x32 (H x W) grid per patch (area-averaged from the 128x128 window), Kelvin, NaN if < 95% valid
Labels : cloud  = SCL{8,9,10} fraction >= 0.30            (project rule, task_definition.md)
         veg    = SCL4 / (SCL4 + SCL5) >= 0.5, only where SCL4+SCL5 covers >= 50% of the patch
"""

from pathlib import Path

import cv2
import numpy as np

P = 128
HERE = Path(__file__).parent
d = np.load(HERE / "scene_cache.npz")
rgb_full = np.dstack([d["red"], d["green"], d["blue"]])  # R, G, B order
scl, st = d["scl"], d["st"]
n = scl.shape[0] // P

rgb, thm, cloud_frac, n4, n5, rc = [], [], [], [], [], []
for r in range(n):
    for c in range(n):
        sl = (slice(r * P, (r + 1) * P), slice(c * P, (c + 1) * P))
        rgb.append(np.clip(rgb_full[sl].astype(np.float32) / 6000.0, 0, 1))
        s = scl[sl]
        cloud_frac.append(np.isin(s, [8, 9, 10]).mean())
        n4.append((s == 4).mean()); n5.append((s == 5).mean())
        t = st[sl]
        if np.isfinite(t).mean() >= 0.95:
            t = np.where(np.isfinite(t), t, np.nanmean(t))
            thm.append(cv2.resize(t, (32, 24), interpolation=cv2.INTER_AREA))  # dsize = (W, H)
        else:
            thm.append(np.full((24, 32), np.nan, np.float32))
        rc.append((r, c))

rgb = (np.stack(rgb) * 255).round().astype(np.uint8)
thm = np.stack(thm).astype(np.float32)
cloud_frac, n4, n5, rc = map(np.array, (cloud_frac, n4, n5, rc))
y_cloud = (cloud_frac >= 0.30).astype(np.int64)
veg_valid = (n4 + n5) >= 0.5
y_veg = np.where(veg_valid, (n4 / np.maximum(n4 + n5, 1e-9) >= 0.5), 0).astype(np.int64)
has_thermal = np.isfinite(thm).all(axis=(1, 2))

np.savez_compressed(HERE / "patches.npz", rgb=rgb, thm=thm, y_cloud=y_cloud, y_veg=y_veg,
                    veg_valid=veg_valid, has_thermal=has_thermal, cloud_frac=cloud_frac,
                    veg_frac=np.where(veg_valid, n4 / np.maximum(n4 + n5, 1e-9), np.nan), rc=rc,
                    nd_rc=d["nd_rc"])

nd_patch = tuple(int(v) // P for v in d["nd_rc"])
print(f"grid {n}x{n} = {len(rgb)} patches | with thermal: {has_thermal.sum()}")
for name, m in (("ALL", np.ones(len(rgb), bool)), ("THERMAL SUBSET", has_thermal)):
    vm = m & veg_valid
    print(f"[{name}] cloud task: n={m.sum()} pos={y_cloud[m].sum()} ({y_cloud[m].mean():.2f}) | "
          f"veg task: n={vm.sum()} pos={y_veg[vm].sum()} ({y_veg[vm].mean():.2f})")
i = int(np.where((rc[:, 0] == nd_patch[0]) & (rc[:, 1] == nd_patch[1]))[0][0])
print("New Delhi patch (r,c):", nd_patch, "idx", i, "| thermal:", bool(has_thermal[i]),
      "| cloud_frac %.2f -> y_cloud %d | veg_valid %s y_veg %d" % (cloud_frac[i], y_cloud[i], veg_valid[i], y_veg[i]))
