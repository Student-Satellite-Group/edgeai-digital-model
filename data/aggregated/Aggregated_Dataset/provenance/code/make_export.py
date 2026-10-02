"""Package the datasets used in the three experiments into one documented export folder."""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import transform as warp_transform

S = Path(__file__).parent
OUT = Path(r"C:\Users\Pranjal\Desktop\SSG\Aggregated_Dataset")
B02 = Path(r"C:\Users\Pranjal\Desktop\SSG\edgeai-digital-model\data\raw\sentinel2\S2C_43RGM_20260920_0_L2A\B02.tif")
FIRE_SID = "LC09_L2SP_158014_20260817_02_T1"
FIRE_DIR = S / "landsat" / FIRE_SID
P = 128
S2_OPT, S2_THM = "S2C_43RGM_20260920_0_L2A", "LC09_L2SP_147040_20260921_02_T1"

for d in ("sentinel2_delhi", "landsat9_siberia_fire", "provenance/code", "provenance/logs_and_results", "provenance/reports"):
    (OUT / d).mkdir(parents=True, exist_ok=True)


def geo(T, crs, rc):
    x0, y0, x1, y1 = [], [], [], []
    for r, c in rc:
        a = T * (float(c * P), float(r * P)); b = T * (float((c + 1) * P), float((r + 1) * P))
        x0.append(a[0]); y0.append(a[1]); x1.append(b[0]); y1.append(b[1])
    x0, y0, x1, y1 = map(np.array, (x0, y0, x1, y1))
    lon, lat = warp_transform(crs, "EPSG:4326", list((x0 + x1) / 2), list((y0 + y1) / 2))
    return dict(crs=str(crs), x_min=np.minimum(x0, x1), y_min=np.minimum(y0, y1), x_max=np.maximum(x0, x1),
                y_max=np.maximum(y0, y1), center_lon=np.array(lon), center_lat=np.array(lat))


def tstats(thm):
    f = thm.reshape(len(thm), -1)
    with np.errstate(all="ignore"):
        mn, md, mx, me, sd = f.min(1), np.median(f, 1), f.max(1), f.mean(1), f.std(1)
    return dict(thermal_min_K=mn, thermal_median_K=md, thermal_max_K=mx, thermal_mean_K=me, thermal_std_K=sd, thermal_contrast_K=mx - md)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.6g}" if abs(v) < 1e5 else f"{v:.2f}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    return str(v)


def write_csv(path, cols, n, get):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([fmt(get(c, i)) for c in cols])


# ------------------------------------------------------------------ Sentinel-2 / Delhi
Pz = np.load(S / "patches.npz")
rgb, thm, rc = Pz["rgb"], Pz["thm"], Pz["rc"]
N = len(rgb)
with rasterio.open(B02) as ds:
    T20, crs = ds.transform * Affine.scale(2.0, 2.0), ds.crs
scl = np.load(S / "scene_cache.npz")["scl"]
scl_frac = np.zeros((N, 12), np.float32)
for i, (r, c) in enumerate(rc):
    scl_frac[i] = np.bincount(scl[r * P:(r + 1) * P, c * P:(c + 1) * P].ravel(), minlength=12)[:12] / (P * P)

cloud_frac = scl_frac[:, [8, 9, 10]].sum(1)
assert np.allclose(cloud_frac, Pz["cloud_frac"], atol=1e-5), "cloud fraction mismatch vs experiment data"
assert ((cloud_frac >= 0.30).astype(int) == Pz["y_cloud"]).all(), "cloud labels mismatch"
veg_cover = scl_frac[:, 4] + scl_frac[:, 5]
veg_valid = veg_cover >= 0.5
assert (veg_valid == Pz["veg_valid"]).all(), "veg validity mismatch"
veg_frac = np.where(veg_valid, scl_frac[:, 4] / np.maximum(veg_cover, 1e-9), np.nan)
y_veg = np.where(veg_valid, (veg_frac >= 0.5).astype(np.int8), -1).astype(np.int8)
assert (y_veg[veg_valid] == Pz["y_veg"][veg_valid]).all(), "veg labels mismatch"
y_cloud = Pz["y_cloud"].astype(np.int8)
fold_s2 = ((rc[:, 0] * 4) // 42).astype(np.int8)
has_t = Pz["has_thermal"]
ids_s2 = np.array([f"s2delhi_r{r:02d}_c{c:02d}" for r, c in rc])
g2, t2 = geo(T20, crs, rc), tstats(thm)

np.savez_compressed(OUT / "sentinel2_delhi" / "patches.npz", rgb=rgb, thm=thm, y_cloud=y_cloud, y_veg=y_veg, veg_valid=veg_valid,
                    has_thermal=has_t, cloud_frac=cloud_frac.astype(np.float32), veg_frac=veg_frac.astype(np.float32),
                    scl_frac=scl_frac, fold=fold_s2, rc=rc.astype(np.int16), patch_id=ids_s2)
cols2 = (["patch_id", "row", "col", "crs", "x_min", "y_min", "x_max", "y_max", "center_lon", "center_lat", "fold", "has_thermal",
          "y_cloud", "cloud_frac", "veg_valid", "y_veg", "veg_frac"] + [f"scl_frac_{k}" for k in range(12)]
         + list(t2) + ["optical_scene", "thermal_scene"])


def get2(c, i):
    if c == "patch_id": return ids_s2[i]
    if c in ("row", "col"): return rc[i][0 if c == "row" else 1]
    if c in g2: return g2[c] if c == "crs" else g2[c][i]
    if c == "fold": return fold_s2[i]
    if c == "has_thermal": return has_t[i]
    if c == "y_cloud": return y_cloud[i]
    if c == "cloud_frac": return cloud_frac[i]
    if c == "veg_valid": return veg_valid[i]
    if c == "y_veg": return "" if y_veg[i] < 0 else y_veg[i]
    if c == "veg_frac": return veg_frac[i]
    if c.startswith("scl_frac_"): return scl_frac[i, int(c.split("_")[-1])]
    if c in t2: return t2[c][i]
    return {"optical_scene": S2_OPT, "thermal_scene": S2_THM if has_t[i] else ""}[c]


write_csv(OUT / "sentinel2_delhi" / "manifest.csv", cols2, N, get2)
print(f"[S2/Delhi] {N} patches | cloud pos {int(y_cloud.sum())} | veg valid {int(veg_valid.sum())} pos {int((y_veg == 1).sum())} | thermal {int(has_t.sum())}")

# ------------------------------------------------------------------ Landsat 9 / Siberia
F = np.load(FIRE_DIR / "fire_patches.npz")
O = np.load(S / "fire_oof.npz")
frgb, fthm, fy, fn, frc = F["rgb"], F["thm"], F["y"].astype(np.int8), F["nfire"], F["rc"]
assert (O["rc"] == frc).all() and (O["y"] == F["y"]).all(), "OOF file does not match patch file"
NF = len(frgb)
ffold = O["fold"].astype(np.int8)
with rasterio.open(FIRE_DIR / "swir22.tif") as ds:
    T30, fcrs = ds.transform, ds.crs
gf, tf_ = geo(T30, fcrs, frc), tstats(fthm)
ids_f = np.array([f"l9siberia_r{r:02d}_c{c:02d}" for r, c in frc])
oof = {v: np.mean([O[f"{v}__seed{s}"] for s in range(3)], axis=0).astype(np.float32) for v in ("rgb", "thermal", "both")}
extra = {f"oof_{v}_seed{s}": O[f"{v}__seed{s}"] for v in ("rgb", "thermal", "both") for s in range(3)}
np.savez_compressed(OUT / "landsat9_siberia_fire" / "patches.npz", rgb=frgb, thm=fthm, y_fire=fy, n_fire_px=fn.astype(np.int16),
                    fold=ffold, rc=frc.astype(np.int16), patch_id=ids_f, oof_rgb_mean=oof["rgb"], oof_thermal_mean=oof["thermal"],
                    oof_both_mean=oof["both"], **extra)
colsf = (["patch_id", "row", "col", "crs", "x_min", "y_min", "x_max", "y_max", "center_lon", "center_lat", "fold", "n_fire_px", "y_fire"]
         + list(tf_) + ["oof_rgb", "oof_thermal", "oof_both", "scene"])


def getf(c, i):
    if c == "patch_id": return ids_f[i]
    if c in ("row", "col"): return frc[i][0 if c == "row" else 1]
    if c in gf: return gf[c] if c == "crs" else gf[c][i]
    if c == "fold": return ffold[i]
    if c == "n_fire_px": return fn[i]
    if c == "y_fire": return fy[i]
    if c in tf_: return tf_[c][i]
    if c.startswith("oof_"): return oof[c[4:]][i]
    return FIRE_SID


write_csv(OUT / "landsat9_siberia_fire" / "manifest.csv", colsf, NF, getf)
print(f"[Siberia] {NF} patches | fire {int(fy.sum())} | fold sizes {[int((ffold == k).sum()) for k in range(4)]} | fold fire {[int(fy[ffold == k].sum()) for k in range(4)]}")

# ------------------------------------------------------------------ combined manifest
allcols = ["patch_id", "dataset", "scene_optical", "scene_thermal", "row", "col", "crs", "x_min", "y_min", "x_max", "y_max", "center_lon",
           "center_lat", "fold_scheme", "fold", "has_thermal", "y_cloud", "y_veg", "y_fire", "thermal_median_K", "thermal_max_K", "thermal_contrast_K"]
with open(OUT / "manifest_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(allcols)
    for i in range(N):
        w.writerow([ids_s2[i], "sentinel2_delhi", S2_OPT, S2_THM if has_t[i] else "", rc[i][0], rc[i][1], g2["crs"], *(fmt(g2[k][i]) for k in ("x_min", "y_min", "x_max", "y_max", "center_lon", "center_lat")),
                    "row_band_4fold", fold_s2[i], int(has_t[i]), y_cloud[i], "" if y_veg[i] < 0 else y_veg[i], "",
                    *(fmt(t2[k][i]) for k in ("thermal_median_K", "thermal_max_K", "thermal_contrast_K"))])
    for i in range(NF):
        w.writerow([ids_f[i], "landsat9_siberia_fire", FIRE_SID, FIRE_SID, frc[i][0], frc[i][1], gf["crs"], *(fmt(gf[k][i]) for k in ("x_min", "y_min", "x_max", "y_max", "center_lon", "center_lat")),
                    "spatial_block_4fold", ffold[i], 1, "", "", fy[i], *(fmt(tf_[k][i]) for k in ("thermal_median_K", "thermal_max_K", "thermal_contrast_K"))])
print("manifest_all.csv rows:", N + NF)

# ------------------------------------------------------------------ previews
def preview(path, r_, t_, idx, titles, head):
    fig, ax = plt.subplots(2, len(idx), figsize=(1.9 * len(idx), 4.4))
    for j, i in enumerate(idx):
        ax[0, j].imshow(r_[i]); ax[0, j].set_title(titles[j], fontsize=7)
        ax[1, j].imshow(t_[i], cmap="inferno", interpolation="nearest")
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("RGB 128x128", fontsize=8); ax[1, 0].set_ylabel("thermal 24x32", fontsize=8)
    fig.suptitle(head, fontsize=9); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


rng = np.random.default_rng(0)
pc = rng.choice(np.where((y_cloud == 1) & has_t)[0], 4, replace=False); pn = rng.choice(np.where((y_cloud == 0) & has_t)[0], 4, replace=False)
sel = np.concatenate([pc, pn])
preview(OUT / "sentinel2_delhi" / "preview.png", rgb, thm, sel, [f"{'CLOUD' if y_cloud[i] else 'no-cloud'} {cloud_frac[i]:.0%}" for i in sel],
        "Sentinel-2 / Delhi patches (left 4: cloud, right 4: no-cloud; thermal from Landsat 9, ~24 h later)")
pos = np.where(fy == 1)[0]; pos = pos[np.argsort(fn[pos])]
sp = pos[np.linspace(0, len(pos) - 1, 4).astype(int)]; sn = rng.choice(np.where(fy == 0)[0], 4, replace=False)
sel = np.concatenate([sp, sn])
preview(OUT / "landsat9_siberia_fire" / "preview.png", frgb, fthm, sel, [f"FIRE {fn[i]}px" if fy[i] else "no fire" for i in sel],
        "Landsat 9 / Siberia patches (left 4: fire, increasing size; right 4: no fire; RGB and thermal from the same scene)")

# ------------------------------------------------------------------ provenance
code = ["fetch_scene.py", "build_patches.py", "experiments.py", "single_image.py", "find_fires.py", "fetch_landsat.py", "screen_scene.py",
        "screen_many.py", "build_fire_patches.py", "train_fire.py", "analyze_fire.py", "make_export.py"]
logs = ["experiments_log.txt", "find_fires.log", "screen_many.log", "train_fire.log", "analyze_fire.log", "fire_results.json", "fire_candidates.json"]
for n in code:
    if (S / n).exists(): shutil.copy2(S / n, OUT / "provenance/code" / n)
for n in logs:
    if (S / n).exists(): shutil.copy2(S / n, OUT / "provenance/logs_and_results" / n)
for src, n in ((Path(r"C:\Users\Pranjal\Desktop\SSG"), "Pipeline_Test_Run_Report.md"), (Path(r"C:\Users\Pranjal\Desktop\SSG"), "Thermal_Anomaly_Test_Run_Report.md")):
    if (src / n).exists(): shutil.copy2(src / n, OUT / "provenance/reports" / n)
repo = Path(r"C:\Users\Pranjal\Desktop\SSG\edgeai-digital-model")
for n in ("model_architecture.py", "task_definition.md"):
    shutil.copy2(repo / n, OUT / "provenance/code" / n)
print("export written to", OUT)
