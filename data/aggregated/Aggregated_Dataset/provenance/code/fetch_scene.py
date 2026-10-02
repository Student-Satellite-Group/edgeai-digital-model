"""Build a co-temporal Sentinel-2 (2026-09-20) + Landsat 9 thermal (2026-09-21)
scene over New Delhi on a common 20 m grid, cached as scene_cache.npz.

  B02/B03/B04/SCL : local files (downloaded by the user)
  ST_B10          : Planetary Computer Landsat 9 surface temperature, warped to the S2 grid
"""

from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform

HERE = Path(__file__).parent
LOCAL = Path(r"C:\Users\Pranjal\Desktop\SSG\edgeai-digital-model\data\raw\sentinel2\S2C_43RGM_20260920_0_L2A")
OUT = HERE / "scene_cache.npz"
PC = "https://planetarycomputer.microsoft.com/api"
LS_ID = "LC09_L2SP_147040_20260921_02_T1"
FACTOR = 2  # 10 m -> 20 m
NEW_DELHI = (77.2, 28.6)  # lon, lat

one = lambda pat: next(LOCAL.glob(pat))
FILES = {"blue": one("B02*.tif"), "green": one("B03*.tif"),
         "red": one("*B04*.tif"), "scl": one("*SCL*.tif")}
print({k: v.name for k, v in FILES.items()})

ls = requests.get(f"{PC}/stac/v1/collections/landsat-c2-l2/items/{LS_ID}", timeout=60).json()
ls_tok = requests.get(f"{PC}/sas/v1/token/landsat-c2-l2", timeout=30).json()["token"]
ls_url = ls["assets"]["lwir11"]["href"] + "?" + ls_tok

env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60",
           GDAL_HTTP_CONNECTTIMEOUT="20", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="2")
with rasterio.Env(**env):
    with rasterio.open(FILES["blue"]) as ds:
        H, W = ds.height // FACTOR, ds.width // FACTOR
        transform = ds.transform * Affine.scale(ds.width / W, ds.height / H)
        crs, full_tf, bounds = ds.crs, ds.transform, ds.bounds
    print("S2 grid:", H, "x", W, "|", crs)

    def read(key, resampling):
        with rasterio.open(FILES[key]) as ds:
            assert ds.crs == crs and tuple(ds.bounds) == tuple(bounds), f"{key}: grid differs from B02"
            return ds.read(1, out_shape=(H, W), resampling=resampling)

    blue, green, red = (read(k, Resampling.average) for k in ("blue", "green", "red"))
    scl = read("scl", Resampling.nearest)
    print("S2 bands read; all share B02's tile bounds/CRS")

    # N05.12 baseline: BOA reflectance = DN * 0.0001 - 0.1
    nodata = (red == 0) | (green == 0) | (blue == 0)
    red, green, blue = (a.astype(np.uint16) for a in (red, green, blue))

    lb = ls["assets"]["lwir11"]["raster:bands"][0]
    with rasterio.open(ls_url) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=W, height=H,
                       resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt:
            st_dn = vrt.read(1)
    st = np.where(st_dn > 0, st_dn * lb["scale"] + lb["offset"], np.nan).astype(np.float32)
    print("Landsat thermal read + warped to the S2 grid")

x, y = warp_transform("EPSG:4326", crs, [NEW_DELHI[0]], [NEW_DELHI[1]])
nd_row, nd_col = rasterio.transform.rowcol(transform, x[0], y[0])
np.savez_compressed(OUT, red=red, green=green, blue=blue, scl=scl, st=st,
                    nodata=nodata, nd_rc=np.array([nd_row, nd_col]))
print("valid S2 frac: %.3f" % (1 - nodata.mean()))
print("thermal valid frac on S2 grid: %.3f" % np.isfinite(st).mean())
print("ST range K: %.1f - %.1f | mean %.1f" % (np.nanmin(st), np.nanmax(st), np.nanmean(st)))
print("SCL fractions -> cloud(8,9,10): %.3f | veg(4): %.3f | non-veg(5): %.3f | nodata(0): %.3f"
      % (np.isin(scl, [8, 9, 10]).mean(), (scl == 4).mean(), (scl == 5).mean(), (scl == 0).mean()))
print("median DN R,G,B:", np.median(red), np.median(green), np.median(blue))
print("New Delhi pixel (row, col):", nd_row, nd_col)
print("saved", OUT)
