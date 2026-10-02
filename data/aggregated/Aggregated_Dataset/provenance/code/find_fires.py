"""Rank large fire clusters in MODIS Terra 8-day fire composites (Planetary Computer), Jun-Aug 2026."""
import json
import time

import numpy as np
import rasterio
import requests
from rasterio.warp import transform as warp_transform
from scipy import ndimage

PC = "https://planetarycomputer.microsoft.com/api"
REGIONS = {
    "NA-west/Canada": [-135, 35, -100, 60],
    "Siberia": [90, 50, 140, 65],
    "Mediterranean/S-Europe": [-10, 35, 40, 45],
    "South America": [-70, -20, -45, 0],
}
DT = "2026-06-20/2026-08-31"


def search(bbox):
    out, url, body = [], f"{PC}/stac/v1/search", {"collections": ["modis-14A2-061"], "bbox": bbox, "datetime": DT, "limit": 100}
    r = requests.post(url, json=body, timeout=60).json()
    out += r.get("features", [])
    return [f for f in out if f["id"].startswith("MOD14A2")]


tok = requests.get(f"{PC}/sas/v1/token/modiseuwest/modis-061-cogs", timeout=30).json()["token"]
seen, rows = set(), []
for name, bbox in REGIONS.items():
    items = search(bbox)
    print(f"{name}: {len(items)} Terra composites", flush=True)
    for f in items:
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        url = f["assets"]["FireMask"]["href"] + "?" + tok
        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    content = resp.content
                    break
            except requests.RequestException:
                time.sleep(2)
        else:
            continue
        with rasterio.MemoryFile(content) as mf, mf.open() as ds:
            a = ds.read(1)
            tf, crs = ds.transform, ds.crs
        fire = a >= 8
        if fire.sum() == 0:
            continue
        lab, n = ndimage.label(fire, structure=np.ones((3, 3)))
        sizes = ndimage.sum(fire, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        r, c = ndimage.center_of_mass(lab == k)
        x, y = tf * (c + 0.5, r + 0.5)
        lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
        rows.append(dict(id=f["id"], region=name, start=f["properties"]["start_datetime"][:10], fire_px=int(fire.sum()),
                         clusters=int(n), biggest=int(sizes.max()), lon=round(lon[0], 3), lat=round(lat[0], 3)))
rows.sort(key=lambda d: -d["biggest"])
json.dump(rows, open("fire_candidates.json", "w"), indent=1)
print("\nTOP CANDIDATES (largest connected 1-km fire cluster in an 8-day composite):")
for d in rows[:20]:
    print(f"  {d['region']:24s} {d['start']} tile {d['id'].split('.')[2]} | fire px {d['fire_px']:5d} | biggest cluster {d['biggest']:4d} px | centre lon {d['lon']:8.3f} lat {d['lat']:7.3f}")
print("DONE")
