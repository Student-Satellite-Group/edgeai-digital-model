"""Download Landsat C2 L2 band files for one scene from Planetary Computer (requests, resumable).

usage: python fetch_landsat.py <scene_id> <asset1,asset2,...>
files land in scratchpad/landsat/<scene_id>/<asset>.tif
"""
import sys
import time
from pathlib import Path

import requests

PC = "https://planetarycomputer.microsoft.com/api"
ROOT = Path(__file__).parent / "landsat"


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    part = dest.with_suffix(".part")
    size = int(requests.head(url, timeout=60).headers["Content-Length"])
    pos = part.stat().st_size if part.exists() else 0
    step = 8 << 20
    while pos < size:
        for attempt in range(6):
            try:
                r = requests.get(url, headers={"Range": f"bytes={pos}-{min(pos + step, size) - 1}"}, timeout=45)
                if r.status_code == 206:
                    with open(part, "ab") as f:
                        f.write(r.content)
                    pos += len(r.content)
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            break
    if pos != size:
        raise IOError(f"incomplete download {dest.name}: {pos}/{size}")
    part.rename(dest)


if __name__ == "__main__":
    sid, assets = sys.argv[1], sys.argv[2].split(",")
    item = requests.get(f"{PC}/stac/v1/collections/landsat-c2-l2/items/{sid}", timeout=60).json()
    for a in assets:
        t = time.time()
        tok = requests.get(f"{PC}/sas/v1/token/landsat-c2-l2", timeout=30).json()["token"]
        download(item["assets"][a]["href"] + "?" + tok, ROOT / sid / f"{a}.tif")
        print(f"{sid} {a}: {(ROOT / sid / f'{a}.tif').stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f}s", flush=True)
    print("DONE", sid, flush=True)
