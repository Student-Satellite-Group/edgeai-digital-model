# data/ — raw imagery & manifests

- `manifest_raw.csv` — the authoritative record of downloaded scenes
  (scene_id, source, platform, acquisition date, cloud cover, band paths).
  Paths use forward slashes regardless of OS.
- `raw/sentinel2/`, `raw/landsat/` — **tiles are NOT committed to git**
  (git-ignored: each Landsat B10 tile is ~55 MB and a full Sentinel-2 L2A
  scene is much larger, so the binary data stays local). The directories are
  recreated automatically by `download_tiles.py`; re-fetch with:

      python download_tiles.py --limit 4

  (pilot, public mirrors, no credentials) or `--full` once `.env` credentials
  are configured (see `data_access.md`).