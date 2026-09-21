# Data-Access Accounts & Access Documentation (Task 3.1)

Status: **SETUP COMPLETE (credentials pending user signup)** — PROVISIONAL.

## Platform 1 — Copernicus (Sentinel-2 optical)

| Item | Value |
|------|-------|
| Name | Copernicus Open Access Hub (Data Hub) |
| URL | https://scihub.copernicus.eu/ |
| Product | Sentinel-2 L2A (optical, 10 m / 20 m bands) |
| Purpose | Optical proxy imagery for cloud/no-cloud classification |
| Credentials | Referenced from `.env` (`COPERNICUS_USERNAME`, `COPERNICUS_PASSWORD`) — never stored in this doc |

### Verification performed
- [x] Search interface reachable (network probe OK)
- [x] Sentinel-2 product query format confirmed via public STAC endpoint
- [ ] **Account signup + authenticated search — done by maintainer** (requires personal email; cannot be automated here)

## Platform 2 — USGS / Landsat thermal (public mirror used for pilot)

| Item | Value |
|------|-------|
| Name | Google Cloud `gcp-public-data-landsat` (public bucket, no auth) |
| URL | https://console.cloud.google.com/storage/browser/gcp-public-data-landsat |
| Product | Landsat 8 Collection 1 Level-1 (thermal band 10) — used for the credential-free pilot |
| Full-access product | Landsat 8/9 Collection 2 Level 2 (thermal band 10/11) via USGS (requires `.env`) |
| Credentials | Referenced from `.env` — never stored in this doc |

### Verification performed (pilot)
- [x] `download_tiles.py --limit 2` fetched 2 real Landsat B10 tiles (~55 MB each, uint16, EPSG:32643) into `data/raw/landsat/<scene>/B10.TIF`
- [x] Tile validity confirmed with `rasterio.open` (7541 x 7681)
- [x] Landsat-8 Collection 2 Scene-path access (USGS landsatlook STAC) returns auth-walled HTML redirects → **not usable without credentials**; the downloader therefore uses the public GCS mirror for pilot runs and flags FULL mode for authenticated bulk access.

## Access notes

1. Copy `.env.template` → `.env` and fill in credentials (git-ignored).
2. `download_tiles.py` reads these variables to authenticate against both hubs.
3. Public mirrors used by the pilot: Sentinel-2 via earth-search STAC (`earth-search.aws.element84.com`), Landsat via Google Cloud `gcp-public-data-landsat`. Neither requires credentials.
4. `.gitignore` excludes `.env` so credentials are never committed.

## Issues / resolutions
- Copernicus SciHub is migrating to the new Copernicus Data Space Ecosystem; older SciHub auth may need updating. If OAuth2 tokens are required, update `download_tiles.py` accordingly.
- Landsat Collection 2 requires the newer USGS earthdata login flow; documented in the downloader docstring.