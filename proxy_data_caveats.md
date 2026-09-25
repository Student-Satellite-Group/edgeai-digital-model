# Proxy Data Caveats

**Task:** 4.6 - Document Proxy-Data Caveats  
**Status:** All Phase 4 results are PROVISIONAL and based on proxy satellite data.

---

## Executive Summary

Phase 4 registration and fusion results use **proxy data** — publicly available
Sentinel-2 L2A (optical) and Landsat-8 (thermal) scenes downloaded from the ESA
and NASA archives. This data is used as a stand-in for the mission's own sensor
payload, which has not yet been built. Every metric, parameter, and finding in
Phase 4 must be treated as provisional until real flight-hardware data is
available.

---

## Caveat 1: Temporal Parallax and Cloud Drift

**Problem:**  
Sentinel-2 has a ~10:30 AM local overpass time, while Landsat-8 passes the
same region roughly 22 minutes later (~10:52 AM local time). Clouds move at
speeds of 5–50 m/s at low altitudes, which translates to **660 m – 66 km of
lateral drift** over a 22-minute gap at typical synoptic scales. This means:

- A cloud feature at pixel (x, y) in the optical image may have drifted
  tens of pixels in the thermal image by the time Landsat captures it.
- Sub-kilometer thermal signatures (isolated cumulus cells, building heat
  plumes) can be **completely de-correlated** between the two sensors.
- The ORB/RANSAC registration pipeline flags these cases as low-inlier scenes,
  but cannot fully compensate for physical cloud drift.

**Implication for Week 1 model:**  
Because thermal and optical are not temporally co-located, the RGB optical
branch carries the primary discriminative signal for cloud/no-cloud
classification. The thermal branch provides supplementary context but its
contribution is degraded by temporal parallax.

**Thermal pipeline freeze decision (2026-09-25):**  
Per team evaluation, the thermal pipeline does not provide a clear advantage
over the optical-only baseline under proxy data conditions. The thermal branch
is **frozen** (held at its current implementation) pending real flight-hardware
data, where zero temporal parallax will be guaranteed.

---

## Caveat 2: Zero-Time-Delta Resolution on Orbit

**What changes when real hardware arrives:**  
The mission payload co-locates both sensors — the optical camera and the thermal
imager — on the same CubeSat bus with simultaneous (or near-simultaneous)
electronic exposure triggering. This eliminates the 22-minute parallax
entirely: **Delta-t = 0** between optical and thermal captures.

With zero temporal parallax:
- Cloud drift between optical and thermal becomes zero.
- ORB/RANSAC registration handles only geometric misalignment (lens
  parallax, sensor offsets), which is sub-pixel for a rigidly co-mounted
  payload.
- The thermal branch contribution to the model is expected to improve
  significantly.

All Phase 4 registration parameters and RMSE numbers will need to be
re-calibrated against real dual-sensor data before the Week 2 model update.

---

## Caveat 3: Radiometric and GSD Differences

| Source         | Band             | Native GSD | Spectral Range      |
|:---------------|:-----------------|:----------:|:--------------------|
| Sentinel-2 L2A | B02 (Blue)       | 10 m       | 458–523 nm          |
| Sentinel-2 L2A | B03 (Green)      | 10 m       | 543–578 nm          |
| Sentinel-2 L2A | B04 (Red)        | 10 m       | 650–680 nm          |
| Landsat-8 OLI  | Band 10 (Thermal)| 30 m (100m)| 10.6–11.19 µm (TIR) |
| Mission design | Optical (PROVISIONAL) | **7.19 m** | RGB broadband |
| Mission design | Thermal (PROVISIONAL) | **7.19 m eq.** | 8–14 µm TIR |

- Proxy optical GSD (10 m) is coarser than design target (7.19 m); tiles are
  downsampled to 128x128, effectively blurring fine spatial features.
- Proxy thermal (30 m native / 100 m IFOV) is far coarser than the final
  on-orbit thermal sensor spec. All thermal features appear smeared.
- Radiometric calibration between Sentinel-2 (ESA) and Landsat-8 (USGS)
  differs in atmospheric correction methodology.

---

## Caveat 4: Summary Statement

> **All Phase 4 results (registration tuning, RMSE measurements, fusion
> alpha values, and classification accuracy) are provisional and based on
> proxy satellite data. They are not representative of mission performance
> and must be re-evaluated once real flight-hardware data is available.**

---

## Affected Deliverables

| Deliverable                    | Provisional? | Notes |
|:-------------------------------|:------------:|:------|
| registration_tuning_log.md     | YES          | Re-run tune_registration.py on real data |
| rmse_results.md                | YES          | Re-run rmse_harness.py on real data |
| data/labeled/manifest.csv      | YES          | Synthetic patch dataset |
| labeling_method.md             | YES          | SCL labels from proxy Sentinel-2 scenes |
| qc_report.md                   | YES          | Validated against synthetic tiles |
| fusion.py alpha parameter      | YES          | Re-sweep on real data |

---

*Document written: 2026-09-25. Update when real flight-hardware data arrives.*
