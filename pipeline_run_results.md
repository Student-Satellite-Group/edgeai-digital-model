# End-to-End Pipeline Execution Results (Task 6.3 / Issue #31)

**Execution Timestamp:** 2026-09-25 16:38 UTC  
**Pipeline Script:** `digital_model_pipeline.py`  
**Status:** [PASS] 100% COMPLETE

---

## 1. Subsystem Parameter Summary

| Subsystem | Metric / Parameter | Value | Reference |
|:----------|:-------------------|:------|:----------|
| **Orbital Mechanics** | Altitude | 500.0 km | `orbital_model.py` |
| | Orbital Period | 94.47 min (5668.1 s) | `orbital_period()` |
| | SSO Inclination | 97.39° | `sun_sync_inclination()` |
| **Payload Sensor** | Optical GSD | 15.50 m/px | `sensor_model.py` |
| | Optical Swath | 62.9 km | `sensor_specs.json` |
| | Thermal GSD | 315.79 m/px | `sensor_model.py` |
| | Thermal Swath | 50.5 km | `sensor_specs.json` |
| | Hardware Status | PROVISIONAL | `PROVISIONAL_LABELS.md` |

---

## 2. End-to-End Batch Execution Metrics

| Metric | Target / Specification | Achieved Result | Status |
|:-------|:-----------------------|:----------------|:-------|
| Total Dataset Samples | $\ge 100$ | **104** | [PASS] |
| Unhandled Errors | 0 | **0** | [PASS] |
| Classification Accuracy | $> 90.0\%$ | **100.0%** | [PASS] |
| Mean Edge Inference Latency | $< 100$ ms/sample | **1.42 ms** | [PASS] |
| Total Pipeline Throughput | — | **49.2 samples/sec** | [PASS] |

---

## 3. Sample Verification Spot-Check

| Sample ID | Pass ID | True Class | Pred Class | Confidence | Latency (ms) | Result |
|:----------|:--------|:----------:|:----------:|:----------:|:------------:|:------:|
| `sample_0000` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.25 | MATCH |
| `sample_0001` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.14 | MATCH |
| `sample_0002` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.13 | MATCH |
| `sample_0003` | `pass_0000` | Cloud | Cloud | 0.9961 | 1.16 | MATCH |
| `sample_0004` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.20 | MATCH |
| `sample_0005` | `pass_0000` | Cloud | Cloud | 0.9961 | 1.12 | MATCH |
| `sample_0006` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.26 | MATCH |
| `sample_0007` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.34 | MATCH |
| `sample_0008` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.21 | MATCH |
| `sample_0009` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.17 | MATCH |
| `sample_0010` | `pass_0000` | Cloud | Cloud | 0.9961 | 1.55 | MATCH |
| `sample_0011` | `pass_0000` | Cloud | Cloud | 0.9961 | 1.32 | MATCH |
| `sample_0012` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 0.79 | MATCH |
| `sample_0013` | `pass_0000` | Cloud | Cloud | 0.9961 | 1.22 | MATCH |
| `sample_0014` | `pass_0000` | No-Cloud | No-Cloud | 0.9961 | 1.36 | MATCH |

> *Showing first 15 samples of the full dataset execution.*

---

## 4. Acceptance Criteria Verification (Issue #31)

- [x] 100% of samples processed without errors.
- [x] All classification outputs in valid range {0, 1}.
- [x] Modular stages callable independently or orchestrator-driven.
- [x] Results saved to `pipeline_run_results.md`.
