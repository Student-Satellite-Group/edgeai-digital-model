# Quantization Accuracy Delta Report (Task 5.6 / Issue #27)
**Generated:** 2026-09-26 10:15:03 UTC
**Target Architecture:** ARM Cortex-M / ESP32-S3 / Edge TPU
**Calibration Method:** Full Integer Int8 Quantization with Representative Multi-Modal Dataset

## Summary Results

| Task | Test Set Size | Float32 Size | Int8 Size | Size Reduction | Float32 Acc | Int8 Acc | Accuracy Delta | Status |
|:-----|:-------------:|:------------:|:---------:|:--------------:|:-----------:|:--------:|:--------------:|:------:|
| **Cloud** | 462 | 82.4 KB | 43.7 KB | -47.0% | 99.57% | 99.35% | **+0.22 pp** | **ACCEPTABLE (Passes Flight Budget)** |
| **Vegetation** | 308 | 82.4 KB | 43.7 KB | -47.0% | 89.94% | 91.23% | **-1.30 pp** | **ACCEPTABLE (Passes Flight Budget)** |
| **Fire** | 549 | 82.5 KB | 43.7 KB | -47.0% | 96.90% | 96.72% | **+0.18 pp** | **ACCEPTABLE (Passes Flight Budget)** |

## Conclusion & Flight Readiness
- **Zero QAT Escalation Triggered:** All multi-modal tasks demonstrated an accuracy drop strictly below the $2.0\text{ pp}$ threshold.
- **Embedded Footprint:** Every quantized model fits within $\approx 43\text{ KB}$, enabling on-chip SRAM residency on ESP32-S3 and Raspberry Pi Zero 2W.
- **Quantization Acceptance:** **PASSED**.