# Aggregated Multi-Modal Training & Cross-Validation Log
**Generated:** 2026-09-26 10:13:56 UTC
**Dataset:** Aggregated Multi-Modal Dataset (3,612 samples)

## Out-of-Fold (OOF) Benchmark Summary

| Task | Dataset | Samples (Pos / Total) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:-----|:--------|:---------------------:|:--------:|:---------:|:------:|:--------:|:-------:|
| **Cloud** | Sentinel-2 Delhi | 735 / 1764 | **94.90%** | 0.9046 | 0.9810 | **0.9413** | **0.9929** |
| **Vegetation** | Sentinel-2 Delhi | 462 / 933 | **92.18%** | 0.8898 | 0.9610 | **0.9240** | **0.9801** |
| **Fire** | Landsat-9 Siberia | 134 / 1848 | **93.94%** | 0.5743 | 0.6343 | **0.6028** | **0.9130** |

## Per-Fold Performance Breakdown

### Task: Cloud
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 0 | 95.24% | 0.8686 | 0.9675 | 0.9154 | 0.9927 |
| Fold 1 | 93.10% | 0.8854 | 1.0000 | 0.9392 | 0.9910 |
| Fold 2 | 94.37% | 0.9120 | 0.9828 | 0.9461 | 0.9941 |
| Fold 3 | 96.90% | 0.9554 | 0.9615 | 0.9585 | 0.9961 |

### Task: Vegetation
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 0 | 89.94% | 0.8765 | 1.0000 | 0.9342 | 0.9821 |
| Fold 1 | 94.65% | 0.9273 | 0.8947 | 0.9107 | 0.9891 |
| Fold 2 | 95.12% | 0.8841 | 0.9683 | 0.9242 | 0.9943 |
| Fold 3 | 90.56% | 0.9032 | 0.9180 | 0.9106 | 0.9794 |

### Task: Fire
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 0 | 96.36% | 0.7826 | 0.5455 | 0.6429 | 0.9123 |
| Fold 1 | 94.58% | 0.8214 | 0.6765 | 0.7419 | 0.9201 |
| Fold 2 | 86.90% | 0.4643 | 0.7647 | 0.5778 | 0.8940 |
| Fold 3 | 94.68% | 0.4390 | 0.5455 | 0.4865 | 0.9216 |
