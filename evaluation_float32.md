# Float32 Model Evaluation Report — Issue #25 / Task 5.4

**Generated:** 2026-09-25 16:26 UTC  
**Model:** `models\model_float32.h5`  
**Parameters:** 19,090  
**Data source:** real manifest  
**Test samples:** 18

## Overall Metrics

| Metric    | Value   |
|:----------|--------:|
| Accuracy  | 1.0000 (100.0%) |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

## Confusion Matrix

| | **Predicted: No-Cloud** | **Predicted: Cloud** |
|:--|:---:|:---:|
| **True: No-Cloud** | 11 | 0 |
| **True: Cloud**    | 0 | 7 |

## Per-Class Report

| Class     | Precision | Recall | F1-Score | Support |
|:----------|----------:|-------:|---------:|--------:|
| No-Cloud (0) | 1.0000 | 1.0000 | 1.0000 | 11 |
| Cloud    (1) | 1.0000  | 1.0000  | 1.0000  | 7 |

## Acceptance Criteria

- All metrics computed and documented: ✅
- Confusion matrix generated: ✅
- Accuracy > 90%: ✅ (100.0%)

## Proxy Data Note

Evaluation is on synthetic proxy tiles (see `proxy_data_caveats.md`).
Replace with real Sentinel-2 tiles for production evaluation.
