# Quantization Accuracy Delta — Issue #27 / Task 5.6

**Generated:** 2026-09-25 16:27 UTC  
**Data source:** real manifest  
**Test samples:** 18

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:--------:|:------:|:--------:|
| Float32 TFLite | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Int8 TFLite    | 1.0000  | 1.0000  | 1.0000  | 1.0000  |
| **Delta**      | **+0.00 pp** | — | — | — |

## Categorisation

**Acceptable (< 2%)**

| Threshold | Action |
|:----------|:-------|
| < 2 pp    | Acceptable — deploy int8 as-is |
| 2–5 pp    | Marginal — review, optional QAT |
| > 5 pp    | Unacceptable — trigger QAT (Issue #28) |

**QAT required:** No

## Float32 TFLite Confusion Matrix

| | Pred No-Cloud | Pred Cloud |
|:--|:---:|:---:|
| True No-Cloud | 11 | 0 |
| True Cloud    | 0 | 7 |

## Int8 TFLite Confusion Matrix

| | Pred No-Cloud | Pred Cloud |
|:--|:---:|:---:|
| True No-Cloud | 11 | 0 |
| True Cloud    | 0 | 7 |

## Acceptance Criteria

- Both models evaluated on identical test set: ✅
- Delta computed and categorised: ✅
- Hardware-independent measurement (dev-machine TFLite interpreter): ✅

## Proxy Data Note

All results are from synthetic proxy tiles (see `proxy_data_caveats.md`).
Re-run after `download_tiles.py` provides real Sentinel-2 imagery.
