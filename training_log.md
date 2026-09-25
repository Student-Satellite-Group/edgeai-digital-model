# Training Log — Cloud CNN (Issue #24 / Task 5.3)

**Generated:** 2026-09-25 16:25 UTC  
**Note:** real manifest

## Dataset Split

| Split | Samples |
|:------|--------:|
| Train | 71 |
| Val   | 15 |
| Test  | 18 |
| Total | 104 |

## Training Curves

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
|     1 | 0.5300 | 0.8310 | 0.3790 | 1.0000 |
|     2 | 0.2553 | 0.9859 | 0.0940 | 1.0000 |
|     3 | 0.1653 | 0.9718 | 0.0393 | 1.0000 |
|     4 | 0.1383 | 0.9718 | 0.0200 | 1.0000 |
|     5 | 0.0938 | 0.9859 | 0.0118 | 1.0000 |
|     6 | 0.0591 | 1.0000 | 0.0075 | 1.0000 |
|     7 | 0.0728 | 0.9859 | 0.0049 | 1.0000 |
|     8 | 0.0356 | 1.0000 | 0.0034 | 1.0000 |
|     9 | 0.0232 | 1.0000 | 0.0024 | 1.0000 |
|    10 | 0.0226 | 1.0000 | 0.0018 | 1.0000 |
|    11 | 0.0311 | 1.0000 | 0.0012 | 1.0000 |
|    12 | 0.0167 | 1.0000 | 0.0009 | 1.0000 |
|    13 | 0.0230 | 0.9859 | 0.0006 | 1.0000 |
|    14 | 0.0101 | 1.0000 | 0.0005 | 1.0000 |
|    15 | 0.0167 | 1.0000 | 0.0004 | 1.0000 |
|    16 | 0.0086 | 1.0000 | 0.0003 | 1.0000 |
|    17 | 0.0155 | 1.0000 | 0.0002 | 1.0000 |
|    18 | 0.0092 | 1.0000 | 0.0001 | 1.0000 |
|    19 | 0.0088 | 1.0000 | 0.0001 | 1.0000 |
|    20 | 0.0067 | 1.0000 | 0.0001 | 1.0000 |

## Validation Metrics

| Metric    | Value   |
|:----------|--------:|
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

### Confusion Matrix

| | Pred No-Cloud | Pred Cloud |
|:--|:---:|:---:|
| **True No-Cloud** | 10 | 0 |
| **True Cloud**    | 0 | 5 |

## Test Metrics (Held-Out)

| Metric    | Value   |
|:----------|--------:|
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

### Confusion Matrix

| | Pred No-Cloud | Pred Cloud |
|:--|:---:|:---:|
| **True No-Cloud** | 11 | 0 |
| **True Cloud**    | 0 | 7 |

## Acceptance Criteria

- Model trains without memory issues: ✅
- Test accuracy > 90%: ✅ (100.0%)
- Metrics logged: ✅

## Proxy Data Note

All results are from synthetic proxy tiles (see `proxy_data_caveats.md`).
Real Sentinel-2 tiles will replace these when `download_tiles.py` runs.
