# RMSE Results — Registration Accuracy Harness

**Task:** 4.4 - Build RMSE Measurement Harness  
**Samples evaluated:** 20  
**Successful RMSE measurements:** 0  
**Failed:** 20  

---

## Summary

| Scene Type | Mean RMSE (px) |
|:-----------|---------------:|
| Overall    | nan         |
| Clear      | nan         |
| Cloudy     | nan         |

---

## Per-Sample Results

| sample_id   | scene  | n_tie_pts | RMSE (px) | inlier_frac | status |
|:------------|:------:|----------:|----------:|------------:|:-------|
| sample_0001 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0003 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0006 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0016 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0023 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0027 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0044 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0048 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0054 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0055 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0057 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0059 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0061 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0063 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0072 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0076 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0087 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0096 | clear | 0 | N/A | 0.00% | insufficient_features |
| sample_0097 | cloudy | 0 | N/A | 0.00% | insufficient_features |
| sample_0099 | cloudy | 0 | N/A | 0.00% | insufficient_features |

---

## Notes

- ORB params used: nfeatures=500, scaleFactor=1.2, nlevels=8, ransac_thresh=5 px.
  Run tune_registration.py and update constants if better params are found.
- Thermal images upsampled from 24x32 to 128x128 for homography estimation.
- Tie points generated via ORB feature matching (no manual annotation needed).
- All results are provisional proxy-data measurements. See proxy_data_caveats.md.
- Re-run after downloading real Sentinel-2 / Landsat-8 tiles.
