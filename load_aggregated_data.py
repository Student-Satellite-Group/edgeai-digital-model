"""
load_aggregated_data.py — Data loading utilities for the Aggregated Multi-Modal Dataset.

Supports:
  1. Sentinel-2 Delhi Dataset (1,764 patches, Cloud & Vegetation tasks)
  2. Landsat-9 Siberia Fire Dataset (1,848 patches, Wildfire Thermal Anomaly task)
  3. Combined Multi-Scene Dataset (3,612 total patches)
  4. Fold-isolated Thermal Normalization (strict zero data leakage across folds)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "aggregated", "Aggregated_Dataset")
DELHI_DIR = os.path.join(DATA_ROOT, "sentinel2_delhi")
SIBERIA_DIR = os.path.join(DATA_ROOT, "landsat9_siberia_fire")
MANIFEST_ALL_PATH = os.path.join(DATA_ROOT, "manifest_all.csv")


def load_delhi_data() -> Dict[str, np.ndarray]:
    """Load Sentinel-2 Delhi multi-modal patches (1,764 patches).

    Returns:
        Dict with keys: 'rgb', 'thm', 'y_cloud', 'y_veg', 'veg_valid',
                        'has_thermal', 'cloud_frac', 'veg_frac', 'fold', 'patch_id'
    """
    npz_path = os.path.join(DELHI_DIR, "patches.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Delhi patches not found at {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def load_siberia_fire_data() -> Dict[str, np.ndarray]:
    """Load Landsat-9 Siberia Wildfire multi-modal patches (1,848 patches).

    Returns:
        Dict with keys: 'rgb', 'thm', 'y_fire', 'n_fire_px', 'fold', 'patch_id', 'oof_both_mean'
    """
    npz_path = os.path.join(SIBERIA_DIR, "patches.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Siberia fire patches not found at {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def load_manifest_all() -> pd.DataFrame:
    """Load combined manifest across all 3,612 patches."""
    if not os.path.exists(MANIFEST_ALL_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_ALL_PATH}")
    return pd.read_csv(MANIFEST_ALL_PATH)


def prepare_fold_data(
    data: Dict[str, np.ndarray],
    task: str = "cloud",
    test_fold: int = 0,
    val_fold: Optional[int] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare train and test splits for a given spatial fold with fold-isolated thermal normalization.

    Args:
        data: dictionary returned from load_delhi_data() or load_siberia_fire_data()
        task: 'cloud', 'vegetation', or 'fire'
        test_fold: fold index (0, 1, 2, 3) to hold out as test set
        val_fold: optional validation fold index

    Returns:
        (X_train_rgb, X_train_thm, y_train), (X_test_rgb, X_test_thm, y_test)
    """
    folds = data["fold"]
    rgb = data["rgb"]
    thm = data["thm"]

    # Select target labels and filter valid mask
    if task == "cloud":
        y = data["y_cloud"]
        valid_mask = np.ones(len(y), dtype=bool)
    elif task == "vegetation":
        y = data["y_veg"]
        valid_mask = data["veg_valid"] == 1
    elif task == "fire":
        y = data["y_fire"]
        valid_mask = np.ones(len(y), dtype=bool)
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'cloud', 'vegetation', or 'fire'.")

    # Filter by valid mask
    rgb = rgb[valid_mask]
    thm = thm[valid_mask]
    y = y[valid_mask]
    folds = folds[valid_mask]

    train_idx = (folds != test_fold)
    test_idx = (folds == test_fold)

    # Normalize RGB to [0, 1]
    X_train_rgb = rgb[train_idx].astype(np.float32) / 255.0
    X_test_rgb = rgb[test_idx].astype(np.float32) / 255.0

    # Ensure thermal shape is (N, 24, 32, 1)
    train_thm = thm[train_idx].copy()
    test_thm = thm[test_idx].copy()
    if train_thm.ndim == 3:
        train_thm = np.expand_dims(train_thm, -1)
        test_thm = np.expand_dims(test_thm, -1)

    # Compute fold-isolated thermal mean and std from training set only (ignoring NaNs)
    thm_train_valid = train_thm[~np.isnan(train_thm)]
    if len(thm_train_valid) > 0:
        mean_t = float(np.mean(thm_train_valid))
        std_t = float(np.std(thm_train_valid)) + 1e-6
    else:
        mean_t, std_t = 0.0, 1.0

    # Apply z-score normalization and fill NaNs (missing thermal) with 0.0 (the normalized mean)
    X_train_thm = (train_thm - mean_t) / std_t
    X_test_thm = (test_thm - mean_t) / std_t
    X_train_thm = np.nan_to_num(X_train_thm, nan=0.0).astype(np.float32)
    X_test_thm = np.nan_to_num(X_test_thm, nan=0.0).astype(np.float32)

    y_train = y[train_idx].astype(np.int32)
    y_test = y[test_idx].astype(np.int32)

    return (X_train_rgb, X_train_thm, y_train), (X_test_rgb, X_test_thm, y_test)


if __name__ == "__main__":
    print("Testing data loader...")
    delhi = load_delhi_data()
    print(f"Delhi patches: {delhi['rgb'].shape}, Cloud labels: {np.bincount(delhi['y_cloud'])}")
    siberia = load_siberia_fire_data()
    print(f"Siberia patches: {siberia['rgb'].shape}, Fire labels: {np.bincount(siberia['y_fire'])}")
    manifest = load_manifest_all()
    print(f"Total manifest rows: {len(manifest)}")

    (tr_rgb, tr_thm, tr_y), (te_rgb, te_thm, te_y) = prepare_fold_data(delhi, task="cloud", test_fold=0)
    print(f"Fold 0 Cloud Train: RGB {tr_rgb.shape}, THM {tr_thm.shape}, Y {tr_y.shape}")
    print(f"Fold 0 Cloud Test : RGB {te_rgb.shape}, THM {te_thm.shape}, Y {te_y.shape}")
    print("DataLoader test PASSED!")
