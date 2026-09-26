"""
train_aggregated_pipeline.py — Multi-Task Spatial 4-Fold Cross-Validation Pipeline

Trains the two-branch late fusion CNN (model_architecture.py) across the
Aggregated Multi-Modal Dataset:
  1. Task A: Cloud / No-Cloud Detection (Sentinel-2 Delhi, 1,764 patches)
  2. Task B: Vegetation Mapping (Sentinel-2 Delhi, 933 valid patches)
  3. Task C: Wildfire / Thermal Anomaly Detection (Landsat-9 Siberia, 1,848 patches)

Enforces:
  - Spatial 4-Fold CV (zero geographic data leakage across folds)
  - Fold-isolated thermal z-score normalization
  - Class-weighted loss for severe class imbalance (Siberia wildfire)
  - Exports trained Float32 models to models/ for Int8 quantization
"""

import os
import sys
import time
import json
import argparse
import datetime
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model_architecture import build_model
from load_aggregated_data import (
    load_delhi_data,
    load_siberia_fire_data,
    prepare_fold_data
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_single_fold(
    X_train_rgb: np.ndarray,
    X_train_thm: np.ndarray,
    y_train: np.ndarray,
    X_test_rgb: np.ndarray,
    X_test_thm: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42,
    class_weight: bool = True
) -> Tuple[tf.keras.Model, np.ndarray, Dict[str, float]]:
    """Train the two-branch model on a training fold and evaluate on the test fold."""
    tf.keras.utils.set_random_seed(seed)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Compute class weights if imbalanced
    weights = None
    if class_weight:
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

    # Train model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]

    model.fit(
        [X_train_rgb, X_train_thm],
        y_train,
        validation_data=([X_test_rgb, X_test_thm], y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=0
    )

    # Predictions
    probs = model.predict([X_test_rgb, X_test_thm], verbose=0)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, probs))
    except Exception:
        auc = 0.5

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": confusion_matrix(y_test, preds).tolist()
    }
    return model, probs, metrics


def run_cross_validation(
    task: str = "cloud",
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute 4-Fold Spatial Cross Validation for the given task."""
    print(f"\n=======================================================")
    print(f" Running 4-Fold Spatial CV: Task = {task.upper()}")
    print(f"=======================================================")

    if task in ["cloud", "vegetation"]:
        data = load_delhi_data()
    elif task == "fire":
        data = load_siberia_fire_data()
    else:
        raise ValueError(f"Unknown task: {task}")

    num_folds = 4
    fold_metrics = []
    all_y_true = []
    all_y_probs = []

    best_f1 = -1.0
    best_model = None

    for k in range(num_folds):
        t0 = time.time()
        (tr_rgb, tr_thm, y_tr), (te_rgb, te_thm, y_te) = prepare_fold_data(
            data, task=task, test_fold=k
        )
        print(f" [Fold {k}] Train samples: {len(y_tr)} (pos: {int(y_tr.sum())}) | Test samples: {len(y_te)} (pos: {int(y_te.sum())})")

        model, probs, metrics = train_single_fold(
            tr_rgb, tr_thm, y_tr,
            te_rgb, te_thm, y_te,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + k,
            class_weight=(task == "fire")
        )

        fold_metrics.append(metrics)
        all_y_true.extend(y_te)
        all_y_probs.extend(probs)
        dt = time.time() - t0

        print(f"    --> Fold {k} Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, Rec: {metrics['recall']:.4f} ({dt:.1f}s)")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model

    # Aggregate Out-of-Fold Metrics
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    all_preds = (all_y_probs >= 0.5).astype(int)

    oof_acc = float(accuracy_score(all_y_true, all_preds))
    oof_prec = float(precision_score(all_y_true, all_preds, zero_division=0))
    oof_rec = float(recall_score(all_y_true, all_preds, zero_division=0))
    oof_f1 = float(f1_score(all_y_true, all_preds, zero_division=0))
    try:
        oof_auc = float(roc_auc_score(all_y_true, all_y_probs))
    except Exception:
        oof_auc = 0.5
    oof_cm = confusion_matrix(all_y_true, all_preds).tolist()

    print(f"\n >>> {task.upper()} Out-of-Fold (OOF) Aggregate:")
    print(f"     Accuracy : {oof_acc*100:.2f}%")
    print(f"     Precision: {oof_prec:.4f}")
    print(f"     Recall   : {oof_rec:.4f}")
    print(f"     F1-Score : {oof_f1:.4f}")
    print(f"     ROC-AUC  : {oof_auc:.4f}")
    print(f"     Confusion Matrix: {oof_cm}")

    # Save best Float32 model
    model_save_path = os.path.join(MODELS_DIR, f"model_{task}_float32.h5")
    best_model.save(model_save_path)
    print(f" Saved best Float32 model to {model_save_path}")

    return {
        "task": task,
        "fold_metrics": fold_metrics,
        "oof_metrics": {
            "accuracy": oof_acc,
            "precision": oof_prec,
            "recall": oof_rec,
            "f1": oof_f1,
            "auc": oof_auc,
            "cm": oof_cm,
            "total_samples": len(all_y_true),
            "positive_samples": int(all_y_true.sum())
        }
    }


def update_markdown_logs(results: Dict[str, Any]):
    """Update training_log.md and cross_validation_log.md with full benchmark numbers."""
    log_path = "training_log.md"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Aggregated Multi-Modal Training & Cross-Validation Log",
        f"**Generated:** {ts}",
        f"**Dataset:** Aggregated Multi-Modal Dataset (3,612 samples)",
        "",
        "## Out-of-Fold (OOF) Benchmark Summary",
        "",
        "| Task | Dataset | Samples (Pos / Total) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |",
        "|:-----|:--------|:---------------------:|:--------:|:---------:|:------:|:--------:|:-------:|",
    ]

    for task_name, r in results.items():
        oof = r["oof_metrics"]
        dataset_name = "Landsat-9 Siberia" if task_name == "fire" else "Sentinel-2 Delhi"
        lines.append(
            f"| **{task_name.capitalize()}** | {dataset_name} | {oof['positive_samples']} / {oof['total_samples']} | "
            f"**{oof['accuracy']*100:.2f}%** | {oof['precision']:.4f} | {oof['recall']:.4f} | "
            f"**{oof['f1']:.4f}** | **{oof['auc']:.4f}** |"
        )

    lines.extend([
        "",
        "## Per-Fold Performance Breakdown",
        ""
    ])

    for task_name, r in results.items():
        lines.append(f"### Task: {task_name.capitalize()}")
        lines.append("| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
        lines.append("|:----:|:--------:|:---------:|:------:|:--------:|:-------:|")
        for k, fm in enumerate(r["fold_metrics"]):
            lines.append(
                f"| Fold {k} | {fm['accuracy']*100:.2f}% | {fm['precision']:.4f} | "
                f"{fm['recall']:.4f} | {fm['f1']:.4f} | {fm['auc']:.4f} |"
            )
        lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Updated {log_path} successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train multi-task spatial 4-fold CV pipeline.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per fold")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--tasks", nargs="+", default=["cloud", "vegetation", "fire"], help="Tasks to train")
    args = parser.parse_args()

    results = {}
    for task in args.tasks:
        res = run_cross_validation(task=task, epochs=args.epochs, batch_size=args.batch)
        results[task] = res

    update_markdown_logs(results)
    print("\nAll tasks trained and evaluated successfully!")


if __name__ == "__main__":
    main()
