"""
evaluate_float32.py — Task 5.4 (Issue #25): Evaluate Float32 Accuracy

Loads the trained float32 Keras model (models/model_float32.h5) and the
held-out test split, runs inference, and reports all metrics defined in
task_definition.md (accuracy, precision, recall, F1, confusion matrix).

Produces:
    evaluation_float32.md — full evaluation report

Usage:
    python evaluate_float32.py

Requires train_pipeline.py to have run first (needs model_float32.h5 and
the same manifest/seed so the same test split is produced).
"""

import os
import csv
import datetime
import numpy as np

# Re-use helpers from train_pipeline
from train_pipeline import (
    load_manifest, synthetic_dataset, build_dataset,
    stratified_split, classification_report, confusion_matrix_2x2,
    DEFAULT_SEED, MANIFEST_PATH,
)

MODELS_DIR  = "models"
H5_PATH     = os.path.join(MODELS_DIR, "model_float32.h5")
REPORT_PATH = "evaluation_float32.md"


def run_evaluation(seed: int = DEFAULT_SEED):
    import tensorflow as tf

    print("=" * 60)
    print("evaluate_float32.py — Issue #25 / Task 5.4")
    print("=" * 60)

    # 1. Load model
    if not os.path.isfile(H5_PATH):
        raise FileNotFoundError(
            f"{H5_PATH} not found — run train_pipeline.py first."
        )
    print(f"  Loading float32 model from {H5_PATH} ...")
    model = tf.keras.models.load_model(H5_PATH)
    print(f"  Parameters: {model.count_params():,}")

    # 2. Reproduce the SAME test split as training
    rows = load_manifest(MANIFEST_PATH)
    note = "real manifest"
    if len(rows) < 10:
        print("  Manifest sparse — using synthetic proxy dataset.")
        rows = synthetic_dataset(n_cloud=60, n_nocloud=60, seed=seed)
        note = "synthetic proxy data"

    rgb_arr, thm_arr, lbl_arr = build_dataset(rows, seed)
    if rgb_arr is None:
        raise RuntimeError("Dataset build failed.")

    _, _, te_idx = stratified_split(rows, lbl_arr, seed)
    X_te_rgb = rgb_arr[te_idx]
    X_te_thm = thm_arr[te_idx]
    y_true   = lbl_arr[te_idx]

    print(f"  Test split: {len(te_idx)} samples ({note})")

    # 3. Predict
    print("  Running inference ...")
    y_prob = model.predict([X_te_rgb, X_te_thm], verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # 4. Metrics
    m = classification_report(y_true, y_pred)

    print(f"\n  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    print(f"  Confusion matrix — TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    # 5. Per-class metrics
    # No-cloud class (class 0)
    nc_prec = m['tn'] / (m['tn'] + m['fn']) if (m['tn'] + m['fn']) > 0 else 0.0
    nc_rec  = m['tn'] / (m['tn'] + m['fp']) if (m['tn'] + m['fp']) > 0 else 0.0
    nc_f1   = (2*nc_prec*nc_rec/(nc_prec+nc_rec)) if (nc_prec+nc_rec) > 0 else 0.0

    # Cloud class (class 1)
    c_prec  = m['precision']
    c_rec   = m['recall']
    c_f1    = m['f1']

    # 6. Write report
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Float32 Model Evaluation Report — Issue #25 / Task 5.4",
        "",
        f"**Generated:** {now}  ",
        f"**Model:** `{H5_PATH}`  ",
        f"**Parameters:** {model.count_params():,}  ",
        f"**Data source:** {note}  ",
        f"**Test samples:** {len(te_idx)}",
        "",
        "## Overall Metrics",
        "",
        "| Metric    | Value   |",
        "|:----------|--------:|",
        f"| Accuracy  | {m['accuracy']:.4f} ({m['accuracy']:.1%}) |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall    | {m['recall']:.4f} |",
        f"| F1-Score  | {m['f1']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "| | **Predicted: No-Cloud** | **Predicted: Cloud** |",
        "|:--|:---:|:---:|",
        f"| **True: No-Cloud** | {m['tn']} | {m['fp']} |",
        f"| **True: Cloud**    | {m['fn']} | {m['tp']} |",
        "",
        "## Per-Class Report",
        "",
        "| Class     | Precision | Recall | F1-Score | Support |",
        "|:----------|----------:|-------:|---------:|--------:|",
        f"| No-Cloud (0) | {nc_prec:.4f} | {nc_rec:.4f} | {nc_f1:.4f} | {m['tn']+m['fp']} |",
        f"| Cloud    (1) | {c_prec:.4f}  | {c_rec:.4f}  | {c_f1:.4f}  | {m['fn']+m['tp']} |",
        "",
        "## Acceptance Criteria",
        "",
        f"- All metrics computed and documented: ✅",
        f"- Confusion matrix generated: ✅",
        f"- Accuracy > 90%: {'✅' if m['accuracy'] > 0.90 else '⚠️'} ({m['accuracy']:.1%})",
        "",
        "## Proxy Data Note",
        "",
        "Evaluation is on synthetic proxy tiles (see `proxy_data_caveats.md`).",
        "Replace with real Sentinel-2 tiles for production evaluation.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  -> {REPORT_PATH} written")
    print("\n[PASS] Issue #25 complete - evaluate_float32.py done.")
    return m


if __name__ == "__main__":
    run_evaluation()
