"""
quantization_delta.py — Task 5.6 (Issue #27): Compute Quantization Accuracy Delta

Runs BOTH the float32 TFLite model and the int8 TFLite model on the IDENTICAL
held-out test set (same seed/split as train_pipeline.py), then computes the
accuracy drop and categorises it:

    < 2 %  → Acceptable
    2–5 %  → Marginal (review)
    > 5 %  → Unacceptable (trigger QAT fallback, Issue #28)

Outputs:
    quantization_delta.md

Usage:
    python quantization_delta.py

Requires train_pipeline.py and quantize_model.py to have run first.
"""

import os
import datetime
import numpy as np

from train_pipeline import (
    load_manifest, synthetic_dataset, build_dataset,
    stratified_split, classification_report,
    DEFAULT_SEED, MANIFEST_PATH, RGB_SHAPE, THERMAL_SHAPE,
)

MODELS_DIR  = "models"
F32_TFLITE  = os.path.join(MODELS_DIR, "model_float32.tflite")
INT8_TFLITE = os.path.join(MODELS_DIR, "model_int8.tflite")
DELTA_PATH  = "quantization_delta.md"


def _run_tflite(model_path, rgb_arr, thm_arr):
    """Run TFLite model on all samples; return predicted class indices."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    preds = []
    for i in range(len(rgb_arr)):
        for inp in in_details:
            dtype = inp["dtype"]
            if inp["shape"][1] == RGB_SHAPE[0]:
                val = rgb_arr[i:i+1].astype(np.float32)
            else:
                val = thm_arr[i:i+1].astype(np.float32)

            # Check if input is quantized (int8)
            if dtype == np.int8:
                scales = inp.get("quantization_parameters", {}).get("scales", [])
                zero_points = inp.get("quantization_parameters", {}).get("zero_points", [])
                if len(scales) > 0 and scales[0] > 0:
                    val = np.round(val / scales[0] + zero_points[0])
                else:
                    val = val * 127
                val = np.clip(val, -128, 127).astype(np.int8)
            else:
                val = val.astype(dtype)

            interp.set_tensor(inp["index"], val)
        interp.invoke()
        out = interp.get_tensor(out_details[0]["index"])[0]
        preds.append(int(np.argmax(out)))
    return np.array(preds)


def compute_delta(seed: int = DEFAULT_SEED):
    print("=" * 60)
    print("quantization_delta.py - Issue #27 / Task 5.6")
    print("=" * 60)

    for path in [F32_TFLITE, INT8_TFLITE]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found — run quantize_model.py first."
            )

    # 1. Reproduce identical test split
    rows = load_manifest(MANIFEST_PATH)
    note = "real manifest"
    if len(rows) < 10:
        rows = synthetic_dataset(n_cloud=60, n_nocloud=60, seed=seed)
        note = "synthetic proxy data"

    rgb_arr, thm_arr, lbl_arr = build_dataset(rows, seed)
    _, _, te_idx = stratified_split(rows, lbl_arr, seed)

    X_te_rgb = rgb_arr[te_idx]
    X_te_thm = thm_arr[te_idx]
    y_true   = lbl_arr[te_idx]
    print(f"  Test split: {len(te_idx)} samples ({note})")

    # 2. Run both models
    print("  Running float32 TFLite ...")
    y_f32 = _run_tflite(F32_TFLITE, X_te_rgb, X_te_thm)
    print("  Running int8 TFLite ...")
    y_i8  = _run_tflite(INT8_TFLITE, X_te_rgb, X_te_thm)

    # 3. Compute accuracy
    m_f32 = classification_report(y_true, y_f32)
    m_i8  = classification_report(y_true, y_i8)

    acc_f32   = m_f32["accuracy"]
    acc_i8    = m_i8["accuracy"]
    delta     = acc_f32 - acc_i8
    delta_pct = delta * 100

    if abs(delta_pct) < 2.0:
        category = "Acceptable (< 2%)"
        qat_needed = False
    elif abs(delta_pct) < 5.0:
        category = "Marginal (2%-5%) - review recommended"
        qat_needed = False
    else:
        category = "Unacceptable (> 5%) - QAT fallback required (Issue #28)"
        qat_needed = True

    print(f"\n  Float32 accuracy : {acc_f32:.4f} ({acc_f32:.1%})")
    print(f"  Int8    accuracy : {acc_i8:.4f} ({acc_i8:.1%})")
    print(f"  Delta            : {delta_pct:+.2f} pp -> {category}")

    # 4. Write report
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Quantization Accuracy Delta — Issue #27 / Task 5.6",
        "",
        f"**Generated:** {now}  ",
        f"**Data source:** {note}  ",
        f"**Test samples:** {len(te_idx)}",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score |",
        "|:------|:--------:|:--------:|:------:|:--------:|",
        f"| Float32 TFLite | {acc_f32:.4f} | {m_f32['precision']:.4f} | {m_f32['recall']:.4f} | {m_f32['f1']:.4f} |",
        f"| Int8 TFLite    | {acc_i8:.4f}  | {m_i8['precision']:.4f}  | {m_i8['recall']:.4f}  | {m_i8['f1']:.4f}  |",
        f"| **Delta**      | **{delta_pct:+.2f} pp** | — | — | — |",
        "",
        "## Categorisation",
        "",
        f"**{category}**",
        "",
        "| Threshold | Action |",
        "|:----------|:-------|",
        "| < 2 pp    | Acceptable — deploy int8 as-is |",
        "| 2–5 pp    | Marginal — review, optional QAT |",
        "| > 5 pp    | Unacceptable — trigger QAT (Issue #28) |",
        "",
        f"**QAT required:** {'Yes — trigger Issue #28' if qat_needed else 'No'}",
        "",
        "## Float32 TFLite Confusion Matrix",
        "",
        "| | Pred No-Cloud | Pred Cloud |",
        "|:--|:---:|:---:|",
        f"| True No-Cloud | {m_f32['tn']} | {m_f32['fp']} |",
        f"| True Cloud    | {m_f32['fn']} | {m_f32['tp']} |",
        "",
        "## Int8 TFLite Confusion Matrix",
        "",
        "| | Pred No-Cloud | Pred Cloud |",
        "|:--|:---:|:---:|",
        f"| True No-Cloud | {m_i8['tn']} | {m_i8['fp']} |",
        f"| True Cloud    | {m_i8['fn']} | {m_i8['tp']} |",
        "",
        "## Acceptance Criteria",
        "",
        "- Both models evaluated on identical test set: ✅",
        "- Delta computed and categorised: ✅",
        f"- Hardware-independent measurement (dev-machine TFLite interpreter): ✅",
        "",
        "## Proxy Data Note",
        "",
        "All results are from synthetic proxy tiles (see `proxy_data_caveats.md`).",
        "Re-run after `download_tiles.py` provides real Sentinel-2 imagery.",
    ]

    with open(DELTA_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> {DELTA_PATH} written")
    print("\n[PASS] Issue #27 complete - quantization_delta.py done.")
    return delta_pct, qat_needed


if __name__ == "__main__":
    compute_delta()
