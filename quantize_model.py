"""
quantize_model.py — Task 5.5 (Issue #26): Convert to TFLite + Int8 Quantization

Converts models/model_float32.h5 to:
    1. models/model_float32.tflite  — baseline float32 TFLite flatbuffer
    2. models/model_int8.tflite     — post-training int8 quantized TFLite

Provides a representative_dataset built from the same synthetic proxy tiles
used during training (real tiles slot in automatically when available).

Usage:
    python quantize_model.py

Acceptance Criteria (Issue #26):
    - Both TFLite models load and run without errors
    - Int8 quantization applied
"""

import os
import datetime
import numpy as np

from train_pipeline import (
    load_manifest, synthetic_dataset, build_dataset,
    DEFAULT_SEED, MANIFEST_PATH, RGB_SHAPE, THERMAL_SHAPE,
)

MODELS_DIR  = "models"
H5_PATH     = os.path.join(MODELS_DIR, "model_float32.h5")
F32_TFLITE  = os.path.join(MODELS_DIR, "model_float32.tflite")
INT8_TFLITE = os.path.join(MODELS_DIR, "model_int8.tflite")


def _representative_dataset(rgb_arr, thm_arr, n_samples: int = 50):
    """Generator yielding calibration batches for int8 quantization."""
    indices = np.random.RandomState(DEFAULT_SEED).permutation(len(rgb_arr))[:n_samples]
    for i in indices:
        yield [
            rgb_arr[i:i+1].astype(np.float32),
            thm_arr[i:i+1].astype(np.float32),
        ]


def convert(seed: int = DEFAULT_SEED):
    import tensorflow as tf

    print("=" * 60)
    print("quantize_model.py — Issue #26 / Task 5.5")
    print("=" * 60)

    # 1. Load Keras model
    if not os.path.isfile(H5_PATH):
        raise FileNotFoundError(
            f"{H5_PATH} not found — run train_pipeline.py first."
        )
    print(f"  Loading {H5_PATH} ...")
    model = tf.keras.models.load_model(H5_PATH)
    print(f"  Params: {model.count_params():,}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Float32 TFLite
    print("\n  Converting to float32 TFLite ...")
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter_f32.convert()
    with open(F32_TFLITE, "wb") as fh:
        fh.write(tflite_f32)
    f32_kb = len(tflite_f32) / 1024
    print(f"  -> {F32_TFLITE}  ({f32_kb:.1f} KB)")

    # 3. Verify float32 TFLite
    print("  Verifying float32 TFLite interpreter ...")
    interp_f32 = tf.lite.Interpreter(model_content=tflite_f32)
    interp_f32.allocate_tensors()
    in_details = interp_f32.get_input_details()
    out_details = interp_f32.get_output_details()
    # Run one forward pass
    dummy_rgb = np.zeros((1, *RGB_SHAPE), dtype=np.float32)
    dummy_thm = np.zeros((1, *THERMAL_SHAPE), dtype=np.float32)
    for inp in in_details:
        if inp["shape"][1] == RGB_SHAPE[0]:
            interp_f32.set_tensor(inp["index"], dummy_rgb)
        else:
            interp_f32.set_tensor(inp["index"], dummy_thm)
    interp_f32.invoke()
    out_f32 = interp_f32.get_tensor(out_details[0]["index"])
    print(f"  Float32 TFLite test output: {out_f32} [OK]")

    # 4. Build representative dataset
    rows = load_manifest(MANIFEST_PATH)
    if len(rows) < 10:
        rows = synthetic_dataset(n_cloud=60, n_nocloud=60, seed=seed)
    rgb_arr, thm_arr, _ = build_dataset(rows, seed)

    # 5. Int8 quantization
    print("\n  Converting to int8 quantized TFLite ...")
    converter_i8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_i8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_i8.representative_dataset = lambda: _representative_dataset(rgb_arr, thm_arr)
    converter_i8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_i8.inference_input_type  = tf.int8
    converter_i8.inference_output_type = tf.int8
    try:
        tflite_i8 = converter_i8.convert()
    except Exception as e:
        print(f"  [WARN] Full int8 I/O conversion failed ({e}) - "
              "retrying with float I/O (dynamic-range int8) ...")
        converter_i8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_i8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_i8.representative_dataset = lambda: _representative_dataset(rgb_arr, thm_arr)
        tflite_i8 = converter_i8.convert()

    with open(INT8_TFLITE, "wb") as fh:
        fh.write(tflite_i8)
    i8_kb = len(tflite_i8) / 1024
    print(f"  -> {INT8_TFLITE}  ({i8_kb:.1f} KB)")

    # 6. Verify int8 TFLite
    print("  Verifying int8 TFLite interpreter ...")
    interp_i8 = tf.lite.Interpreter(model_content=tflite_i8)
    interp_i8.allocate_tensors()
    in_details_i8  = interp_i8.get_input_details()
    out_details_i8 = interp_i8.get_output_details()
    for inp in in_details_i8:
        dtype = inp["dtype"]
        if inp["shape"][1] == RGB_SHAPE[0]:
            dummy = np.zeros((1, *RGB_SHAPE), dtype=dtype)
        else:
            dummy = np.zeros((1, *THERMAL_SHAPE), dtype=dtype)
        interp_i8.set_tensor(inp["index"], dummy)
    interp_i8.invoke()
    out_i8 = interp_i8.get_tensor(out_details_i8[0]["index"])
    print(f"  Int8 TFLite test output: {out_i8} [OK]")

    # 7. Size report
    reduction = (1 - i8_kb / f32_kb) * 100
    print(f"\n  Size: float32={f32_kb:.1f} KB -> int8={i8_kb:.1f} KB "
          f"({reduction:.1f}% reduction)")
    print("\n[PASS] Issue #26 complete - quantize_model.py done.")

    return f32_kb, i8_kb


if __name__ == "__main__":
    convert()
