"""
quantize_model.py — Multi-Task Int8 Quantization for Edge AI Models

Converts Float32 Keras models to:
  1. models/model_{task}_float32.tflite — baseline Float32 TFLite flatbuffer
  2. models/model_{task}_int8.tflite    — post-training full-integer Int8 TFLite

Supports tasks: 'cloud', 'vegetation', 'fire', and creates default aliases.
Calibrates using representative samples from the Aggregated Dataset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from load_aggregated_data import (
    load_delhi_data,
    load_siberia_fire_data,
    prepare_fold_data
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def representative_dataset_gen(rgb_arr: np.ndarray, thm_arr: np.ndarray, n_samples: int = 100):
    """Yield multi-modal calibration batches for full integer Int8 quantization."""
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(rgb_arr))[:n_samples]
    for i in indices:
        yield [
            rgb_arr[i:i+1].astype(np.float32),
            thm_arr[i:i+1].astype(np.float32)
        ]


def quantize_task_model(task: str = "cloud"):
    print(f"\n=======================================================")
    print(f" Quantizing Model: Task = {task.upper()}")
    print(f"=======================================================")

    h5_path = os.path.join(MODELS_DIR, f"model_{task}_float32.h5")
    if not os.path.exists(h5_path):
        # fallback check for generic model_float32.h5
        if task == "cloud" and os.path.exists(os.path.join(MODELS_DIR, "model_float32.h5")):
            h5_path = os.path.join(MODELS_DIR, "model_float32.h5")
        else:
            raise FileNotFoundError(f"Float32 model not found at {h5_path}. Run train_aggregated_pipeline.py first.")

    model = tf.keras.models.load_model(h5_path)
    print(f" Loaded model: {h5_path} (Parameters: {model.count_params():,})")

    f32_tflite_path = os.path.join(MODELS_DIR, f"model_{task}_float32.tflite")
    int8_tflite_path = os.path.join(MODELS_DIR, f"model_{task}_int8.tflite")

    # 1. Convert to Float32 TFLite
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    f32_content = converter_f32.convert()
    with open(f32_tflite_path, "wb") as f:
        f.write(f32_content)
    f32_kb = len(f32_content) / 1024.0
    print(f" -> Exported Float32 TFLite: {f32_tflite_path} ({f32_kb:.2f} KB)")

    # 2. Prepare calibration data
    if task in ["cloud", "vegetation"]:
        raw_data = load_delhi_data()
    else:
        raw_data = load_siberia_fire_data()
    (tr_rgb, tr_thm, _), _ = prepare_fold_data(raw_data, task=task, test_fold=0)

    # 3. Convert to Int8 TFLite with Full Integer Quantization
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = lambda: representative_dataset_gen(tr_rgb, tr_thm, n_samples=100)
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32  # standard edge float input adapter
    converter_int8.inference_output_type = tf.float32

    int8_content = converter_int8.convert()
    with open(int8_tflite_path, "wb") as f:
        f.write(int8_content)
    int8_kb = len(int8_content) / 1024.0
    reduction = ((f32_kb - int8_kb) / f32_kb) * 100.0
    print(f" -> Exported Int8 TFLite   : {int8_tflite_path} ({int8_kb:.2f} KB, -{reduction:.1f}% size reduction)")

    # Create default aliases if cloud
    if task == "cloud":
        with open(os.path.join(MODELS_DIR, "model_float32.tflite"), "wb") as f:
            f.write(f32_content)
        with open(os.path.join(MODELS_DIR, "model_int8.tflite"), "wb") as f:
            f.write(int8_content)
        print(" -> Created default aliases models/model_float32.tflite & models/model_int8.tflite")

    return {
        "task": task,
        "f32_kb": f32_kb,
        "int8_kb": int8_kb,
        "reduction_pct": reduction
    }


def main():
    tasks = ["cloud", "vegetation", "fire"]
    for t in tasks:
        quantize_task_model(t)
    print("\nAll models quantized to Int8 TFLite successfully!")


if __name__ == "__main__":
    main()
