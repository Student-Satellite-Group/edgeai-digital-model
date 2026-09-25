"""
model_architecture.py — Task 5.2 (revised): two-branch, feature-level
(late) fusion CNN for cloud/no-cloud classification (see task_definition.md).

SUPERSEDES the original single-branch, pixel-fusion design. Two problems
drove the change:

1. Compute cost: the original 256x256 single-input model measured ~654M
   MACs/inference (TF FLOPs profiler) -- unworkable on ESP32-S3-class
   hardware (no NN accelerator, architecture doc Section 4.2), despite its
   parameter count (187,650) looking fine in isolation. Root cause: full
   resolution processed through several blocks before any downsampling, and
   pooling *after* the expensive pointwise convs rather than a strided
   depthwise conv doing the downsampling for free.

2. Resolution-mismatch handling: the original design warped the native
   32x24 thermal grid (768 real pixels) onto the RGB frame's full pixel grid
   via homography before fusing -- meaning nearly every "thermal" pixel the
   classifier saw was interpolated, not measured. That is exactly the
   "silently up-sampled and assumed aligned" failure mode Architecture
   document Section 2.2 says this project is designed to avoid; the original
   pixel-fusion mechanism reintroduced it one step later than expected.

This version processes each modality at its own native/appropriate
resolution and fuses learned *feature vectors* instead of raw pixels:

    RGB branch  (128x128x3, strided depthwise-separable CNN) -> GAP -> f_rgb
    Thermal branch (32x24x1, native resolution, tiny CNN)    -> GAP -> f_thermal
    Concatenate([f_rgb, f_thermal]) -> Dense(32, ReLU) -> Dropout(0.3)
        -> Dense(2, softmax)  ->  cloud / no-cloud

Single-task only (cloud/no-cloud) -- a second head for thermal-anomaly
detection was discussed and deliberately NOT added here, since no anomaly
label/definition exists yet (task_definition.md locks cloud/no-cloud only).

registration.py's role changes with this design: it no longer needs to warp
thermal onto RGB's full pixel grid for this classifier's input -- it only
needs to identify which RGB crop corresponds to the current thermal
footprint, so both branches see the same physical patch of ground.
fusion.py's pixel-level weighted-overlay output is unaffected by this file
but is no longer this classifier's input; it remains a separate data
product (Section 7.3's "fused output frame") for logging/visualization.

This is the model definition only -- training happens in Task 5.3.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

RGB_INPUT_SHAPE = (128, 128, 3)
THERMAL_INPUT_SHAPE = (32, 24, 1)


def _rgb_branch(rgb_input, widths=(16, 32, 64, 128)):
    """Strided depthwise-separable CNN over the RGB crop, ending in a
    GlobalAveragePooling2D feature vector (learned RGB descriptor).

    Downsampling happens via stride=2 on the initial conv and every
    depthwise conv (not a separate MaxPooling2D after full-resolution
    pointwise convs) -- this is what fixes the compute-cost problem: the
    expensive 1x1 pointwise conv in each block runs at the already-halved
    resolution, not before it.
    """
    x = layers.Conv2D(widths[0], 3, strides=2, padding="same")(rgb_input)
    x = layers.BatchNormalization(momentum=0.6)(x)
    x = layers.ReLU()(x)

    for filters in widths[1:]:
        x = layers.DepthwiseConv2D(3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.6)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.6)(x)
        x = layers.ReLU()(x)

    return layers.GlobalAveragePooling2D(name="f_rgb")(x)


def _thermal_branch(thermal_input, widths=(8, 16)):
    """Tiny CNN over the *native-resolution* 32x24 thermal grid -- no
    upsampling. There are only 768 real input values, so this branch is
    deliberately shallow: two conv layers, the second strided, then GAP."""
    x = layers.Conv2D(widths[0], 3, padding="same")(thermal_input)
    x = layers.BatchNormalization(momentum=0.6)(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.6)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(widths[1], 1, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.6)(x)
    x = layers.ReLU()(x)

    return layers.GlobalAveragePooling2D(name="f_thermal")(x)


def build_model(
    rgb_input_shape=RGB_INPUT_SHAPE,
    thermal_input_shape=THERMAL_INPUT_SHAPE,
    num_classes=2,
    max_params=2_000_000,
):
    """Build the two-branch, feature-level fusion CNN.

    Returns a Keras Model taking [rgb_input, thermal_input] and producing a
    (num_classes,) softmax. Raises AssertionError if parameter count exceeds
    max_params (same acceptance criterion as the original Task 5.2).
    """
    rgb_input = layers.Input(shape=rgb_input_shape, name="rgb_input")
    thermal_input = layers.Input(shape=thermal_input_shape, name="thermal_input")

    f_rgb = _rgb_branch(rgb_input)
    f_thermal = _thermal_branch(thermal_input)

    fused = layers.Concatenate(name="fused_features")([f_rgb, f_thermal])
    x = layers.Dense(32, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model([rgb_input, thermal_input], outputs)

    param_count = model.count_params()
    assert param_count <= max_params, (
        f"Model has {param_count} params, exceeds ceiling {max_params}"
    )
    return model


def _measure_macs(model, rgb_shape=RGB_INPUT_SHAPE, thermal_shape=THERMAL_INPUT_SHAPE):
    """Measures real MACs via TensorFlow's FLOPs profiler (not an estimate) --
    this is the check that would have caught the original design's ~654M
    MAC/inference cost before it shipped."""
    @tf.function
    def fwd(rgb, thm):
        return model([rgb, thm])

    concrete = fwd.get_concrete_function(
        tf.TensorSpec([1, *rgb_shape], tf.float32),
        tf.TensorSpec([1, *thermal_shape], tf.float32),
    )
    info = tf.compat.v1.profiler.profile(
        graph=concrete.graph,
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
    )
    return info.total_float_ops / 2.0  # MACs = FLOPs / 2


if __name__ == "__main__":
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    macs = _measure_macs(model)
    print(f"\nVerified: total params {model.count_params():,} (ceiling: 2,000,000).")
    print(f"Verified: {macs:,.0f} MACs/inference "
          f"(vs. ~654,000,000 MACs for the original single-branch, 256x256 design).")
    print("Run Task 5.3 training pipeline (train_model.py) to train.")
