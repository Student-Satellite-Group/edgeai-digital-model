"""
model_architecture.py — Task 5.2: MobileNet-style depthwise-separable CNN for
cloud/no-cloud classification (see task_definition.md, Task 5.1).

Design target: < 2,000,000 parameters (edge parameter ceiling), reusing the
guide's structure — initial 3x3 convolution, four depthwise-separable blocks
with 1x1 pointwise projections and 2x2 max pooling, then global average
pooling + dropout + softmax head.

Layer diagram (Task 5.2 deliverable):

    Input (256,256,3)
      | Conv2D 32@3x3 (same) + BN + ReLU
      | DepthwiseSep block [64]  : DW 3x3 + BN+ReLU, 1x1 Conv 64 + BN+ReLU, MaxPool2x2
      | DepthwiseSep block [128]
      | DepthwiseSep block [256]
      | DepthwiseSep block [512]
      | GlobalAveragePooling2D
      | Dropout(0.3)
      | Dense 2 (softmax)  ->  cloud / no-cloud

This is the model definition only — training happens in Task 5.3.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(input_shape=(256, 256, 3), num_classes=2, max_params=2_000_000):
    """Build a MobileNet-style depthwise-separable CNN under the parameter ceiling.

    Raises AssertionError if the model exceeds `max_params` parameters.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable convolution blocks
    for filters in (64, 128, 256, 512):
        x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    # Verify parameter count is under the ceiling (acceptance criterion).
    param_count = model.count_params()
    assert param_count <= max_params, (
        f"Model has {param_count} params, exceeds ceiling {max_params}"
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print(f"\nVerified: total params {model.count_params()} under the 2M ceiling.")
    print("Run Task 5.3 training pipeline (train_model.py) to train.")