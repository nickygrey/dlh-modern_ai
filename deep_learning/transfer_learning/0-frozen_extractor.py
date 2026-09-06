#!/usr/bin/env python3
"""Module to build a frozen feature extractor using MobileNetV2."""
from tensorflow import keras


def build_feature_extractor(input_shape=(224, 224, 3)):
    """Build a feature extractor using a frozen pretrained MobileNetV2.

    Args:
        input_shape (tuple, optional): Shape of input images.
            Defaults to (224, 224, 3).

    Returns:
        keras.Model: Feature extractor model with frozen base and pooling.
    """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
