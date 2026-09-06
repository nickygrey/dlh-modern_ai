#!/usr/bin/env python3
"""Module to attach a classification head to a feature extractor."""
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """Attach a custom classification head to a feature extractor.

    Args:
        base_model (keras.Model): Feature extractor base model.
        num_classes (int): Number of classification output classes.

    Returns:
        keras.Model: Model with attached classification head.
    """
    x = base_model.output
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model
