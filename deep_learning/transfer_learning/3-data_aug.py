#!/usr/bin/env python3
"""Module to build a data augmentation pipeline."""
from tensorflow import keras


def build_data_augmentation():
    """Build a Keras Sequential model for image data augmentation.

    Returns:
        keras.Sequential: Sequential model with data augmentation layers.
    """
    model = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42)
    ])
    return model
