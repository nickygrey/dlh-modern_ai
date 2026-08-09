#!/usr/bin/env python3
"""Module to save and load Keras models."""
from tensorflow import keras


def save_model(model, filepath):
    """Save a Keras model to the specified filepath.

    Args:
        model: A trained Keras model instance.
        filepath (str): Path where the model will be saved.

    Returns:
        None
    """
    model.save(filepath)


def load_model(filepath):
    """Load a Keras model from the specified filepath.

    Args:
        filepath (str): Path from where the model will be loaded.

    Returns:
        keras.Model: Reloaded Keras model instance.
    """
    return keras.models.load_model(filepath)
