#!/usr/bin/env python3
"""Module to save and load Keras model weights."""


def save_model_weights(model, filepath):
    """Save only the weights of a trained Keras model.

    Args:
        model: A trained Keras model instance.
        filepath (str): File path where the weights will be saved.

    Returns:
        None
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """Load weights into a compatible Keras model instance.

    Args:
        model: A compatible Keras model instance.
        filepath (str): File path from where the weights will be loaded.

    Returns:
        None
    """
    model.load_weights(filepath)
