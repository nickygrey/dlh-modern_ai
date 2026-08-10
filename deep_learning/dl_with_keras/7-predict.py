#!/usr/bin/env python3
"""Module to generate predictions using a trained Keras model."""
import tensorflow as tf


def predict(model, X, verbose=0):
    """Generate class predictions for input data using a Keras model.

    Args:
        model: A trained Keras model instance.
        X: Input data array with shape (num_examples, input_features).
        verbose (int): Verbosity level during prediction.

    Returns:
        tf.Tensor: Predicted class labels for the input data.
    """
    probabilities = model.predict(X, verbose=verbose)
    return tf.argmax(probabilities, axis=1)
