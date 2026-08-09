#!/usr/bin/env python3
"""Module to train a Keras model."""


def train_model(model, X, Y, epochs, verbose=1):
    """Train a Keras model on input data and labels.

    Args:
        model: Keras model instance to train.
        X: Input data array.
        Y: Target labels array.
        epochs (int): Number of training epochs.
        verbose (int): Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        None
    """
    model.fit(X, Y, epochs=epochs, verbose=verbose)
