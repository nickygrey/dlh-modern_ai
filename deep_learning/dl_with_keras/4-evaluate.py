#!/usr/bin/env python3
"""Module to evaluate a trained Keras model's performance."""


def evaluate_model(model, X, Y, verbose=0):
    """Assess a trained Keras model's performance on a given dataset.

    Args:
        model: A trained Keras model instance.
        X: Input features array with shape (num_examples, input_features).
        Y: True labels array with shape (num_examples, num_classes).
        verbose (int): Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        tuple: (loss, accuracy) calculated on the provided data.
    """
    loss, accuracy = model.evaluate(X, Y, verbose=verbose)
    return loss, accuracy
