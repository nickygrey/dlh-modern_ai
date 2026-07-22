#!/usr/bin/env python3
"""Module to generate predictions using a trained classifier."""


def generate_predictions(clf, X):
    """Generate predicted class labels for input samples.

    Args:
        clf: Trained Scikit-learn classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    return clf.predict(X)
