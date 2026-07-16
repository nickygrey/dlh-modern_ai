#!/usr/bin/env python3
"""Module to standardize tabular data."""
from sklearn import preprocessing


def Standardize(X):
    """Standardize a 2D numpy array using StandardScaler.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
