#!/usr/bin/env python3
"""Module to perform Principal Component Analysis (PCA) on tabular data."""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Perform PCA on tabular data using Scikit-learn.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int | float | None): Number of components or variance
            fraction to keep.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_pca, pca) where:
            - X_pca (numpy.ndarray): Transformed data.
            - pca (sklearn.decomposition.PCA): Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
