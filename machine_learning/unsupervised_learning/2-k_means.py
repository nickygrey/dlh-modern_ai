#!/usr/bin/env python3
"""Module to perform K-Means clustering."""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Fit a K-Means clustering model on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to find.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.cluster.KMeans: The fitted KMeans instance.
    """
    kmeans = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    return kmeans.fit(X)
