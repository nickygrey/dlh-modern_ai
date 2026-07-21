#!/usr/bin/env python3
"""Module to evaluate optimal number of clusters for K-Means."""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means performance across different cluster counts.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>= 2).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (ks, inertia_values, silhouette_values) where:
            - ks (list[int]): List of cluster numbers evaluated.
            - inertia_values (list[float]): Inertia for each k.
            - silhouette_values (list[float]): Silhouette score for each k.
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)
        ks.append(k)
        inertia_values.append(model.inertia_)
        score = metrics.silhouette_score(X, model.labels_)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
