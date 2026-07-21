#!/usr/bin/env python3
"""Module to perform agglomerative hierarchical clustering."""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(
    X, n_clusters, random_state, n_components, use_pca_data=True
):
    """Perform Agglomerative hierarchical clustering on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.
        n_components (int): Number of PCA components.
        use_pca_data (bool): Whether to apply PCA before clustering.

    Returns:
        tuple: (model, X_used, score) where:
            - model: Fitted AgglomerativeClustering instance.
            - X_used: Data array used for fitting.
            - score: Silhouette score of the clustering (or None).
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(
            X,
            n_components=n_components,
            random_state=random_state
        )
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    model.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None

    return model, X_used, score
