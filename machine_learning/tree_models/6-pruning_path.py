#!/usr/bin/env python3
"""Module to retrieve the cost-complexity pruning path for a decision tree."""


def get_pruning_path(clf, X, y):
    """Retrieve the cost-complexity pruning path for a DecisionTreeClassifier.

    Args:
        clf: DecisionTreeClassifier instance.
        X: Input feature matrix.
        y: Target label array.

    Returns:
        tuple: (ccp_alphas, impurities) as NumPy arrays.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
