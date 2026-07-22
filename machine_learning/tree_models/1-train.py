#!/usr/bin/env python3
"""Module to train a tree-based classifier."""


def train_tree(clf, X, y):
    """Train a Scikit-learn classifier on input features and target labels.

    Args:
        clf: Scikit-learn classifier instance.
        X: Input feature array or DataFrame.
        y: Target label array or Series.

    Returns:
        None
    """
    clf.fit(X, y)
