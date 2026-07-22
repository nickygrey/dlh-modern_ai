#!/usr/bin/env python3
"""Module to build a Decision Tree Classifier."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Build and return a DecisionTreeClassifier instance.

    Args:
        min_samples_leaf (int): Min samples required at a leaf node.
        min_samples_split (int): Min samples required to split a node.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.tree.DecisionTreeClassifier: Configured classifier.
    """
    return tree.DecisionTreeClassifier(
        criterion='gini',
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
