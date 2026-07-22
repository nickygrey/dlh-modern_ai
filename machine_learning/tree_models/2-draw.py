#!/usr/bin/env python3
"""Module to display the decision rules of a trained decision tree."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print a text representation of the decision tree rules.

    Args:
        clf: Trained DecisionTreeClassifier instance.
        feature_names (list): List of input feature names.
        class_names (list): List of target class names.

    Returns:
        None
    """
    rules = tree.export_text(
        clf,
        feature_names=feature_names,
        class_names=list(class_names)
    )
    print(rules, end="")
