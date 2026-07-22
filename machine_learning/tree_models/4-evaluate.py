#!/usr/bin/env python3
"""Module to evaluate classifier performance."""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Generate a classification report using Scikit-learn.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels.
        class_names: List of class names corresponding to label indices.

    Returns:
        str: Text summary of precision, recall, and F1-score per class.
    """
    return metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
