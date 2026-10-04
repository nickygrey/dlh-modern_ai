#!/usr/bin/env python3
"""Module to calculate evaluation metrics for a classification model."""
import numpy as np
import sklearn.metrics


def compute_metrics(predictions):
    """Calculate key evaluation metrics for classification predictions.

    Args:
        predictions: An object containing model predictions and label_ids
            (such as transformers.EvalPrediction or types.SimpleNamespace).

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1.
    """
    if hasattr(predictions, "predictions"):
        logits = predictions.predictions
        labels = predictions.label_ids
    elif isinstance(predictions, (tuple, list)):
        logits, labels = predictions
    else:
        logits = predictions["predictions"]
        labels = predictions["label_ids"]

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    accuracy = float(sklearn.metrics.accuracy_score(labels, preds))
    precision = float(sklearn.metrics.precision_score(
        labels, preds, average="weighted", zero_division=0
    ))
    recall = float(sklearn.metrics.recall_score(
        labels, preds, average="weighted", zero_division=0
    ))
    f1 = float(sklearn.metrics.f1_score(
        labels, preds, average="weighted", zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
