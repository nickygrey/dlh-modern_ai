#!/usr/bin/env python3
"""Module to extract and sort Random Forest feature importances."""
import numpy as np


def feature_importance(rf):
    """Compute feature importances and indices sorted in ascending order.

    Args:
        rf: A trained Scikit-learn RandomForestClassifier instance.

    Returns:
        tuple: (importances, indices) where importances is a NumPy array
               of scores and indices is a NumPy array of feature indices
               sorted from least to most important.
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
