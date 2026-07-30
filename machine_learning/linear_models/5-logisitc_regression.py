#!/usr/bin/env python3
"""Module to instantiate a Logistic Regression classifier."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Create and return an untrained Logistic Regression model.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        linear_model.LogisticRegression: An untrained model instance.
    """
    return linear_model.LogisticRegression(random_state=random_state)
