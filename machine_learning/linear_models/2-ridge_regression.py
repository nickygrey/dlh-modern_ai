#!/usr/bin/env python3
"""Module to create a Ridge Regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """Create and return an untrained Ridge regression model.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        linear_model.Ridge: An untrained Ridge regression model instance.
    """
    return linear_model.Ridge(random_state=random_state)
