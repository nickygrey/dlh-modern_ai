#!/usr/bin/env python3
"""Module to create a Lasso Regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """Create and return an untrained Lasso regression model.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        linear_model.Lasso: An untrained Lasso regression model instance.
    """
    return linear_model.Lasso(random_state=random_state)
