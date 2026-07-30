#!/usr/bin/env python3
"""Module containing the Linear_Regression function."""
from sklearn import linear_model


def Linear_Regression():
    """Create and return an untrained Linear Regression model.

    Returns:
        linear_model.LinearRegression: An untrained linear regression model.
    """
    return linear_model.LinearRegression()
