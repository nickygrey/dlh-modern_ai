#!/usr/bin/env python3
"""Module to generate SHAP explainer and SHAP values for models."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Create SHAP explainer using X_train and compute values for X_test.

    Args:
        model: A trained regression model.
        X_train: Input training data used as background dataset.
        X_test: Input test data to explain.

    Returns:
        tuple: (explainer, shap_values) containing the SHAP explainer
               object and SHAP values calculated on X_test.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
