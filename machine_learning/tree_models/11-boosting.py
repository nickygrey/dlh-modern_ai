#!/usr/bin/env python3
"""Module to compare boosting classifiers."""
from sklearn import ensemble
import lightgbm as lgb
import xgboost as xgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Initialize and return an untrained boosting classifier.

    Args:
        name (str): Name of boosting algorithm.
        n_estimators (int): Number of boosting iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        An untrained boosting classifier instance.

    Raises:
        ValueError: If name is not a supported boosting algorithm.
    """
    if name == 'adaboost':
        return ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == 'gradientboosting':
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")
