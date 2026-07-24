#!/usr/bin/env python3
"""Module to perform pre-pruning hyperparameter tuning on decision trees."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """Perform Grid Search to find optimal pre-pruning hyperparameters.

    Args:
        X: Input feature matrix.
        y: Target label array.
        clf: Untrained DecisionTreeClassifier instance.

    Returns:
        dict: Best combination of hyperparameters found during grid search.
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 5)),
        'min_samples_leaf': list(range(2, 5)),
        'min_samples_split': list(range(2, 5))
    }

    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid
    )
    grid_search.fit(X, y)

    return grid_search.best_params_
