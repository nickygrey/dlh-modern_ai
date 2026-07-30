#!/usr/bin/env python3
"""Module to train and evaluate decision trees with post-pruning."""
from sklearn import tree
train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas,
                             random_state, min_samples_leaf,
                             min_samples_split):
    """Train and evaluate decision trees over a range of ccp_alpha values.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        ccp_alphas: Array of pruning alpha values.
        random_state (int): Seed for reproducibility.
        min_samples_leaf (int): Min samples per leaf.
        min_samples_split (int): Min samples to split an internal node.

    Returns:
        tuple: (clfs, train_scores, test_scores)
    """
    clfs = []
    train_scores = []
    test_scores = []

    for alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            ccp_alpha=alpha,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )
        train_tree(clf, X_train, y_train)

        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    return clfs, train_scores, test_scores
