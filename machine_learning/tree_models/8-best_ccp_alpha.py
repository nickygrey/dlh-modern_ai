#!/usr/bin/env python3
"""Module to select the best ccp_alpha for decision tree pruning."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best ccp_alpha and corresponding classifier instance.

    Args:
        clfs (list): List of trained DecisionTreeClassifier instances.
        train_scores (list): List of training accuracy scores.
        test_scores (list): List of test accuracy scores.
        ccp_alphas (list or ndarray): Array of ccp_alpha values.

    Returns:
        tuple: (best_alpha, best_clf)
    """
    best_idx = 0
    best_key = None

    for i in range(len(clfs)):
        test_acc = test_scores[i]
        diff = abs(train_scores[i] - test_scores[i])
        alpha = ccp_alphas[i]

        # Candidate key to maximize: (test_accuracy, -difference, alpha)
        key = (test_acc, -diff, alpha)

        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    return ccp_alphas[best_idx], clfs[best_idx]
