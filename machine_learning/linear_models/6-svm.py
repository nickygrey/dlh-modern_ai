#!/usr/bin/env python3
"""Module to instantiate Support Vector Machine (SVM) classifiers."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Create and return an untrained Support Vector Classifier (SVC).

    Args:
        name (str): The kernel type ('linear', 'poly', or 'rbf').
        random_state (int): Seed for reproducibility.

    Returns:
        svm.SVC: An untrained Support Vector Classifier instance.
    """
    return svm.SVC(kernel=name, random_state=random_state)
