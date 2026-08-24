#!/usr/bin/env python3
"""Module to execute hyperparameter search and return best hyperparameters."""


def search_and_return_best_model(
    tuner, x_train, y_train, epochs, validation_split, verbose=0
):
    """Search for the best hyperparameters using a configured Keras Tuner.

    Args:
        tuner: Configured Keras Tuner instance.
        x_train (numpy.ndarray): Training input features.
        y_train (numpy.ndarray): Training target labels.
        epochs (int): Number of epochs per trial.
        validation_split (float): Fraction of training data for validation.
        verbose (int): Verbosity mode (0 = silent, 1 = search bar).

    Returns:
        HyperParameters: The best hyperparameter configuration found.
    """
    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
    )

    return tuner.get_best_hyperparameters(num_trials=1)[0]
