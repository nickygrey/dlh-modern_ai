#!/usr/bin/env python3
"""Module to initialize a Keras Tuner."""
import keras_tuner as kt


def initiate_tuner(
    tuner_type, build_model, seed,
    hyperband_iterations, max_trials,
    objective="val_accuracy"
):
    """Initialize a Keras Tuner for hyperparameter tuning.

    Args:
        tuner_type (str): Tuner algorithm type ('Hyperband',
                          'RandomSearch', or 'BayesianOptimization').
        build_model (callable): Function that returns a compiled model.
        seed (int): Random seed for reproducibility.
        hyperband_iterations (int): Number of iterations for Hyperband.
        max_trials (int): Max number of trials for search tuners.
        objective (str): Metric to optimize. Defaults to 'val_accuracy'.

    Returns:
        kt.Tuner: Configured Keras Tuner instance.
    """
    if tuner_type == 'Hyperband':
        tuner = kt.Hyperband(
            build_model,
            objective=objective,
            max_epochs=10,
            factor=3,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    elif tuner_type == 'RandomSearch':
        tuner = kt.RandomSearch(
            build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    elif tuner_type == 'BayesianOptimization':
        tuner = kt.BayesianOptimization(
            build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    else:
        raise ValueError(f"Unknown tuner type: {tuner_type}")

    return tuner
