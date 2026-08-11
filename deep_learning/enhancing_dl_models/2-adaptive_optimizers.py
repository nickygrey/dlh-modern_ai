#!/usr/bin/env python3
"""Module to return configured Keras optimizers."""
from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """Configure and return a Keras optimizer based on specified parameters.

    Args:
        name (str): Name of optimizer ('sgd', 'adam', or 'rmsprop').
        learning_rate (float): Learning rate.
        momentum (float): Momentum factor for SGD.
        beta_1 (float): Exponential decay rate for 1st moment (Adam).
        beta_2 (float): Exponential decay rate for 2nd moment (Adam).
        rho (float): Decay factor for RMSprop.

    Returns:
        keras.optimizers.Optimizer: Configured Keras optimizer instance.
    """
    if name == 'sgd':
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )
    if name == 'adam':
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    if name == 'rmsprop':
        return keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
        )
    return None
