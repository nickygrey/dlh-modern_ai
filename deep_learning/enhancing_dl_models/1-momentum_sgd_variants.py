#!/usr/bin/env python3
"""Module to configure momentum-based SGD optimizer variants."""
from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """Configure and return an SGD-based optimizer variant.

    Args:
        name (str): Optimizer variant ('SGD', 'SGD+Momentum', or
                    'SGD+Momentum+Nesterov').
        lr (float): The learning rate.
        momentum (float): The momentum factor.
        nesterov (bool): Whether to apply Nesterov acceleration.

    Returns:
        keras.optimizers.SGD: Configured Keras SGD optimizer instance.
    """
    if name == 'SGD':
        return keras.optimizers.SGD(learning_rate=lr)
    elif name == 'SGD+Momentum':
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    elif name == 'SGD+Momentum+Nesterov':
        return keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=nesterov
        )
    return keras.optimizers.SGD(learning_rate=lr)
