#!/usr/bin/env python3
"""Module to configure gradient descent variants in Keras."""
from tensorflow import keras


def train_with_gradient_descent_variant(
    variant, learning_rate, x_train, batch_size
):
    """Configure gradient descent optimizer and batch size based on variant.

    Args:
        variant (str): Variant type ('batch', 'stochastic', 'mini_batch').
        learning_rate (float): Learning rate for SGD optimizer.
        x_train (numpy.ndarray): Training dataset input data.
        batch_size (int): Custom batch size for mini_batch variant.

    Returns:
        tuple: (optimizer, bs) where optimizer is a Keras SGD instance
               and bs is the calculated batch size integer.
    """
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    if variant == 'batch':
        bs = x_train.shape[0]
    elif variant == 'stochastic':
        bs = 1
    elif variant == 'mini_batch':
        bs = batch_size
    else:
        bs = batch_size

    return optimizer, bs
