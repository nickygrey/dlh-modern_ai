#!/usr/bin/env python3
"""Module to create an EarlyStopping callback for Keras training."""
from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """Create and return a configured EarlyStopping callback.

    Args:
        patience (int): Number of epochs with no improvement after which
                        training will be stopped.
        monitor (str): Metric to monitor. Defaults to 'val_loss'.
        verbose (int): Verbosity mode. Defaults to 1.

    Returns:
        keras.callbacks.EarlyStopping: Configured EarlyStopping callback.
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        restore_best_weights=True
    )
