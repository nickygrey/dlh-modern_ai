#!/usr/bin/env python3
"""Module to log Keras model training metrics to TensorBoard."""
import datetime
from tensorflow import keras


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """Log training metrics and weight histograms to TensorBoard.

    Args:
        log_dir (str): Base directory where logs should be saved.
        model: Keras model instance.
        X: Input features data.
        Y: Target labels data.
        epochs (int): Number of training epochs.
        verbose (int): Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        None
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"{log_dir}/{timestamp}"

    tb_callback = keras.callbacks.TensorBoard(
        log_dir=path,
        histogram_freq=1
    )

    model.fit(
        X,
        Y,
        epochs=epochs,
        verbose=verbose,
        callbacks=[tb_callback]
    )
