#!/usr/bin/env python3
"""Module to compile a Keras model for training."""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """Configure a Keras model for training.

    Args:
        model: Keras model instance to compile.
        learning_rate (float): Learning rate for SGD optimizer.

    Returns:
        None
    """
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
