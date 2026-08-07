#!/usr/bin/env python3
"""Module to build a Keras Sequential neural network model."""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Build a shallow neural network model using Keras Sequential API.

    Args:
        input_dim (int): Number of input features.
        neurons_h (int): Number of neurons in the hidden layer.

    Returns:
        keras.Model: Constructed Keras Sequential model.
    """
    model = keras.Sequential([
        keras.layers.Dense(
            neurons_h,
            activation='sigmoid',
            input_shape=(input_dim,)
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model
