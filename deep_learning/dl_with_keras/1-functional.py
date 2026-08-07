#!/usr/bin/env python3
"""Module to build a Keras model using the Functional API."""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Build a shallow neural network using Keras Functional API.

    Args:
        input_dim (int): Number of input features.
        neurons_h (int): Number of neurons in the hidden layer.

    Returns:
        keras.Model: Constructed Keras Functional model.
    """
    inputs = keras.Input(shape=(input_dim,))
    hidden = keras.layers.Dense(neurons_h, activation='sigmoid')(inputs)
    outputs = keras.layers.Dense(10, activation='softmax')(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
