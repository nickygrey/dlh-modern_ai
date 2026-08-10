#!/usr/bin/env python3
"""Module to build a deep neural network model using Keras Sequential API."""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """Build a deep neural network for multi-class classification.

    Args:
        input_dim (int): Number of input features.
        hidden_layers (list): List of integers representing the number of
                              neurons in each hidden layer.

    Returns:
        keras.Model: Constructed Keras Sequential model.
    """
    model = keras.Sequential()

    for i, nodes in enumerate(hidden_layers):
        if i == 0:
            model.add(keras.layers.Dense(
                nodes,
                activation='relu',
                input_shape=(input_dim,)
            ))
        else:
            model.add(keras.layers.Dense(nodes, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
