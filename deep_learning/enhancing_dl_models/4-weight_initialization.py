#!/usr/bin/env python3
"""Module to build Keras model with activation-based weight initialization."""
from tensorflow import keras


def build_model_initializer_by_activation(
    input_dim, hidden_units, activation
):
    """Build a model with appropriate weight initializers for activations.

    Args:
        input_dim (int): Number of input features.
        hidden_units (int): Number of neurons in the hidden layer.
        activation (str): Activation function ('sigmoid', 'tanh',
                          'relu', or 'leaky_relu').

    Returns:
        keras.Model: Keras model with configured hidden and output layers.
    """
    if activation in ['sigmoid', 'tanh']:
        initializer = 'glorot_uniform'
    elif activation in ['relu', 'leaky_relu']:
        initializer = 'he_normal'
    else:
        initializer = 'glorot_uniform'

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(
        hidden_units,
        activation=activation,
        kernel_initializer=initializer
    ))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
