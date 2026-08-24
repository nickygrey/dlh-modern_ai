#!/usr/bin/env python3
"""Module to build Keras model with dropout regularization."""
from tensorflow import keras


def build_model_with_dropout(
    input_dim, hidden_units, n_layers,
    dropout_rate_input, dropout_rate_hidden
):
    """Build a Keras model with dropout regularization.

    Args:
        input_dim (int): Number of input features.
        hidden_units (int): Number of neurons in each hidden layer.
        n_layers (int): Number of hidden layers.
        dropout_rate_input (float): Dropout rate after the input layer.
        dropout_rate_hidden (float): Dropout rate after each hidden layer.

    Returns:
        keras.Model: Compiled Keras model instance.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if not (0 <= dropout_rate_input <= 1):
        raise ValueError("dropout_rate_input must be between 0 and 1")
    if not (0 <= dropout_rate_hidden <= 1):
        raise ValueError("dropout_rate_hidden must be between 0 and 1")

    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dropout(rate=dropout_rate_input)(inputs)

    for _ in range(n_layers):
        x = keras.layers.Dense(units=hidden_units, activation='relu')(x)
        x = keras.layers.Dropout(rate=dropout_rate_hidden)(x)

    outputs = keras.layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
