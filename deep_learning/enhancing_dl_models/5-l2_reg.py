#!/usr/bin/env python3
"""Module to build Keras model with L2 regularization on hidden layers."""
from tensorflow import keras


def build_model_with_L2_regularization(
    input_dim, hidden_units, n_layers, lambda_l2
):
    """Build a Keras model with L2 regularization on hidden layers.

    Args:
        input_dim (int): Number of input features.
        hidden_units (int): Number of neurons in each hidden layer.
        n_layers (int): Number of hidden layers.
        lambda_l2 (float): L2 regularization factor.

    Returns:
        keras.Model: Compiled Keras model instance.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be non-negative")

    l2_regularizer = keras.regularizers.l2(lambda_l2)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for _ in range(n_layers):
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            kernel_regularizer=l2_regularizer
        )(x)

    outputs = keras.layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
