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
        keras.Model: Compiled Keras model with configured layers.
    """
    if activation in ['sigmoid', 'tanh']:
        initializer = keras.initializers.GlorotUniform()
        act_func = activation
    elif activation == 'relu':
        initializer = keras.initializers.HeNormal()
        act_func = activation
    elif activation == 'leaky_relu':
        initializer = keras.initializers.HeNormal()
        act_func = keras.layers.LeakyReLU()
    else:
        raise ValueError(
            "activation must be 'sigmoid', 'tanh', 'relu', or 'leaky_relu'"
        )

    inputs = keras.Input(shape=(input_dim,))
    hidden = keras.layers.Dense(
        units=hidden_units,
        activation=act_func,
        kernel_initializer=initializer
    )(inputs)
    outputs = keras.layers.Dense(units=10, activation='softmax')(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
