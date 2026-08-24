#!/usr/bin/env python3
"""Module to build a tunable Keras model using Keras Tuner."""
from tensorflow import keras


def build_model(hp):
    """Build a Keras model with tunable hyperparameters.

    Args:
        hp (keras_tuner.HyperParameters): HyperParameters instance defining
                                          the search space.

    Returns:
        keras.Model: Compiled Keras Sequential model.
    """
    num_layers = hp.Int('num_layers', min_value=1, max_value=2)
    units = hp.Int('units', min_value=4, max_value=12, step=4)
    activation = hp.Choice('activation', values=['relu', 'sigmoid'])
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))

    for _ in range(num_layers):
        model.add(keras.layers.Dense(
            units=units,
            activation=activation
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
