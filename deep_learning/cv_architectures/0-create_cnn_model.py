#!/usr/bin/env python3
"""Module to create a Convolutional Neural Network (CNN) model."""
from tensorflow import keras


def create_cnn_model(
    input_shape, filters, kernel_sizes, activations, pooling_type='max'
):
    """Create and return a Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input data (excluding batch size).
        filters (list): Number of filters in each convolutional layer.
        kernel_sizes (list): Size of kernels for each convolutional layer.
        activations (list): Activation functions for each convolutional layer.
        pooling_type (str): Type of pooling ('max' or 'avg').
                            Defaults to 'max'.

    Returns:
        keras.Model: Configured Keras Sequential CNN model.
    """
    model = keras.Sequential()

    for i in range(len(filters)):
        if i == 0:
            model.add(keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                activation=activations[i],
                input_shape=input_shape
            ))
        else:
            model.add(keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                activation=activations[i]
            ))

        if pooling_type == 'avg':
            model.add(keras.layers.AveragePooling2D())
        else:
            model.add(keras.layers.MaxPooling2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
