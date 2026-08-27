#!/usr/bin/env python3
"""Module to create a Depthwise Separable Convolution block."""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Implement a Depthwise Separable Convolution block for MobileNetV1.

    Args:
        X (tensor): Input tensor.
        filters (int): Number of output channels for pointwise convolution.
        stride (int): Stride applied to the depthwise convolution.
                      Defaults to 1.

    Returns:
        tensor: Output tensor of depthwise separable convolution block.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding="same",
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
