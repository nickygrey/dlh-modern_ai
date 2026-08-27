#!/usr/bin/env python3
"""Module to build MobileNetV1 backbone architecture."""
from tensorflow import keras


def depthwise_separable_conv(x, filters, stride=1):
    """Implement a Depthwise Separable Convolution block.

    Args:
        x (tensor): Input tensor.
        filters (int): Number of filters for pointwise convolution.
        stride (int): Stride for depthwise convolution. Defaults to 1.

    Returns:
        tensor: Output tensor after depthwise separable convolution.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(x)
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


def mobilenet_backbone(inputs):
    """Build the feature extraction backbone of MobileNetV1.

    Args:
        inputs (tensor): Input tensor to the network.

    Returns:
        tensor: Output tensor of the MobileNet backbone.
    """
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    block_configs = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1)
    ]

    for filters, stride in block_configs:
        x = depthwise_separable_conv(x, filters=filters, stride=stride)

    return x
