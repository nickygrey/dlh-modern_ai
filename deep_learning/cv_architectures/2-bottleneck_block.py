#!/usr/bin/env python3
"""Module to create a ResNet bottleneck residual block."""
from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Implement a ResNet bottleneck residual block.

    Args:
        x (tensor): Input tensor.
        filters (int): Number of filters for the 3x3 convolution.
        stride (int): Stride for the first convolution (downsampling).
        downsample (bool): Whether to apply a projection shortcut.
        name (str, optional): Prefix name for the block layers.

    Returns:
        tensor: Output tensor of the bottleneck residual block.
    """
    conv1_name = f"{name}_conv1" if name else None
    bn1_name = f"{name}_bn1" if name else None
    relu1_name = f"{name}_relu1" if name else None
    conv2_name = f"{name}_conv2" if name else None
    bn2_name = f"{name}_bn2" if name else None
    relu2_name = f"{name}_relu2" if name else None
    conv3_name = f"{name}_conv3" if name else None
    bn3_name = f"{name}_bn3" if name else None
    shortcut_conv_name = f"{name}_shortcut_conv" if name else None
    shortcut_bn_name = f"{name}_shortcut_bn" if name else None
    add_name = f"{name}_add" if name else None
    out_name = f"{name}_out" if name else None

    # 1x1 Convolution (Channel reduction)
    y = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=stride,
        padding='same',
        use_bias=False,
        name=conv1_name
    )(x)
    y = keras.layers.BatchNormalization(name=bn1_name)(y)
    y = keras.layers.ReLU(name=relu1_name)(y)

    # 3x3 Convolution (Bottleneck representation)
    y = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        use_bias=False,
        name=conv2_name
    )(y)
    y = keras.layers.BatchNormalization(name=bn2_name)(y)
    y = keras.layers.ReLU(name=relu2_name)(y)

    # 1x1 Convolution (Channel expansion by factor of 4)
    y = keras.layers.Conv2D(
        filters=filters * 4,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        name=conv3_name
    )(y)
    y = keras.layers.BatchNormalization(name=bn3_name)(y)

    # Residual (skip) connection
    if downsample:
        shortcut = keras.layers.Conv2D(
            filters=filters * 4,
            kernel_size=(1, 1),
            strides=stride,
            padding='same',
            use_bias=False,
            name=shortcut_conv_name
        )(x)
        shortcut = keras.layers.BatchNormalization(
            name=shortcut_bn_name
        )(shortcut)
    else:
        shortcut = x

    # Add shortcut and apply final activation
    y = keras.layers.Add(name=add_name)([y, shortcut])
    y = keras.layers.ReLU(name=out_name)(y)

    return y
