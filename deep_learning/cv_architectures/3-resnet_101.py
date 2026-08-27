#!/usr/bin/env python3
"""Module to build ResNet-101 architecture."""
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

    y = keras.layers.Conv2D(
        filters=filters * 4,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        name=conv3_name
    )(y)
    y = keras.layers.BatchNormalization(name=bn3_name)(y)

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

    y = keras.layers.Add(name=add_name)([y, shortcut])
    y = keras.layers.ReLU(name=out_name)(y)

    return y


def make_layer(x, blocks, filters, stride=1, name=None):
    """Create a ResNet stage containing multiple bottleneck blocks.

    Args:
        x (tensor): Input tensor.
        blocks (int): Number of bottleneck blocks in the stage.
        filters (int): Number of base filters.
        stride (int): Stride for the first block. Defaults to 1.
        name (str, optional): Name prefix for layers.

    Returns:
        tensor: Output tensor after applying the stage.
    """
    x = bottleneck_block(
        x, filters, stride=stride, downsample=True,
        name=f'{name}_block1'
    )
    for i in range(1, blocks):
        x = bottleneck_block(
            x, filters, stride=1, downsample=False,
            name=f'{name}_block{i + 1}'
        )
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """Build the ResNet-101 model architecture.

    Args:
        input_shape (tuple): Shape of the input tensor.
                             Defaults to (224, 224, 3).
        num_classes (int): Number of classification classes.
                           Defaults to 1000.

    Returns:
        keras.Model: Configured ResNet-101 Keras model.
    """
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        name='maxpool'
    )(x)

    x = make_layer(x, blocks=3, filters=64, stride=1, name='layer1')
    x = make_layer(x, blocks=4, filters=128, stride=2, name='layer2')
    x = make_layer(x, blocks=23, filters=256, stride=2, name='layer3')
    x = make_layer(x, blocks=3, filters=512, stride=2, name='layer4')

    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        name='fc'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet101')
    return model
