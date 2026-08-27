#!/usr/bin/env python3
"""Module to compile and train a Convolutional Neural Network (CNN)."""
import numpy as np
from tensorflow import keras


def compile_and_train_cnn(
    model,
    epochs,
    batch_size,
    x_train=None,
    y_train=None,
    x_val=None,
    y_val=None,
    optimizer_name='adam',
    optimizer_params=None,
    **kwargs
):
    """Compile and train a CNN model on Fashion-MNIST dataset.

    Args:
        model (keras.Model): CNN model to train.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        x_train (numpy.ndarray, optional): Training features.
        y_train (numpy.ndarray, optional): Training labels.
        x_val (numpy.ndarray, optional): Validation features.
        y_val (numpy.ndarray, optional): Validation labels.
        optimizer_name (str): Name of optimizer ('adam', 'sgd', 'rmsprop').
        optimizer_params (dict, optional): Keyword arguments for optimizer.

    Returns:
        tuple: (trained model, training history object).
    """
    if isinstance(x_train, str):
        optimizer_name = x_train
        optimizer_params = y_train if isinstance(y_train, dict) else None
        x_train = None
        y_train = None

    if x_train is None or y_train is None:
        (x_tr, y_tr), _ = keras.datasets.fashion_mnist.load_data()
        x_tr = x_tr.astype("float32") / 255.0
        x_tr = x_tr[..., np.newaxis]
        y_tr = keras.utils.to_categorical(y_tr, 10)
        x_train, x_val = x_tr[:50000], x_tr[50000:]
        y_train, y_val = y_tr[:50000], y_tr[50000:]

    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if y_train.ndim == 1:
        y_train = keras.utils.to_categorical(y_train, 10)

    if x_val is not None and x_val.ndim == 3:
        x_val = x_val[..., np.newaxis]
    if y_val is not None and y_val.ndim == 1:
        y_val = keras.utils.to_categorical(y_val, 10)

    params = optimizer_params if optimizer_params is not None else {}
    opt_map = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop,
        'adagrad': keras.optimizers.Adagrad,
        'adadelta': keras.optimizers.Adadelta,
        'adamax': keras.optimizers.Adamax,
        'nadam': keras.optimizers.Nadam,
    }

    if isinstance(optimizer_name, str):
        opt_cls = opt_map.get(optimizer_name.lower())
        if opt_cls:
            optimizer = opt_cls(**params)
        else:
            optimizer = keras.optimizers.get(optimizer_name)
    else:
        optimizer = optimizer_name

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    val_data = (
        (x_val, y_val)
        if x_val is not None and y_val is not None
        else None
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data,
        verbose=1
    )

    return model, history
