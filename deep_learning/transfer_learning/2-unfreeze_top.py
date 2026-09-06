#!/usr/bin/env python3
"""Module to unfreeze top layers of a base model."""
from tensorflow import keras


def unfreeze_top_layers(model, n_layers):
    """Unfreeze the last n_layers of base model in transfer learning pipeline.

    Args:
        model (keras.Model): Model containing base model or base model itself.
        n_layers (int): Number of last layers in base model to unfreeze.

    Returns:
        None
    """
    base_model = model
    if hasattr(model, "layers"): 
        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                base_model = layer
                break

    model.trainable = True
    base_model.trainable = True

    num_layers = len(base_model.layers)
    split_idx = max(0, num_layers - n_layers)

    for layer in base_model.layers[:split_idx]:
        layer.trainable = False
    for layer in base_model.layers[split_idx:]:
        layer.trainable = True

    return None
