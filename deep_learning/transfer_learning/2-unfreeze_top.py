#!/usr/bin/env python3
"""Module to unfreeze top layers of a base model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreeze the last n_layers of base model in transfer learning.

    Args:
        model: Model containing base model or base model itself.
        n_layers (int): Number of last layers in base model to unfreeze.

    Returns:
        None
    """
    base_model = None
    if hasattr(model, "layers"):
        if (len(model.layers) > 0 and hasattr(model.layers[0], "layers")
                and len(model.layers[0].layers) > 0):
            base_model = model.layers[0]
        else:
            for layer in model.layers:
                if hasattr(layer, "layers") and len(layer.layers) > 0:
                    base_model = layer
                    break

    if base_model is None:
        base_model = model

    num_layers = len(base_model.layers)
    split_idx = max(0, num_layers - n_layers)

    for layer in base_model.layers[:split_idx]:
        layer.trainable = False
    for layer in base_model.layers[split_idx:]:
        layer.trainable = True

    return None
