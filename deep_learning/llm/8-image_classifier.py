#!/usr/bin/env python3
"""Module to create an image classification pipeline using Transformers."""
import transformers


def image_classifier(model):
    """Create an image classification pipeline using a pre-trained model.

    Args:
        model (str): Name of the pre-trained model to use.

    Returns:
        transformers.pipelines.Pipeline: Hugging Face image classification
            pipeline object.
    """
    return transformers.pipeline(
        task="image-classification",
        model=model
    )
