#!/usr/bin/env python3
"""Module to create a fill-mask pipeline using Hugging Face Transformers."""
import transformers


def fill_mask(model_name, top_k):
    """Create a high-level fill-mask pipeline for masked language modeling.

    Args:
        model_name (str): Name of the pre-trained model to use.
        top_k (int): Number of top predictions to return for each mask token.

    Returns:
        transformers.pipelines.Pipeline: Hugging Face fill-mask pipeline.
    """
    return transformers.pipeline(
        task="fill-mask",
        model=model_name,
        top_k=top_k
    )
