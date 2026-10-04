#!/usr/bin/env python3
"""Module to initialize an inference pipeline for text classification."""
import transformers


def inference_mode(model_path, top_k):
    """Initialize a text classification pipeline using a fine-tuned model.

    Args:
        model_path (str): Path to the saved model and tokenizer.
        top_k (int): Number of top predictions to return per input.

    Returns:
        transformers.pipelines.Pipeline: Pipeline object ready to classify
            new texts.
    """
    return transformers.pipeline(
        task="text-classification",
        model=model_path,
        top_k=top_k
    )
