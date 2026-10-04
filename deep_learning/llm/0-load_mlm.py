#!/usr/bin/env python3
"""Module to load a RoBERTa model for Masked Language Modeling."""
import transformers


def load_mlm(model_name):
    """Load a pre-trained RoBERTa model ready for Masked Language Modeling.

    Args:
        model_name (str): Name of the pre-trained model to load.

    Returns:
        transformers.RobertaForMaskedLM: Model instance ready for inference.
    """
    return transformers.RobertaForMaskedLM.from_pretrained(model_name)
