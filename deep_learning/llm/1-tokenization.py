#!/usr/bin/env python3
"""Module to tokenize text using a pre-trained RoBERTa tokenizer."""
import transformers


def tokenize_text(model_name, sentence, padding=True):
    """Tokenize text using a pre-trained RoBERTa tokenizer.

    Args:
        model_name (str): Name of the pre-trained model to load.
        sentence (str or list[str]): Text or sequence of texts to tokenize.
        padding (bool, optional): Whether to pad token sequences.
            Defaults to True.

    Returns:
        tuple: (tokenizer, inputs) where:
            tokenizer (transformers.RobertaTokenizer): Tokenizer instance.
            inputs (transformers.tokenization_utils_base.BatchEncoding):
                Tokenized representations as PyTorch tensors.
    """
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sentence, padding=padding, return_tensors="pt")
    return tokenizer, inputs
