#!/usr/bin/env python3
"""Module to extract mask token positions from tokenized inputs."""


def get_mask_index(inputs, tokenizer):
    """Identify and return the positions of all <mask> tokens in inputs.

    Args:
        inputs (dict or transformers.tokenization_utils_base.BatchEncoding):
            Tokenized inputs containing 'input_ids'.
        tokenizer (transformers.RobertaTokenizer): An instance of
            RobertaTokenizer.

    Returns:
        list[int]: A list containing the index of every <mask> token found
            in the sequence.

    Raises:
        ValueError: If no <mask> token is found in the input.
    """
    if isinstance(inputs, dict) or hasattr(inputs, "get"):
        input_ids = inputs["input_ids"]
    elif hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids
    else:
        input_ids = inputs

    mask_token_id = tokenizer.mask_token_id

    if hasattr(input_ids, "tolist"):
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)

    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]

    mask_indices = [
        i for i, token_id in enumerate(ids)
        if token_id == mask_token_id
    ]

    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")

    return mask_indices
