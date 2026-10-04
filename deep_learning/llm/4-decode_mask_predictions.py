#!/usr/bin/env python3
"""Module to decode masked token predictions using a RoBERTa tokenizer."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """Convert logits for all mask tokens into vocabulary tokens.

    Args:
        mask_logits_list (list): A list of logits tensors, one for each
            mask token in the sentence.
        tokenizer: A RoBERTa tokenizer instance used to decode token IDs
            into readable strings.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains all
            vocabulary tokens (as strings) corresponding to the logits at each
            mask position.
    """
    if not mask_logits_list:
        return []

    vocab_size = (
        mask_logits_list[0].shape[-1]
        if hasattr(mask_logits_list[0], 'shape')
        else len(tokenizer)
    )

    try:
        vocab_tokens = [
            tokenizer.decode([i]).strip() for i in range(vocab_size)
        ]
    except TypeError:
        vocab_tokens = [
            tokenizer.decode(i).strip() for i in range(vocab_size)
        ]

    return [vocab_tokens[:] for _ in mask_logits_list]
