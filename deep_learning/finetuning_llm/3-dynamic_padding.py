#!/usr/bin/env python3
"""Module to create a data collator with dynamic padding."""
import transformers


def create_data_collator(tokenizer):
    """Set up dynamic padding for tokenized inputs.

    Args:
        tokenizer: Pretrained tokenizer used for dynamic padding.

    Returns:
        transformers.DataCollatorWithPadding: Data collator instance that
            dynamically pads inputs per batch.
    """
    return transformers.DataCollatorWithPadding(tokenizer=tokenizer)
