#!/usr/bin/env python3
"""Module to tokenize and map text inputs of a dataset."""


def tokenize_and_map(dataset, tokenizer, max_length=512, truncation=True,
                     batched=True):
    """Tokenize the text inputs of a dataset using a pretrained tokenizer.

    Args:
        dataset: The dataset with 'train', 'validation', and 'test' splits.
        tokenizer: Pretrained tokenizer.
        max_length (int, optional): Maximum token length for truncation.
            Defaults to 512.
        truncation (bool, optional): Whether to truncate sequences longer
            than max_length. Defaults to True.
        batched (bool, optional): Whether to process the dataset in batches.
            Defaults to True.

    Returns:
        tuple: The tokenized train, validation, and test splits.
    """
    def tokenize(batch):
        """Tokenize a batch of text examples."""
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=truncation
        )

    tokenized_dataset = dataset.map(tokenize, batched=batched)
    return (
        tokenized_dataset["train"],
        tokenized_dataset["validation"],
        tokenized_dataset["test"]
    )
