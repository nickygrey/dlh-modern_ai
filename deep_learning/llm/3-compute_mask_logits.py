#!/usr/bin/env python3
"""Module to compute mask logits using a pre-trained RoBERTa model."""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Compute raw logits for all <mask> tokens in a tokenized sentence.

    Args:
        model: A RobertaForMaskedLM model loaded with pre-trained weights.
        inputs: Tokenized inputs.
        mask_indices (list[int]): Positions of all <mask> tokens in the
            input sequence.

    Returns:
        list[torch.Tensor]: A list containing logits tensors for each
            <mask> token, representing raw predictions at that position.
    """
    with torch.no_grad():
        if isinstance(inputs, dict) or hasattr(inputs, "keys"):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)

        logits = outputs.logits
        if logits.dim() == 3:
            seq_logits = logits[0]
        else:
            seq_logits = logits

        return [seq_logits[idx] for idx in mask_indices]
