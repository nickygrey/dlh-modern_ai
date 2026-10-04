#!/usr/bin/env python3
"""Module to load a DistilBERT tokenizer and classification model."""
import transformers


def load_distilbert(model_name, num_classes, id2label, label2id):
    """Load a tokenizer and sequence classification model.

    Args:
        model_name (str): Name of the pre-trained DistilBERT to load.
        num_classes (int): Total number of output classes.
        id2label (dict): Mapping numeric label IDs to human-readable names.
        label2id (dict): Mapping label names to numeric IDs.

    Returns:
        tuple: (tokenizer, model) where:
            tokenizer: The tokenizer for preprocessing text inputs.
            model: The sequence classification model configured with the
                correct number of classes and label mappings.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )
    return tokenizer, model
