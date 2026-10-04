#!/usr/bin/env python3
"""Module to load the Emotion dataset for sentiment analysis."""
import datasets


def load_emotion_dataset():
    """Load the dair-ai/emotion dataset from Hugging Face.

    Returns:
        datasets.DatasetDict: Dataset object containing train, validation,
            and test splits.
    """
    try:
        return datasets.load_dataset("dair-ai/emotion")
    except Exception:
        return datasets.load_dataset("emotion")
