#!/usr/bin/env python3
"""Module to generate n-grams from a list of tokens."""
import nltk


def generate_ngrams(tokens, n=2):
    """Generate n-grams from a token list.

    Args:
        tokens (list[str]): List of tokens used to generate n-grams.
        n (int, optional): Size of each n-gram. Defaults to 2.

    Returns:
        list[str]: List of n-grams joined with underscores.
    """
    if not isinstance(tokens, list) or len(tokens) < n or n < 1:
        return []

    return ["_".join(gram) for gram in nltk.ngrams(tokens, n)]
