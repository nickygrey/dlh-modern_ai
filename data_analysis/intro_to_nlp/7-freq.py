#!/usr/bin/env python3
"""Module to plot top n word frequency distributions."""
import matplotlib.pyplot as plt
import nltk


def plot_top_n_frequencies(corpus_tokens, n=20):
    """Plot the top n most frequent tokens in a preprocessed corpus.

    Args:
        corpus_tokens (list[list[str]]): Corpus represented as a list of
            token lists.
        n (int, optional): Number of top frequent tokens to display.
            Defaults to 20.

    Returns:
        nltk.FreqDist: The full frequency distribution object.
    """
    flat_tokens = [tok for doc in corpus_tokens for tok in doc]
    fd = nltk.FreqDist(flat_tokens)

    top_tokens = fd.most_common(n)
    words = [item[0] for item in top_tokens]
    counts = [item[1] for item in top_tokens]

    plt.figure(figsize=(12, 5))
    plt.bar(words, counts)
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fd
