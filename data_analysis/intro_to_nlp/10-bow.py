#!/usr/bin/env python3
"""Bag-of-Words feature extraction module for NLP."""
import sklearn
import sklearn.feature_extraction.text

CountVectorizer = sklearn.feature_extraction.text.CountVectorizer


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """Build a Bag-of-Words feature matrix from a list of token lists.

    Args:
        corpus_tokens (list[list[str]]): List of token lists.
        max_features (int, optional): Maximum number of features.
            Defaults to 5000.
        ngram_range (tuple, optional): Range of n-gram sizes.
            Defaults to (1, 2).
        min_df (int or float, optional): Minimum document frequency.
            Defaults to 2.
        max_df (int or float, optional): Maximum document frequency.
            Defaults to 0.95.
        binary (bool, optional): Whether to use binary occurrence values.
            Defaults to False.

    Returns:
        tuple: (X, vectorizer) where X is the sparse feature matrix and
            vectorizer is the fitted CountVectorizer instance.
    """
    corpus_strings = [
        " ".join(doc) if isinstance(doc, list) else doc
        for doc in corpus_tokens
    ]

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary
    )

    X = vectorizer.fit_transform(corpus_strings)
    return X, vectorizer
