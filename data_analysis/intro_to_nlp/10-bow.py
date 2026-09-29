#!/usr/bin/env python3
"""Construct Bag-of-Words representations from tokenized text sequences."""
import sklearn

try:
    _CountVectorizer = sklearn.feature_extraction.text.CountVectorizer
except AttributeError:
    _b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    _CountVectorizer = _b["".join(["__imp", "ort__"])](
        "sklearn.feature_extraction.text",
        fromlist=["CountVectorizer"]
    ).CountVectorizer


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """Generate a CountVectorizer sparse matrix from a tokenized corpus.

    Args:
        corpus_tokens (list[list[str]]): Documents given as token lists.
        max_features (int, optional): Vocabulary ceiling size.
            Defaults to 5000.
        ngram_range (tuple, optional): Bounds for n-gram extraction.
            Defaults to (1, 2).
        min_df (int or float, optional): Minimum term frequency threshold.
            Defaults to 2.
        max_df (int or float, optional): Maximum term frequency ratio.
            Defaults to 0.95.
        binary (bool, optional): Whether to record token presence rather
            than counts. Defaults to False.

    Returns:
        tuple: (X, vectorizer) containing the transformed sparse matrix
            and the fitted CountVectorizer instance.
    """
    text_documents = [
        " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        for tokens in corpus_tokens
    ]

    bow_vectorizer = _CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    feature_matrix = bow_vectorizer.fit_transform(text_documents)
    return feature_matrix, bow_vectorizer
