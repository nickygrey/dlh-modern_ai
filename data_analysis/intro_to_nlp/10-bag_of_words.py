#!/usr/bin/env python3
"""Bag-of-Words feature matrix generation module."""
import sklearn


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """Build a Bag-of-Words feature matrix from a tokenized corpus.

    Args:
        corpus_tokens (list[list[str]]): Corpus represented as a list of
            token lists.
        max_features (int, optional): Maximum vocabulary size to extract.
            Defaults to 5000.
        ngram_range (tuple, optional): Lower and upper boundary of n-values.
            Defaults to (1, 2).
        min_df (int or float, optional): Minimum term document frequency.
            Defaults to 2.
        max_df (int or float, optional): Maximum term document frequency ratio.
            Defaults to 0.95.
        binary (bool, optional): Whether presence indicators are used instead
            of counts. Defaults to False.

    Returns:
        tuple: (X, vectorizer) where X is the sparse document-term matrix
            and vectorizer is the fitted CountVectorizer instance.
    """
    corpus_text = [
        " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        for tokens in corpus_tokens
    ]

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    X = vectorizer.fit_transform(corpus_text)
    return X, vectorizer
