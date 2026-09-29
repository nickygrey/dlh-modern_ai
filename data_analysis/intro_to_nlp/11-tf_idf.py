#!/usr/bin/env python3
"""TF-IDF feature matrix generation module."""
import sklearn


def tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2),
           min_df=2, max_df=0.95, norm='l2'):
    """Build a TF-IDF feature matrix from a list of token lists.

    Args:
        corpus_tokens (list[list[str]]): Corpus represented as a list of
            token lists.
        max_features (int, optional): Maximum vocabulary size to extract.
            Defaults to 5000.
        ngram_range (tuple, optional): Range of n-grams (min_n, max_n).
            Defaults to (1, 2).
        min_df (int or float, optional): Minimum document frequency.
            Defaults to 2.
        max_df (int or float, optional): Maximum document frequency ratio.
            Defaults to 0.95.
        norm (str, optional): Normalization term ('l1' or 'l2').
            Defaults to 'l2'.

    Returns:
        tuple: (X, vectorizer) where:
            X: Sparse TF-IDF feature matrix of shape (n_samples, n_features).
            vectorizer: The fitted TfidfVectorizer object.
    """
    corpus_text = [' '.join(tokens) for tokens in corpus_tokens]

    try:
        tfidf_cls = sklearn.feature_extraction.text.TfidfVectorizer
    except AttributeError:
        __import__('sklearn.feature_extraction.text')
        tfidf_cls = sklearn.feature_extraction.text.TfidfVectorizer

    vectorizer = tfidf_cls(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    X = vectorizer.fit_transform(corpus_text)
    return X, vectorizer
