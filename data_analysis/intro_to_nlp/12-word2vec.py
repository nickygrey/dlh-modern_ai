#!/usr/bin/env python3
"""Module to train Word2Vec and generate message embeddings."""
import gensim.models
import numpy as np


def word2vec_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=2, sg=0, epochs=10, workers=4):
    """Train Word2Vec model and generate per-message embeddings.

    Args:
        corpus_tokens (list[list[str]]): Corpus represented as a list of
            token lists.
        vector_size (int, optional): Dimensionality of word vectors.
            Defaults to 100.
        window (int, optional): Maximum distance between current and
            predicted word. Defaults to 5.
        min_count (int, optional): Minimum frequency count of words.
            Defaults to 2.
        sg (int, optional): Training algorithm: 0 for CBOW, 1 for Skip-gram.
            Defaults to 0.
        epochs (int, optional): Number of training iterations.
            Defaults to 10.
        workers (int, optional): Number of worker threads. Defaults to 4.

    Returns:
        tuple: (X, model) where:
            X (numpy.ndarray): Message embeddings matrix of shape
                (n_messages, vector_size).
            model (gensim.models.Word2Vec): Trained Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )

    X = np.zeros((len(corpus_tokens), vector_size))
    for i, message in enumerate(corpus_tokens):
        in_vocab_vectors = [
            model.wv[token] for token in message
            if token in model.wv
        ]
        if in_vocab_vectors:
            X[i] = np.mean(in_vocab_vectors, axis=0)

    return X, model
