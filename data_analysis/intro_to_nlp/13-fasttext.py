#!/usr/bin/env python3
"""Train FastText subword model and compute message representations."""
import gensim.models
import numpy as np


def fasttext_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """Train FastText model and return per-document embedding vectors.

    Args:
        corpus_tokens (list[list[str]]): List of token lists for training.
        vector_size (int, optional): Dimensionality of word embeddings.
            Defaults to 100.
        window (int, optional): Context window size. Defaults to 5.
        min_count (int, optional): Minimum word occurrence count.
            Defaults to 1.
        sg (int, optional): Training architecture: 0 for CBOW, 1 for
            Skip-gram. Defaults to 0.
        epochs (int, optional): Training iterations. Defaults to 10.
        workers (int, optional): Number of thread workers. Defaults to 4.

    Returns:
        tuple: (X, model) where X is an np.ndarray of shape (n_messages,
            vector_size) containing message vectors and model is the
            trained FastText model instance.
    """
    model = gensim.models.FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )

    embeddings = []
    for message in corpus_tokens:
        token_vectors = [model.wv[token] for token in message]
        if token_vectors:
            embeddings.append(np.mean(token_vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))

    X = np.array(embeddings, dtype=np.float64)
    return X, model
