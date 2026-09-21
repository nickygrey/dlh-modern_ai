#!/usr/bin/env python3
"""Stopwords removal module for SMS messages."""
import nltk


def remove_stopwords(tokens, language="english", extra_words=None,
                     keep_words=None):
    """Remove stopwords from a token list.

    Args:
        tokens (list[str]): List of tokens to filter.
        language (str): NLTK stopword language to load. Defaults to "english".
        extra_words (set[str] | None): Additional words to add to stopwords.
        keep_words (set[str] | None): Words to exclude from stopwords.

    Returns:
        list[str]: Filtered list of tokens.
    """
    if not isinstance(tokens, list):
        return []

    stopwords_set = set(nltk.corpus.stopwords.words(language))

    if extra_words:
        stopwords_set.update(extra_words)
        stopwords_set.update(
            w.lower() for w in extra_words if isinstance(w, str)
        )

    if keep_words:
        stopwords_set.difference_update(keep_words)
        stopwords_set.difference_update(
            w.lower() for w in keep_words if isinstance(w, str)
        )

    return [
        t for t in tokens
        if t.lower() not in stopwords_set and t not in stopwords_set
    ]
