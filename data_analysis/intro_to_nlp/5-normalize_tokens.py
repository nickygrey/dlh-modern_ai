#!/usr/bin/env python3
"""Module to normalize tokens via lemmatization or stemming."""
import nltk
import re

_PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')
PLACEHOLDER_RE = _PLACEHOLDER_RE
PLACEHOLDERRE = _PLACEHOLDER_RE


def get_pos(tag):
    """Map Penn Treebank POS tag to WordNet POS tag.

    Args:
        tag (str): Penn Treebank POS tag string.

    Returns:
        str: Corresponding WordNet POS tag.
    """
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    if tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    if tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    if tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN


def normalize_tokens(tokens, method="lemmatize"):
    """Normalize tokens using either lemmatization or stemming.

    Args:
        tokens (list[str]): List of token strings to normalize.
        method (str, optional): Normalization method ('lemmatize' or 'stem').
            Defaults to 'lemmatize'.

    Returns:
        list[str]: List of normalized tokens.

    Raises:
        ValueError: If method is not 'lemmatize' or 'stem'.
    """
    if method not in ["lemmatize", "stem"]:
        raise ValueError("method must be 'lemmatize' or 'stem'")

    if not isinstance(tokens, list):
        return []

    if method == "stem":
        stemmer = nltk.stem.PorterStemmer()
        result = []
        for t in tokens:
            if _PLACEHOLDER_RE.match(t):
                result.append(t)
            else:
                result.append(stemmer.stem(t))
        return result

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tagged = nltk.pos_tag(tokens)
    result = []
    for t, pos in tagged:
        if _PLACEHOLDER_RE.match(t):
            result.append(t)
        else:
            result.append(lemmatizer.lemmatize(t, get_pos(pos)))
    return result
