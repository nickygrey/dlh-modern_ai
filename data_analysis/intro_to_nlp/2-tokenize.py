#!/usr/bin/env python3
"""Tokenization and emoticon normalization module for SMS messages."""
import nltk

EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    """Normalize emoticons in token list by replacing or removing.

    Args:
        tokens (list): List of token strings.
        emoticon_action (str): Action for emoticons ('replace' or 'remove').

    Returns:
        list: Normalized token list.
    """
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method="tweet"):
    """Tokenize a cleaned SMS message using the specified method.

    Args:
        text (str): Cleaned message string to tokenize.
        method (str, optional): Strategy to use ('tweet', 'word', or 'split').
            Defaults to 'tweet'.

    Returns:
        list: Tokenized message as a list of strings.

    Raises:
        ValueError: If method is not supported.
    """
    if not isinstance(text, str):
        return []

    if method == "tweet":
        tokenizer = nltk.TweetTokenizer(reduce_len=True)
        return tokenizer.tokenize(text)
    if method == "word":
        return nltk.word_tokenize(text)
    if method == "split":
        return text.split()

    raise ValueError("Invalid tokenizer method")
