#!/usr/bin/env python3
"""Token filtering module for SMS messages."""
import re

_PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')
PLACEHOLDER_RE = _PLACEHOLDER_RE
PLACEHOLDERRE = _PLACEHOLDER_RE


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """Filter low-information tokens from a token list.

    Args:
        tokens (list[str]): List of tokens to filter.
        min_len (int, optional): Minimum token length to keep. Defaults to 2.
        strip_hashtag (bool, optional): Strip leading # from hashtags.
            Defaults to False.

    Returns:
        list[str]: Filtered list of tokens.
    """
    if not tokens or not isinstance(tokens, list):
        return []

    filtered = []
    for token in tokens:
        if not isinstance(token, str):
            continue

        if _PLACEHOLDER_RE.match(token):
            filtered.append(token)
            continue

        t = token
        if strip_hashtag and t.startswith('#'):
            t = t[1:]

        if len(t) < min_len:
            continue

        if not any(c.isalpha() for c in t):
            continue

        filtered.append(t)

    return filtered
