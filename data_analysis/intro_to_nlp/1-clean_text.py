#!/usr/bin/env python3
"""Text normalization module for SMS messages."""
import emoji
import re


_DATASET_PLACEHOLDER_MAP = {
    "<#>": "<NUM>",
    "<decimal>": "<NUM>",
    "<time>": "<TIME>",
    "<url>": "<URL>",
    "<email>": "<EMAIL>",
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses with ASCII equivalents."""
    replacements = {
        r"[‘’‚‛]": "'",
        r"[“”„‟]": "\"",
        r"[‐‑‒–—―−]": "-",
        r"…": "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """Clean and normalize raw text strings.

    Args:
        text (str): Input text to clean.
        replace_num (bool): Replace detected numbers and phone patterns.
        replace_url (bool): Replace URLs with <URL>.
        emoji_action (str): Action for emojis (replace, remove, or keep).

    Returns:
        str: Cleaned and normalized text, or "" if text is None.
    """
    if text is None:
        return ""

    # 1. lowercase + strip
    text = text.lower().strip()

    # 2. dataset placeholders
    for placeholder, replacement in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(placeholder, replacement)

    # 3. normalize_unicode_punct()
    text = normalize_unicode_punct(text)

    # 4. URL replacement
    if replace_url:
        text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)

    # 5. number replacement (2 passes)
    if replace_num:
        # Pass 1: phone-like strings
        text = re.sub(r"\+?\d[\d\s\-]{6,}\d", "<NUM>", text)
        # Pass 2: integers, decimals, currency-prefixed amounts
        num_regex = (
            r"(?:£|\$|€)\d+(?:[.,]\d+)*|"
            r"(?<!<)\b\d+(?:[.,]\d+)*\b"
        )
        text = re.sub(num_regex, "<NUM>", text)

    # 6. emoji handling
    if emoji_action == "replace":
        text = emoji.replace_emoji(text, replace="<EMO>")
    elif emoji_action == "remove":
        text = emoji.replace_emoji(text, replace=" ")

    # 7. collapse repeated ! / ?
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)

    # 8. collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
