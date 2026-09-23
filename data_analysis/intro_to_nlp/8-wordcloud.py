#!/usr/bin/env python3
"""Module to generate word clouds from a tokenized corpus."""
import matplotlib.pyplot as plt
import wordcloud


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """Generate and display a word cloud from a preprocessed corpus.

    Args:
        corpus_tokens (list[list[str]]): Corpus represented as a list of
            token lists.
        max_words (int, optional): Maximum number of words in the cloud.
            Defaults to 200.
        label (str | None, optional): Optional title label for the plot.
            Defaults to None.

    Returns:
        wordcloud.WordCloud: The fitted WordCloud object.
    """
    text = " ".join(tok for doc in corpus_tokens for tok in doc)

    wc = wordcloud.WordCloud(
        max_words=max_words,
        background_color="white",
        width=800,
        height=400,
        random_state=42
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    if label:
        plt.title(f"WordCloud — {label}")
    else:
        plt.title("WordCloud")

    plt.tight_layout()

    return wc
