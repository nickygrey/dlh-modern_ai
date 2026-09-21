#!/usr/bin/env python3
"""Module for basic exploration of SMS Spam dataset."""
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """Perform initial dataset exploration with bar chart and histogram.

    Args:
        df (pandas.DataFrame): DataFrame containing SMS messages and labels.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    counts = df['label'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax1)
    ax1.set_title("Ham vs Spam Counts")
    ax1.set_xlabel("label")
    ax1.set_ylabel("count")

    if 'msg_length' in df.columns:
        lengths = df['msg_length']
    else:
        lengths = df['message'].str.len()

    sns.histplot(x=lengths, bins=50, ax=ax2)
    ax2.set_title("Histogram of Raw Message Lengths")
    ax2.set_xlabel("length")
    ax2.set_ylabel("count")

    plt.tight_layout()
    return None
