#!/usr/bin/env python3
"""Module to plot the distribution of the Churn target variable."""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Generate a bar plot showing the distribution of Churn.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Churn' column.
    """
    plt.figure(figsize=(12, 8))
    counts = df['Churn'].value_counts()
    colors = [
        'skyblue' if idx == 'No' else 'salmon' for idx in counts.index
    ]
    plt.bar(counts.index, counts.values, color=colors)
    plt.title("Churn Distribution")
    plt.ylabel("Count")
    plt.show()
