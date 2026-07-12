#!/usr/bin/env python3
"""Module to plot the distribution of the Churn target variable."""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Generate a bar plot showing the distribution of Churn.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Churn' column.
    """
    plt.figure(figsize=(12, 8))

    # Calculate frequencies
    counts = df['Churn'].value_counts()

    # Map colors precisely to the index order ('No' -> skyblue, 'Yes' -> salmon)
    colors = ['skyblue' if idx == 'No' else 'salmon' for idx in counts.index]

    # Render a clean, unadorned bar plot to match the reference structure
    plt.bar(counts.index, counts.values, color=colors)

    plt.show()
