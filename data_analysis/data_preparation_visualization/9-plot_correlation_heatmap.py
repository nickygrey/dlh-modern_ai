#!/usr/bin/env python3
"""Module to plot a correlation heatmap of numeric features."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df):
    """Generate an annotated correlation heatmap for numeric columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(6, 5))
    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Matrix")
    plt.show()
