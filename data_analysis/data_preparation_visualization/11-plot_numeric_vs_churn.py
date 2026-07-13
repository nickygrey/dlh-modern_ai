#!/usr/bin/env python3
"""Module to plot continuous numerical distributions against churn."""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Compare continuous numeric feature distributions by churn status.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        col (str): The name of the numeric column to plot.
    """
    plt.figure(figsize=(12, 8))

    # Split populations based on the Churn label
    data_no = df[df['Churn'] == 'No'][col]
    data_yes = df[df['Churn'] == 'Yes'][col]

    # Generate the side-by-side binned layout naturally
    plt.hist([data_no, data_yes], bins=30, label=['No', 'Yes'])

    # Apply configuration layout titles and legends
    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()
