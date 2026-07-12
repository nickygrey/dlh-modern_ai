#!/usr/bin/env python3
"""Module to plot categorical distributions of a DataFrame."""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """Visualizes the distribution of categorical features in a grid layout.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns_to_plot (list, optional): List of columns to plot.
    """
    if columns_to_plot is None:
        columns_to_plot = [
            c for c in df.columns if df[c].dtype == 'object' and c != 'Churn'
        ]
    else:
        columns_to_plot = list(columns_to_plot)

    n_cols, n_rows = 3, (len(columns_to_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns_to_plot):
        counts = df[col].value_counts()
        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
