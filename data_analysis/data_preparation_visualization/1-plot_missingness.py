#!/usr/bin/env python3
"""Module to visualize missing values in a pandas DataFrame."""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Generate a scatter plot visualizing missing values in a DataFrame.

    Args:
        df: The input pandas DataFrame to analyze.
    """
    plt.figure(figsize=(12, 8))

    # Find coordinates of all missing values
    missing_rows, missing_cols = np.where(df.isnull())

    # Generate scatter plot using the vertical bar marker
    plt.scatter(missing_rows, missing_cols, marker='|')

    # Explicitly map y-ticks to column names
    plt.yticks(range(len(df.columns)), df.columns)

    plt.tight_layout()
    plt.show()
