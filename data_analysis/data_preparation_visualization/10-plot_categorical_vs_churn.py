#!/usr/bin/env python3
"""Module to plot churn rates across categorical features."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Plot the churn rate for each category in the specified column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        col (str): The categorical column to analyze against Churn.
    """
    plt.figure(figsize=(12, 8))

    # Calculate Churn rate (proportion of 'Yes') for each category
    churn_rates = (df['Churn'] == 'Yes').groupby(df[col]).mean()

    plt.bar(churn_rates.index, churn_rates.values)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()
