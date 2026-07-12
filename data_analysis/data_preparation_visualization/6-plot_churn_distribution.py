#!/usr/bin/env python3
"""Module to plot the distribution of the Churn target variable."""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Generate a bar plot showing the distribution of Churn.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Churn' column.
    """
    plt.figure(figsize=(12, 8))

    # Calculate the frequency counts of each class label
    counts = df['Churn'].value_counts()

    # Align colors safely based on the index labels
    colors = ['skyblue' if idx == 'No' else 'salmon' for idx in counts.index]

    # Render the bar plot using pure matplotlib
    plt.bar(counts.index, counts.values, color=colors)

    # Apply mandatory title and label decorations
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
