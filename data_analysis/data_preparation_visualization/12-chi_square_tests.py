#!/usr/bin/env python3
"""Module to perform Chi-Square tests of independence on categorical data."""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Perform Chi-Square tests between categorical columns and Churn.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a 'Churn' column.

    Returns:
        dict: Dictionary with feature names as keys and p-values as values.
    """
    p_values = {}
    for col in df.columns:
        if col != 'Churn' and df[col].dtype == 'object':
            contingency_table = pd.crosstab(df[col], df['Churn'])
            _, p, _, _ = stats.chi2_contingency(contingency_table)
            p_values[col] = p
    return p_values
