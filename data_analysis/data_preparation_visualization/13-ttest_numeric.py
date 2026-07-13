#!/usr/bin/env python3
"""Module to perform Welch's t-tests on continuous numerical features."""
from scipy import stats


def ttest_numeric(df):
    """Compare numeric feature distributions by Churn status.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a 'Churn' column.

    Returns:
        dict: A dictionary mapping column names to their t-test p-value.
    """
    results = {}
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        group_yes = df[df['Churn'] == 'Yes'][col].dropna()
        group_no = df[df['Churn'] == 'No'][col].dropna()

        # Perform Welch's t-test (equal_var=False)
        _, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
        results[col] = p_val

    return results
