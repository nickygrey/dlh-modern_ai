#!/usr/bin/env python3
"""Module to handle missing values in the TotalCharges column."""


def clean_total_charges(df, method='drop'):
    """Clean missing values in TotalCharges using a specified strategy.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        method (str): Strategy to handle missing values ('drop', 'median',
                     or 'impute'). Defaults to 'drop'.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    df_clean = df.copy()

    if method == 'drop':
        return df_clean.dropna(subset=['TotalCharges'])

    if method == 'median':
        median_value = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
            median_value
        )

    if method == 'impute':
        calculated_charges = df_clean['MonthlyCharges'] * df_clean['tenure']
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
            calculated_charges
        )

    return df_clean
