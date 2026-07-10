#!/usr/bin/env python3
"""Module to convert data types of specific columns in a DataFrame."""
import pandas as pd


def convert_columns(df):
    """Convert TotalCharges to numeric and map SeniorCitizen to strings.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame with converted columns.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df
