#!/usr/bin/env python3
"""Module to standardize numeric features using StandardScaler."""
from sklearn import preprocessing


def scale_numeric(df):
    """Standardize numeric columns using StandardScaler.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame with scaled columns.
    """
    df_scaled = df.copy()
    scaler = preprocessing.StandardScaler()
    cols_to_scale = ['MonthlyCharges', 'TotalCharges']

    df_scaled[cols_to_scale] = scaler.fit_transform(
        df_scaled[cols_to_scale]
    )

    return df_scaled
