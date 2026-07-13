#!/usr/bin/env python3
"""Module to engineer new features from the Telco Customer Churn dataset."""
import pandas as pd


def create_features(df):
    """Engineer NumServices and TenureGroup features, dropping raw columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame with engineered features.
    """
    df = df.copy()

    service_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]

    # 1. Compute NumServices
    num_services = pd.Series(0, index=df.index)
    for col in service_cols:
        if col == 'InternetService':
            num_services += df[col].isin(
                ['DSL', 'Fiber optic']
            ).astype(int)
        else:
            num_services += (df[col] == 'Yes').astype(int)

    df['NumServices'] = num_services

    # 2. Compute TenureGroup (right-inclusive, 0-excluded)
    bins = [0, 12, 24, 48, 60, float('inf')]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=bins, labels=labels, right=True
    )

    # 3. Drop original columns used for feature creation
    cols_to_drop = ['tenure'] + service_cols
    df = df.drop(columns=cols_to_drop)

    return df
