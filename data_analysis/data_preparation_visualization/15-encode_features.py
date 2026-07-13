#!/usr/bin/env python3
"""Module to encode categorical and binary variables for modeling."""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode binary, categorical, and target variables for machine learning.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        tuple: (df_encoded, churn_le, binary_oe, tenure_oe) where:
            - df_encoded is the fully encoded DataFrame.
            - churn_le is the fitted LabelEncoder for Churn.
            - binary_oe is the fitted OrdinalEncoder for binary columns.
            - tenure_oe is the fitted OrdinalEncoder for TenureGroup.
    """
    df_enc = df.copy()

    # 1. Encode Churn target variable using LabelEncoder
    churn_le = preprocessing.LabelEncoder()
    df_enc['Churn'] = churn_le.fit_transform(df_enc['Churn'])

    # 2. Encode binary features using OrdinalEncoder
    binary_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling'
    ]
    binary_oe = preprocessing.OrdinalEncoder(dtype='int64')
    df_enc[binary_cols] = binary_oe.fit_transform(df_enc[binary_cols])

    # 3. Encode TenureGroup using OrdinalEncoder (alphabetical order)
    tenure_oe = preprocessing.OrdinalEncoder(dtype='int64')
    df_enc[['TenureGroup']] = tenure_oe.fit_transform(
        df_enc[['TenureGroup']].astype(str)
    )

    # 4. One-hot encode Contract and PaymentMethod with drop_first=True
    df_enc = pd.get_dummies(
        df_enc,
        columns=['Contract', 'PaymentMethod'],
        drop_first=True,
        dtype='int64'
    )

    return df_enc, churn_le, binary_oe, tenure_oe
