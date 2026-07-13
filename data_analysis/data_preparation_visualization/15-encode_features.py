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

    # 2. Encode binary features using OrdinalEncoder (No -> 0, Yes -> 1)
    binary_cols = [
        'Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen'
    ]
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for col in binary_cols:
        df_enc[col] = binary_oe.fit_transform(
            df_enc[[col]].astype(str)
        ).astype('int64')

    # 3. Encode TenureGroup using OrdinalEncoder (alphabetical order)
    tenure_categories = ['0-12', '13-24', '25-48', '49-60', '60+']
    tenure_oe = preprocessing.OrdinalEncoder(categories=[tenure_categories])
    df_enc['TenureGroup'] = tenure_oe.fit_transform(
        df_enc[['TenureGroup']].astype(str)
    ).astype('int64')

    # 4. One-hot encode Contract and PaymentMethod with drop_first=True
    df_enc = pd.get_dummies(
        df_enc,
        columns=['Contract', 'PaymentMethod'],
        drop_first=True,
        dtype='int64'
    )

    return df_enc, churn_le, binary_oe, tenure_oe
