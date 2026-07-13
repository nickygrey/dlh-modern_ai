#!/usr/bin/env python3
"""Module to split data into train and test sets with stratification."""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """Split data into stratified train and test sets.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target (str): Name of the target column.
        test_size (float): Proportion to include in test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) data arrays.
    """
    X = df.drop(columns=[target])
    y = df[target]

    return model_selection.train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
