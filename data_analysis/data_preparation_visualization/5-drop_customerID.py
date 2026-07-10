#!/usr/bin/env python3
"""Module to drop uninformative identification columns from a DataFrame."""


def drop_customerID(df):
    """Drop the customerID column from the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame without the customerID column.
    """
    return df.drop(columns=['customerID'])
