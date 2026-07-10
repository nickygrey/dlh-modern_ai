#!/usr/bin/env python3
"""Module to remove duplicate rows from a pandas DataFrame."""


def remove_duplicates(df):
    """Drop all duplicate rows from the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame to process.

    Returns:
        pandas.DataFrame: The deduplicated DataFrame.
    """
    return df.drop_duplicates()
