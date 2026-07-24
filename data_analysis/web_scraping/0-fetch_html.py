#!/usr/bin/env python3
"""Module to fetch HTML content from a URL."""
import requests


def fetch_html(url, headers=None, timeout=10):
    """Fetch a web page and return its HTML text.

    Args:
        url (str): Target web page URL.
        headers (dict, optional): Custom HTTP headers.
        timeout (int): Request timeout in seconds.

    Returns:
        str: HTML content of the page.
    """
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text
