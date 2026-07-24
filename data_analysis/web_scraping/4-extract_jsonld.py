#!/usr/bin/env python3
"""Module to extract quote data from JSON-LD embedded script tags."""
from bs4 import BeautifulSoup
import json
fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """Extract quote objects from application/ld+json blocks in a web page.

    Args:
        url (str): Target web page URL.

    Returns:
        list[dict]: List of dictionaries containing text, author, and tags.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    quotes = []

    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        content = script.string or script.get_text()
        if not content:
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue

        if isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                items = data["@graph"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "Quote":
                text = item.get("text")
                author_obj = item.get("author")
                if isinstance(author_obj, dict):
                    author = author_obj.get("name")
                elif isinstance(author_obj, str):
                    author = author_obj
                else:
                    author = None

                raw_keywords = item.get("keywords")
                if isinstance(raw_keywords, str):
                    tags = [k.strip() for k in raw_keywords.split(',')]
                elif isinstance(raw_keywords, list):
                    tags = raw_keywords
                else:
                    tags = []

                quotes.append({
                    "text": text,
                    "author": author,
                    "tags": tags
                })

    return quotes
