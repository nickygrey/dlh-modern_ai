#!/usr/bin/env python3
"""Module to scrape quotes using the quotes.toscrape.com API."""
import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """Scrape quote data from all API pages of the target site.

    Args:
        base_url (str): The base URL of the site.

    Returns:
        list[dict]: A list of quote dictionaries with text, author, and tags.
    """
    all_quotes = []
    page = 1

    while True:
        api_url = f"{base_url}/api/quotes?page={page}"
        html_str = fetch_html(api_url)
        data = json.loads(html_str)

        quotes = data.get("quotes", [])
        for q in quotes:
            text = q.get("text", "")
            author_obj = q.get("author")
            if isinstance(author_obj, dict):
                author = author_obj.get("name", "")
            else:
                author = author_obj if author_obj else ""
            tags = q.get("tags", [])

            all_quotes.append({
                "text": text,
                "author": author,
                "tags": tags
            })

        if not data.get("has_next"):
            break

        page += 1

    return all_quotes
