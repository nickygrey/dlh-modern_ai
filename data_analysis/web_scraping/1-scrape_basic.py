#!/usr/bin/env python3
"""Module to scrape basic quotes from quotes.toscrape.com."""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """Scrape quote text, author, and tags from the specified URL.

    Args:
        url (str): Target web page URL.

    Returns:
        list[dict]: A list of dictionaries containing quote metadata.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    quotes_data = []

    for quote in soup.find_all('div', class_='quote'):
        text_elem = quote.find('span', class_='text')
        author_elem = quote.find('small', class_='author')
        tag_elems = quote.find_all('a', class_='tag')

        text = text_elem.get_text() if text_elem else ''
        author = author_elem.get_text() if author_elem else ''
        tags = [tag.get_text() for tag in tag_elems]

        quotes_data.append({
            'text': text,
            'author': author,
            'tags': tags
        })

    return quotes_data
