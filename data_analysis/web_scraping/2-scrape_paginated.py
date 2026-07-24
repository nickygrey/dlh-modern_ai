#!/usr/bin/env python3
"""Module to handle paginated quote scraping."""
from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """Scrape quotes across all pages starting from base_url.

    Args:
        base_url (str): The starting URL for scraping.

    Returns:
        list[dict]: Combined list of quote dictionaries from all pages.
    """
    all_quotes = []
    current_url = base_url

    while current_url:
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, 'html.parser')

        for quote in soup.find_all('div', class_='quote'):
            text_elem = quote.find('span', class_='text')
            author_elem = quote.find('small', class_='author')
            tag_elems = quote.find_all('a', class_='tag')

            text = text_elem.get_text() if text_elem else ''
            author = author_elem.get_text() if author_elem else ''
            tags = [tag.get_text() for tag in tag_elems]

            all_quotes.append({
                'text': text,
                'author': author,
                'tags': tags
            })

        next_li = soup.find('li', class_='next')
        if next_li and next_li.find('a'):
            next_href = next_li.find('a').get('href')
            current_url = parse.urljoin(current_url, next_href)
            time.sleep(1)
        else:
            current_url = None

    return all_quotes
