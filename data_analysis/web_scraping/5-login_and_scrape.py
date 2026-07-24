#!/usr/bin/env python3
"""Module to handle session login and authenticated web scraping."""
from bs4 import BeautifulSoup
import requests


def login_and_scrape(login_url, user, pwd):
    """Log in to a website with credentials and scrape authenticated content.

    Args:
        login_url (str): The URL of the login page.
        user (str): Username for login.
        pwd (str): Password for login.

    Returns:
        list[dict]: List of dictionaries containing quote metadata.
    """
    session = requests.Session()

    response = session.get(login_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_elem = soup.find('input', {'name': 'csrf_token'})
    csrf_token = csrf_elem.get('value') if csrf_elem else ''

    payload = {
        'username': user,
        'password': pwd,
        'csrf_token': csrf_token
    }

    post_response = session.post(login_url, data=payload)
    post_response.raise_for_status()

    target_url = "https://quotes.toscrape.com/"
    page_response = session.get(target_url)
    page_response.raise_for_status()

    target_soup = BeautifulSoup(page_response.text, 'html.parser')
    quotes_data = []

    for quote in target_soup.find_all('div', class_='quote'):
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
