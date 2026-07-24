#!/usr/bin/env python3
"""Module to scrape static product pages using Selenium."""
import time
from selenium import webdriver


def scrape_products(url):
    """Scrape product details from a static product category page.

    Args:
        url (str): Target product category URL.

    Returns:
        list[dict]: List of product dictionaries with title, price,
                    description, and rating.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)
    products = []

    try:
        driver.get(url)
        time.sleep(2)

        cards = driver.find_elements("css selector", "div.thumbnail")
        for card in cards:
            title_elem = card.find_element("css selector", "a.title")
            title = title_elem.get_attribute("title") or title_elem.text

            price_elem = card.find_element("css selector", "h4.price")
            price = price_elem.text

            desc_elem = card.find_element("css selector", "p.description")
            description = desc_elem.text

            try:
                rating_elem = card.find_element(
                    "css selector", ".ratings p[data-rating]"
                )
                rating = int(rating_elem.get_attribute("data-rating"))
            except Exception:
                try:
                    rating_elem = card.find_element(
                        "css selector", "p[data-rating]"
                    )
                    rating = int(rating_elem.get_attribute("data-rating"))
                except Exception:
                    rating = 0

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating
            })
    finally:
        driver.quit()

    return products


def scrape_products_list(url):
    """Scrape product details from a static product category page.

    Args:
        url (str): Target product category URL.

    Returns:
        list[dict]: List of product dictionaries.
    """
    return scrape_products(url)
