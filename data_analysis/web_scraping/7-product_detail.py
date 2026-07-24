#!/usr/bin/env python3
"""Module to scrape single product detail page using Selenium."""
import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """Open a product detail page and extract title, price, desc, and rating.

    Args:
        url (str): Target product detail page URL.
        delay (float): Time to wait for page rendering in seconds.

    Returns:
        dict: Product details dictionary with title, price,
              description, and rating.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(delay)

        caption_elem = driver.find_element("css selector", ".caption")
        h4_elems = caption_elem.find_elements("css selector", "h4")
        if len(h4_elems) > 1:
            title = h4_elems[1].text
        else:
            title = h4_elems[0].text

        price_elem = driver.find_element("css selector", "h4.price")
        price = price_elem.text

        desc_elem = driver.find_element("css selector", "p.description")
        description = desc_elem.text

        try:
            stars = driver.find_elements(
                "css selector", ".ratings p.ws-icon-star"
            )
            if not stars:
                stars = driver.find_elements(
                    "css selector", ".ratings .ws-icon-star"
                )
            if not stars:
                stars = driver.find_elements(
                    "css selector", ".ratings .glyphicon-star"
                )
            rating = len(stars)
        except Exception:
            rating = 0

        product_detail = {
            "title": title,
            "price": price,
            "description": description,
            "rating": rating
        }
    finally:
        driver.quit()

    return product_detail
