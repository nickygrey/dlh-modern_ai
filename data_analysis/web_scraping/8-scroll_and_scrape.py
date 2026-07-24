#!/usr/bin/env python3
"""Module to scroll and scrape infinite-scroll product pages."""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Scroll through an infinite-scroll page and extract unique products.

    Args:
        url (str): Target infinite-scroll page URL.
        scroll_pause (float): Maximum delay in seconds to wait per scroll.

    Returns:
        list[dict]: List of unique product dictionaries.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)
    products = []
    seen = set()

    try:
        driver.get(url)
        time.sleep(1)

        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        while True:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            start = time.time()
            height_changed = False
            while time.time() - start < scroll_pause:
                time.sleep(0.1)
                new_height = driver.execute_script(
                    "return document.body.scrollHeight"
                )
                if new_height > last_height:
                    last_height = new_height
                    height_changed = True
                    break

            if not height_changed:
                break

        cards = driver.find_elements("css selector", "div.thumbnail")
        for card in cards:
            title_elem = card.find_element("css selector", "a.title")
            title = title_elem.get_attribute("title")
            if not title:
                title = title_elem.text

            price_elem = card.find_element("css selector", "h4.price")
            price = price_elem.text

            key = (title, price)
            if key in seen:
                continue
            seen.add(key)

            desc_elem = card.find_element("css selector", "p.description")
            description = desc_elem.text

            try:
                stars = card.find_elements(
                    "css selector", ".ratings p.ws-icon-star"
                )
                if not stars:
                    stars = card.find_elements(
                        "css selector", ".ratings span.ws-icon-star"
                    )
                if not stars:
                    stars = card.find_elements(
                        "css selector", ".ratings .ws-icon-star"
                    )
                if not stars:
                    stars = card.find_elements(
                        "css selector", ".ratings .glyphicon-star"
                    )
                if not stars:
                    stars = card.find_elements(
                        "css selector", ".ratings [class*='star']"
                    )
                rating = len(stars)
            except Exception:
                rating = 0

            if rating == 0:
                try:
                    r_elem = card.find_element(
                        "css selector", ".ratings p[data-rating]"
                    )
                    rating = int(r_elem.get_attribute("data-rating"))
                except Exception:
                    pass

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating
            })
    finally:
        driver.quit()

    return products
