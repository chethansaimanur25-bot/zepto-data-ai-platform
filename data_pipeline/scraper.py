import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://books.toscrape.com/"
TARGET_CATEGORIES = ["Travel", "Mystery", "Historical Fiction", "Classics"]


def get_soup(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def get_category_links():
    soup = get_soup(BASE)
    links = {}
    for a in soup.select("div.side_categories ul li ul li a"):
        links[a.get_text(strip=True)] = urljoin(BASE, a["href"])
    return links


def scrape_category(name, url):
    rows = []
    while url:
        soup = get_soup(url)
        for book in soup.select("article.product_pod"):
            rows.append({
                "title": book.h3.a["title"],
                "price": book.select_one("p.price_color").get_text(strip=True),
                "star_rating": book.select_one("p.star-rating")["class"][1],
                "availability": book.select_one("p.availability").get_text(strip=True),
                "category": name,
            })
        nxt = soup.select_one("li.next a")
        url = urljoin(url, nxt["href"]) if nxt else None
        time.sleep(0.5)
    return rows


if __name__ == "__main__":
    links = get_category_links()
    all_rows = []
    for name in TARGET_CATEGORIES:
        rows = scrape_category(name, links[name])
        print(f"{name}: {len(rows)} books")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv("data_pipeline/raw_books.csv", index=False)
    print(df.shape)
    print(df.head())