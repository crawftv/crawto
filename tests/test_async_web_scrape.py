import pytest
from crawto.async_web_scrape import async_web_scrape
import requests
from bs4 import BeautifulSoup


def filter_biz_url(url):
    if ("?hrid" in url) | ("/adredir?ad_business_id" in url) | ("/search?cflt" in url):
        pass
    else:
        return url


def test_async_web_scrape():
    def scrape_yelp_search(start_number, all_links):
        url = (
            "https://www.yelp.com/search?cflt=coffee&find_loc=New%20York%2C%20NY&start="
        )
        r = requests.get(url + str(start_number)).text
        soup = BeautifulSoup(r, "html.parser")
        a = soup.find_all(
            "a",
            {
                "class": "lemon--a__373c0__IEZFH link__373c0__29943 link-color--blue-dark__373c0__1mhJo link-size--inherit__373c0__2JXk5"
            },
        )
        links = [
            filter_biz_url(i["href"])
            for i in a
            if filter_biz_url(i["href"]) is not None
        ]
        all_links.extend(links)

    all_links = []
    async_web_scrape(list(range(0, 1200, 30)), scrape_yelp_search, 40, all_links)
    assert len(all_links) > 0
