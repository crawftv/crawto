import pytest
from crawto.async_web_scrape import async_web_scrape
import requests
from bs4 import BeautifulSoup


def test_async_web_scrape():
    def scrape_pages(url,all_soup_objects):
        r = requests.get(url).text
        soup = BeautifulSoup(r,"html.parser")
        all_soup_objects.extend(soup)
    urls = ["https://www.wikipedia.com", "https://en.wikipedia.org/wiki/Main_Page", "https://it.wikipedia.org/wiki/Pagina_principale"]
    all_soup_objects=[]
    async_web_scrape(urls, scrape_pages,40,all_soup_objects)
    assert len(all_soup_objects) > 0
