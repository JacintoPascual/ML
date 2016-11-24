import requests
from bs4 import BeautifulSoup

def trade_spider(max_pages):
    page = 1
    while page < max_pages:
        print("Hello")
        url = 'https://buckysroom.org/trade/search.php?page' + str(page)
        print(url)
        source_code = requests.get(url)
        plain_text = source_code.text
        print(plain_text)
        soup = BeautifulSoup(plain_text)
        for link in soup.findAll('a', {'class': 'item-name'}):
            href = "https://buckysroom.org" + link.__getitem__('href')
            title = link.__str__()
            print(href)
        page += 1


def get_simple_item_date(item_url):
    source_code = requests.get(item_url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text)
    for item_name in soup.findAll('div', {'class': 'i-name'}):
        print(item_name.__str__())
        

trade_spider(1)