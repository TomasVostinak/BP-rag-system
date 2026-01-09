#################################################
### Soubor pro crawler a čištění HTML stránek ###
#################################################

import requests
import time

URL_FILE = "data/project-urls.txt"

def fetch_html(url):
    response = requests.get(url, timeout=10)
    if response.ok:
        return response.text
    else:
        print(f"Error fetching {url}: {response.status_code}")

def clean_html(html):
    return 0

def ingest_urls():
    with open(URL_FILE, "r", encoding="utf-8") as file:
        urls = [line.strip() for line in file if line.strip()]

    for i, url in enumerate(urls, start=1):
        print(f"Fetching URL {i}: {url}")
        html = fetch_html(url)

        text = clean_html(html)
    
    time.sleep(0.5)

#fetch_html("https://cs.wikipedia.org/wiki/Jablonec_nad_Nisou")

if __name__ == "__main__":
    ingest_urls()
    pass