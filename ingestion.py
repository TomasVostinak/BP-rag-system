#################################################
### Soubor pro crawler a čištění HTML stránek ###
#################################################

import requests
import time
import bs4

URL_FILE = "data/project-urls.txt"

def fetch_html(url):
    response = requests.get(url, timeout=10)
    if response.ok:
        response.encoding = response.apparent_encoding
        return response.text
    else:
        print(f"Error fetching {url}: {response.status_code}")

def clean_html(html):
    soup = bs4.BeautifulSoup(html, "html.parser")
    main = soup.body

    for tag in main(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line.split()) <= 5:
            continue
        lines.append(line)
    text = "\n".join(lines)

    return text

def ingest_urls():
    with open(URL_FILE, "r", encoding="utf-8") as file:
        urls = [line.strip() for line in file if line.strip()]

    for i, url in enumerate(urls, start=1):
        print(f"Fetching URL {i}: {url}")
        html = fetch_html(url)

        text = clean_html(html)
    
    time.sleep(0.5)

if __name__ == "__main__":
    #ingest_urls()
    html = fetch_html("https://knihovna.mestojablonec.cz/")
    text = clean_html(html)
    print(text)
    pass