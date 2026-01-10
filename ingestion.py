#################################################
### Soubor pro crawler a čištění HTML stránek ###
#################################################

import requests
import pdfplumber
import io
import time
import bs4

URL_FILE = "data/project-urls.txt"

def fetch_page(url, type):
    response = requests.get(url, timeout=10)
    ct = response.headers.get("Content-Type", "").lower()
    print(f"Content-Type: {ct}")
    if response.ok:
        if type == "pdf" or ct == "application/pdf":
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        
        elif type == "docx":
            pass

        elif type == "html":
            response.encoding = response.apparent_encoding
            return response.text
    else:
        print(f"Error fetching {url}: {response.status_code}")
        return None

def clean_html(html):
    soup = bs4.BeautifulSoup(html, "html.parser")
    main = soup.body

    for tag in main(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    return clean_text(text)

def clean_text(text, min_words=6):
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line.split()) <= min_words:
            continue
        lines.append(line)
        
    return "\n".join(lines)

def ingest_urls():
    with open(URL_FILE, "r", encoding="utf-8") as file:
        urls = [line.strip() for line in file if line.strip()]

    for i, url in enumerate(urls, start=1):
        print(f"Fetching URL {i}: {url}")

        if url.endswith(".pdf"):
            page = fetch_page(url, "pdf")
            if page is None:
                time.sleep(0.5)
                continue
            text = clean_text(page)

        elif url.endswith(".docx"):
            page = fetch_page(url, "docx")
            if page is None:
                time.sleep(0.5)
                continue
            text = clean_text(page)

        else:
            page = fetch_page(url, "html")
            if page is None:
                time.sleep(0.5)
                continue
            text = clean_html(page)

        if text:
            print(f"Extracted {len(text)} characters from {url}")
        else:
            print(f"Failed to extract text from {url}")

        # add to output file here

        time.sleep(0.5)

if __name__ == "__main__":
    ingest_urls()
    #page = fetch_page("https://www.mestojablonec.cz/data/files/14/14b/5d51da9b3517c473ca248a494ebbd418368/07-zadost-o-souhlas-podle-17-vodniho-zakona.docx", "html")
    #text = clean_text(page)
    #print(text)