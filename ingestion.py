#################################################
### Soubor pro crawler a čištění HTML stránek ###
#################################################

import io
import time
import requests
import json
import pdfplumber
import docx
import bs4
import re

URL_FILE = "data/project-urls.txt"
OUTPUT_FILE = "data/ingested-text.jsonl"

UNWANTED_PHRASES = ["soubory cookies", "Souhlasím s použitím", "analytických cookies", "statistických cookies", "měřit návštěvnost", "měřit výkon webu"]
UNWANTED_SELECTORS = ["a.fusion-one-page-text-link", "a.odkaz_domu", "a.scroll-to-top-btn", "a.skip-link", "aside#sidebar", "aside.leve_menu", "div#alertbox", "div#container-left-side", "div#cookies_info", "div#cookiescript_injected_wrapper", "div#copy", "div#navbar", "div#skip-link", "div#sliders-container", "div.avada-footer-scripts", "div.cesta_ke_clanku", "div.ch2-region-g0", "div.FixedHeader", "div.fusion-footer", "div.fusion-sliding-bar-wrapper", "div.gc-Modal", "div.gc-Modal-container", "div.HeaderBar", "div.info_podclankem", "div.LayoutWithLeftSidebar-sidebar", "div.LayoutWithRightSidebar-sidebar", "div.nahled_tisk", "div.navbar", "div.Paginator", "div.PopupNav", "div.Search", "div.visually-hidden", "header", "nav.Breadcrumbs", "section.to-top-container"]

def fetch_page(url, retries=3, timeout=10):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            ct = response.headers.get("Content-Type", "").lower()
            print(f"Content-Type: {ct}")
            if not response.ok:
                print(f"Error fetching {url}: {response.status_code}")
                return None, None
            
            if "html" in ct or "text/plain" in ct:
                response.encoding = response.apparent_encoding
                return response.text, "html"
            elif "pdf" in ct:
                with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text, "pdf"
            elif ct in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                doc = docx.Document(io.BytesIO(response.content))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text, "docx"
            else:
                print(f"Unsupported Content-Type for {url}: {ct}")
                return None, None
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url}")
        except requests.exceptions.ReadTimeout:
            print(f"Read timeout fetching {url}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error fetching {url}")
        
        if attempt < retries:
            print(f"Retrying ({attempt}/{retries})...")
            time.sleep(2)
        
    print(f"Failed to fetch {url}")
    return None, None
            
def clean_html(html):
    soup = bs4.BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    
    for selector in UNWANTED_SELECTORS:
         for element in soup.select(selector):
             element.decompose()

    main = soup.body or soup.main or soup
    text = main.get_text(separator="\n", strip=True)

    return clean_text(text)

def clean_text(text, min_words=5):
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line.split()) < min_words:
            continue
        if any(phrase in line for phrase in UNWANTED_PHRASES):
            continue
        lines.append(line)
        
    return " ".join(lines)

def is_low_quality(text):
    if len(text) <= 300:
        return True

    sentences = re.split(r"[.!?]", text)
    if len(sentences) <= 3:
        return True

    return False

def ingest_urls():
    with open(URL_FILE, "r", encoding="utf-8") as file:
        urls = [line.strip() for line in file if line.strip()]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for i, url in enumerate(urls, start=1):
            print(f"Fetching URL {i}: {url}")

            page, type = fetch_page(url)
            if page is None:
                time.sleep(0.5)
                continue
        
            if type == "html":
                text = clean_html(page)
            else:
                text = clean_text(page)

            if is_low_quality(text):
                print("Low quality content, skipping.")
                continue

            if text:
                print(f"Extracted {len(text)} characters from {url}")
                record = {
                    "id": i,
                    "url": url,
                    "text": text
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                print(f"Failed to extract text from {url}")
            
            time.sleep(0.5)

if __name__ == "__main__":
    ingest_urls()