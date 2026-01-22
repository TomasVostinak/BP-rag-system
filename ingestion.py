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

URL_FILE = "data/project-urls.txt"
OUTPUT_FILE = "data/ingested-text.jsonl"

UNWANTED_PHRASES = [
    "soubory cookies",
    "Souhlasím s použitím",
    "analytických cookies",
    "statistických cookies",
    "měřit návštěvnost",
    "měřit výkon webu"
]

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
    main = soup.body or soup.main or soup

    for tag in main(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

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