#################################################
### Soubor pro crawler a čištění HTML stránek ###
#################################################

import io
import time
import requests
import pdfplumber
import docx
import bs4

URL_FILE = "data/project-urls.txt"
OUTPUT_FILE = "data/ingested-text.txt"

def fetch_page(url):
    response = requests.get(url, timeout=10)
    ct = response.headers.get("Content-Type", "").lower()
    print(f"Content-Type: {ct}")
    if response.ok:
        if ct.startswith("text/html") or ct == "text/plain":
            response.encoding = response.apparent_encoding
            return response.text, "html"
        
        elif ct == "application/pdf":
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
        print(f"Error fetching {url}: {response.status_code}")
        return None, None

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
        file.close()

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
            else:
                print(f"Failed to extract text from {url}")

            out_file.write(text + "\n")
            time.sleep(0.5)
        out_file.close()

if __name__ == "__main__":
    ingest_urls()
    #page, type = fetch_page("https://www.mestojablonec.cz/data/files/0f/06a/0c97ba1185799a43d45306f839eed8abdac/zadost-boz-010325.docx")
    #text = clean_text(page)
    #print(text)