##################################################
### Soubor pro chunkování textu z ingestion.py ###
##################################################

import json
from transformers import AutoTokenizer

TEXT_FILE = "data/ingested-text.jsonl"
CHUNK_FILE = "data/chunked-text.jsonl"

MAX_CHARS = 900
OVERLAP = 150
MIN_CHARS = 200

tokenization_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

def calculate_chars_per_token():
    total_chars = 0
    total_tokens = 0

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            text = record["text"]

            tokens = tokenizer.encode(text, add_special_tokens=False)

            total_chars += len(text)
            total_tokens += len(tokens)

    print("Chars per token:", total_chars / total_tokens)

def chunk_by_chars(text, min_chars=MIN_CHARS, max_chars=MAX_CHARS, overlap=OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_chars
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= min_chars:
            token_count = len(
                tokenizer.encode(chunk_text, add_special_tokens=False)
            )

            chunks.append({
                "text": chunk_text,
                "chars": len(chunk_text),
                "tokens": token_count
            })

        start = end - overlap

    return chunks

def process_text():
    global_chunk_id = 0
    with open(TEXT_FILE, "r", encoding="utf-8") as textfile:
        with open(CHUNK_FILE, "w", encoding="utf-8") as chunkfile:
            for line in textfile:
                record = json.loads(line)
                id = record["id"]
                url = record["url"]
                text = record["text"]

                chunks = chunk_by_chars(text)

                for chunk in chunks:
                    global_chunk_id += 1

                    chunk_record = {
                        "doc_id": id,
                        "chunk_id": global_chunk_id,
                        "url": url,
                        "chars": chunk["chars"],
                        "tokens": chunk["tokens"],
                        "text": chunk["text"]
                    }
                    chunkfile.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_text()