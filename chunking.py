##################################################
### Soubor pro chunkování textu z ingestion.py ###
##################################################

import json
from transformers import AutoTokenizer
import nltk
nltk.download("punkt_tab")

TEXT_FILE = "data/ingested-text.jsonl"
CHUNK_FILE = "data/chunked-text.jsonl"

CHUNK_SIZE = 300
OVERLAP = 50
MIN_CHUNK_SIZE = 50

tokenization_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

def split_sentences(text):
    return nltk.sent_tokenize(text, language="czech")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def take_last_tokens(text, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    trimmed_tokens = tokens[-max_tokens:]
    return tokenizer.decode(trimmed_tokens)

def chunk_text(sentences, chunk_size=CHUNK_SIZE, overlap=OVERLAP, min_chunk_size=MIN_CHUNK_SIZE):
    chunks = []
    current_text = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > chunk_size:
            continue

        if current_tokens + sentence_tokens > chunk_size:
            if current_tokens >= min_chunk_size:
                chunks.append({
                    "text": current_text.strip(),
                    "tokens": current_tokens
                })

            overlap_text = take_last_tokens(current_text, overlap)

            current_text = overlap_text + " " + sentence
            current_tokens = count_tokens(current_text)

        else:
            if current_text:
                current_text += " " + sentence
            else:
                current_text = sentence
            current_tokens += sentence_tokens

    if current_tokens >= min_chunk_size:
        chunks.append({
            "text": current_text.strip(),
            "tokens": current_tokens
        })

    return chunks

def process_text():
    with open(TEXT_FILE, "r", encoding="utf-8") as textfile:
        with open(CHUNK_FILE, "w", encoding="utf-8") as chunkfile:
            for line in textfile:
                record = json.loads(line)
                id = record["id"]
                url = record["url"]
                text = record["text"]

                sentences = split_sentences(text)
                chunks = chunk_text(sentences)

                for i, chunk in enumerate(chunks):
                    chunk_record = {
                        "doc_id": id,
                        "chunk_id": i + 1,
                        "url": url,
                        "tokens": chunk["tokens"],
                        "text": chunk["text"]
                    }
                    chunkfile.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_text()