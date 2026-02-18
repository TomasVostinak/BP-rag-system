################################################
### Soubor pro evaluaci chunků z chunking.py ###
################################################

import json

CHUNK_FILE = "data/chunked-text.jsonl"

def load_chunks():
    chunks = []
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

if __name__ == "__main__":
    pass