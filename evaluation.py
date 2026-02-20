################################################
### Soubor pro evaluaci chunků z chunking.py ###
################################################

import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_FILE = "data/chunked-text.jsonl"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_chunks():
    chunks = []
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def deduplicate_chunks(chunks, threshold=0.9):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    unique_indices = []
    used = set()

    for i in range(len(embeddings)):
        print(f"Processing chunk {i}")
        if i in used:
            continue

        unique_indices.append(i)
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        for j, sim in enumerate(sims):
            if sim > threshold:
                used.add(j)

    return [chunks[i] for i in unique_indices]

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    
    unique_chunks = deduplicate_chunks(chunks)
    print(f"After deduplication: {len(unique_chunks)} unique chunks.")