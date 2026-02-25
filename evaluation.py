################################################
### Soubor pro evaluaci chunků z chunking.py ###
################################################

import json
from sentence_transformers import SentenceTransformer
import faiss
import google.genai as genai

client = genai.Client()

CHUNK_FILE = "data/chunked-text.jsonl"
OUTPUT_FILE = "data/qa-dataset.jsonl"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_chunks():
    chunks = []
    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks

def deduplicate_chunks(chunks, threshold=0.9):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    unique_indices = []
    used = set()

    print("Deduplicating chunks...")
    for i in range(len(embeddings)):
        if i in used:
            continue

        unique_indices.append(i)
        D, I = index.search(embeddings[i:i+1], len(embeddings))

        for j, score in zip(I[0], D[0]):
            if j != i and score > threshold:
                used.add(j)

    return [chunks[i] for i in unique_indices]

def is_informative(chunk):
    text = chunk["text"]

    if len(text) < 300:
        return False

    # příliš mnoho opakujících se slov
    words = text.split()

    if len(words) == 0:
        return False
    
    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.4:
        return False

    return True

def generate_questions(client, text, n=5):
    prompt = f"""
        Na základě následujícího textu vytvoř {n} konkrétních faktických otázek,
        na které lze odpovědět pouze z tohoto textu.

        Otázky vypiš jako JSON pole ve formátu:

        [
          {{"question": "..."}},
          {{"question": "..."}}
        ]

        Text:
        {text}
        """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def parse_questions(response_text):
    try:
        return json.loads(response_text)
    except:
        return []

def build_qa_dataset(chunks, client, output_file, limit=1000):
    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if count >= limit:
                break

            response_text = generate_questions(client, chunk["text"], n=5)
            questions = parse_questions(response_text)

            for q in questions:
                if count >= limit:
                    break

                record = {
                    "chunk_id": chunk["chunk_id"],
                    "source_url": chunk["url"],
                    "question": q["question"],
                    "answer": chunk["text"]
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    
    unique_chunks = deduplicate_chunks(chunks)
    print(f"After deduplication: {len(unique_chunks)} unique chunks.")

    informative_chunks = [chunk for chunk in unique_chunks if is_informative(chunk)]
    print(f"After filtering: {len(informative_chunks)} informative chunks.")

    build_qa_dataset(informative_chunks, client, OUTPUT_FILE)
    print(f"QA dataset saved to {OUTPUT_FILE}.")