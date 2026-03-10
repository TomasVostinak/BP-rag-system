###################################
### Soubor pro embedding otázek ###
###################################

import json
import faiss
from sentence_transformers import SentenceTransformer
import tqdm

CHUNK_FILE = "data/final-chunks.jsonl"
QA_DATASET = "data/qa-dataset.jsonl"
TOP_K_VALUES = [10, 20, 30]

embedding_model = "intfloat/multilingual-e5-base"
model = SentenceTransformer(embedding_model)

def load_chunks():
    texts = []
    chunk_ids = []
    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)

            texts.append("passage: " + data["text"])
            chunk_ids.append(data["chunk_id"])
    
    return texts, chunk_ids

def load_dataset():
    questions = []
    relevant_ids = []

    with open(QA_DATASET, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)

            questions.append("query: " + data["question"])
            relevant_ids.append(data["chunk_id"])

    return questions, relevant_ids

def create_index(model, texts):
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index

def evaluate():
    pass