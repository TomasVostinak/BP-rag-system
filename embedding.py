###################################
### Soubor pro embedding otázek ###
###################################

import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/final-chunks.jsonl"
QA_DATASET = "data/qa-dataset.jsonl"
INDEX_DIR = "data/index_cache"

TOP_K_VALUES = [10, 20, 30]

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

def get_index(model, model_name, texts):
    os.makedirs(INDEX_DIR, exist_ok=True)

    index_path = os.path.join(
        INDEX_DIR,
        model_name.replace("/", "_") + ".faiss"
    )

    if os.path.exists(index_path):
        print("Model retrieved from cache.")
        return faiss.read_index(index_path)

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

    faiss.write_index(index, index_path)

    return index

def compute_ndcg(rank):
    if rank is None:
        return 0

    return 1 / np.log2(rank + 1)

def evaluate_model(model_name):
    model = SentenceTransformer(model_name)

    texts, chunk_ids = load_chunks()
    questions, relevant_ids = load_dataset()

    print("Creating FAISS index...")

    index = get_index(model, model_name, texts)
    chunk_ids_to_index = {cid: i for i, cid in enumerate(chunk_ids)}
    max_k = max(TOP_K_VALUES)

    recall = {k: 0 for k in TOP_K_VALUES}
    mrr = 0
    ndcg = 0

    total = len(questions)

    question_embeddings = model.encode(
        questions,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    faiss.normalize_L2(question_embeddings)

    print("Running evaluation...")

    for i, q_emb in enumerate(question_embeddings):
        q_emb = q_emb.reshape(1, -1)

        scores, indices = index.search(q_emb, max_k)
        retrieved = indices[0]

        relevant_id = relevant_ids[i]

        if relevant_id not in chunk_ids_to_index:
            continue

        true_index = chunk_ids_to_index[relevant_id]
        rank = None

        for j, idx in enumerate(retrieved):
            if idx == true_index:
                rank = j + 1
                break

        if rank is not None:
            for k in TOP_K_VALUES:
                if rank <= k:
                    recall[k] += 1
            
            mrr += 1 / rank
            ndcg += compute_ndcg(rank)
    
    print("Evaluation complete")

    results = {
        "recall": {k: recall[k] / total for k in TOP_K_VALUES},
        "mrr": mrr / total,
        "ndcg": ndcg / total
    }

    return results
