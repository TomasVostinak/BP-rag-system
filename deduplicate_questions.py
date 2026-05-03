#####################################
### Soubor pro deduplikaci otázek ###
#####################################

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INPUT_QUESTIONS_FILE = "data/otazky.jsonl"
OUTPUT_FILTERED_QUESTIONS_FILE = "data/filtered-otazky.jsonl"
SIMILARITY_THRESHOLD = 0.9
MODEL_NAME = "intfloat/multilingual-e5-base"

def load_questions(file_path):
    questions_data = []
    if not os.path.exists(file_path):
        print(f"Chyba: Vstupní soubor nebyl nalezen na {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            questions_data.append(json.loads(line))
    return questions_data

def save_questions(questions_data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for q_item in questions_data:
            f.write(json.dumps(q_item, ensure_ascii=False) + "\n")

def deduplicate_questions(questions_data, model_name, threshold):
    if not questions_data:
        print("Žádné otázky k deduplikaci.")
        return []

    print(f"Načítám Sentence Transformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Chyba při načítání modelu {model_name}: {e}")
        return questions_data

    question_texts = [q_item["question"] for q_item in questions_data]

    print(f"Generuji embeddingy pro {len(question_texts)} otázek...")
    embeddings = model.encode(
        ["query: " + text for text in question_texts],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    unique_questions = []
    seen_indices = set()

    print(f"Deduplikuji otázky s prahem podobnosti: {threshold}...")
    for i, embedding_i in enumerate(embeddings):
        if i in seen_indices:
            continue

        unique_questions.append(questions_data[i])
        seen_indices.add(i)

        D, I = index.search(embedding_i.reshape(1, -1), len(questions_data))
        
        for j_idx, score in zip(I[0], D[0]):
            if score >= threshold:
                seen_indices.add(j_idx)
    
    return unique_questions

if __name__ == "__main__":
    all_questions = load_questions(INPUT_QUESTIONS_FILE)

    print(f"Načteno {len(all_questions)} otázek ze souboru {INPUT_QUESTIONS_FILE}")

    unique_questions_list = deduplicate_questions(all_questions, MODEL_NAME, SIMILARITY_THRESHOLD)
    print(f"Po deduplikaci s prahem {SIMILARITY_THRESHOLD} zbývá {len(unique_questions_list)} unikátních otázek.")

    save_questions(unique_questions_list, OUTPUT_FILTERED_QUESTIONS_FILE)
    print(f"Unikátní otázky uloženy do souboru {OUTPUT_FILTERED_QUESTIONS_FILE}")