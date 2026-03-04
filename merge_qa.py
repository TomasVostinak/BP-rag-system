import json

QUESTIONS_FILE = "data/otazky.jsonl"
CHUNKS_FILE = "data/final-chunks.jsonl"
OUTPUT_FILE = "data/qa-dataset.jsonl"

def load_chunks():
    chunk_map = {}

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_map[record["chunk_id"]] = record

    return chunk_map

def merge_files():
    chunk_map = load_chunks()

    merged_count = 0
    missing_count = 0

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as qf, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for line in qf:
            question_record = json.loads(line)
            chunk_id = question_record["chunk_id"]

            if chunk_id not in chunk_map:
                print(chunk_id)
                missing_count += 1
                continue

            chunk_record = chunk_map[chunk_id]

            merged_record = {
                "chunk_id": chunk_id,
                "doc_id": chunk_record["doc_id"],
                "url": chunk_record["url"],
                "question": question_record["question"],
                "relevant_text": chunk_record["text"]
            }

            out.write(json.dumps(merged_record, ensure_ascii=False) + "\n")
            merged_count += 1

    print(f"Merged: {merged_count}")
    print(f"Missing chunk_ids: {missing_count}")

if __name__ == "__main__":
    merge_files()