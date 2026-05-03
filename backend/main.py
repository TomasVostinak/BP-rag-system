##########################################
### Soubor pro hlavní FastAPI aplikaci ###
##########################################

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import pandas as pd
from collections import Counter
import google.genai as genai
import uvicorn
import random

from embedding import evaluate_model, retrieve_chunks 

BASE_DIR = os.path.dirname(__file__) # 'backend'

DATA_DIR = os.path.join(BASE_DIR, "..", "data") # 'data'
RESULTS_FILE = os.path.join(DATA_DIR, "eval-results.jsonl")
CHUNK_FILE = os.path.join(DATA_DIR, "final-chunks.jsonl")
AB_TEST_RESULTS_FILE = os.path.join(DATA_DIR, "ab-test-results.jsonl")

STATIC_DIR = os.path.join(BASE_DIR, "static") # 'backend/static'

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

app = FastAPI(
    title="RAG Evaluation Arena API",
    description="API pro porovnávání embedding modelů na datasetu otázek z webů města Jablonec nad Nisou.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# modely

class EvalRequest(BaseModel):
    model_name: str

class QueryRequest(BaseModel):
    query: str
    top_k: int
    models_to_compare: list[str]

class LLMRequest(BaseModel):
    query: str
    context: str

class ABTestGetPairRequest(BaseModel):
    query: str
    top_k: int

class ABTestSubmitRequest(BaseModel):
    query: str
    model_a_name: str
    model_b_name: str
    winner: str

# funkce

def save_results(model_name: str, results: dict):
    score = (
        0.6 * results["recall"][10]
        + 0.4 * results["mrr"]
    )

    record = {
        "model": model_name,
        "score": score,
        "recall@10": results["recall"][10],
        "recall@20": results["recall"][20],
        "recall@30": results["recall"][30],
        "mrr": results["mrr"]
    }

    existing_records = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            existing_records = [json.loads(line) for line in file]

    if any(r["model"] == model_name for r in existing_records):
        print(f"Model '{model_name}' již existuje v leaderboardu, přeskočeno ukládání.")
        return

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Výsledky pro model '{model_name}' uloženy.")

def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=["placement", "model", "score", "recall@10", "recall@20", "recall@30", "mrr"])
    
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "placement", df.index + 1)
    return df

def get_chunk_texts_map():
    chunk_texts_map = {}
    if os.path.exists(CHUNK_FILE):
        with open(CHUNK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                chunk_data = json.loads(line)
                chunk_texts_map[str(chunk_data["chunk_id"])] = chunk_data["text"]
    return chunk_texts_map

async def generate_llm_response(query: str, context: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API klíč není nastaven, nelze generovat odpovědi.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
            Jsi asistentem, který pomáhá občanům města Jablonec nad Nisou. Krátce a stručně odpovězte na OTÁZKU POUZE na základě poskytnutého KONTEXTU. Pokud kontext neobsahuje dostatek informací pro zodpovězení otázky, odpovězte, že "Nemám dostatek informací z poskytnutého kontextu k zodpovězení této otázky.".

            OTÁZKA:
            {query}

            KONTEXT:
            {context}
            """
    
    print(f"Generuji odpověď LLM pro dotaz: {query}...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("Odpověď LLM vygenerována.")
    return response.text

# endpointy

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail=f"Frontend soubor {html_path} nenalezen. Ujistěte se, že existuje v adresáři '{STATIC_DIR}'.")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/evaluate")
async def evaluate_model_endpoint(req: EvalRequest):
    try:
        model_name = req.model_name
        if model_name in load_results()["model"].tolist():
            return JSONResponse(content={"message": f"Model '{model_name}' již byl evaluován."}, status_code=200)

        print(f"Evaluace modelu: {model_name}")
        results = evaluate_model(model_name) 
        save_results(model_name, results)
        return JSONResponse(content={"message": f"Evaluace pro model '{model_name}' dokončena a uložena."})
    except Exception as e:
        print(f"Chyba při evaluaci modelu '{req.model_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Chyba při evaluaci: {e}")

@app.get("/api/leaderboard")
async def get_leaderboard_endpoint():
    df = load_results()
    return JSONResponse(content=df.to_dict(orient="records"))

@app.post("/api/compare")
async def compare_models_endpoint(req: QueryRequest):
    try:
        query = req.query
        top_k = req.top_k
        models_to_compare = req.models_to_compare

        if not query:
            raise HTTPException(status_code=400, detail="Dotaz nesmí být prázdný.")
        if not models_to_compare:
             raise HTTPException(status_code=400, detail="Vyberte alespoň jeden model pro porovnání.")

        model_results = {}
        all_chunk_ids = []

        chunk_texts_map = get_chunk_texts_map()

        for model_name in models_to_compare:
            chunks = retrieve_chunks(model_name, query, top_k)
            for c in chunks:
                c["chunk_id"] = str(c["chunk_id"])
                c["full_text"] = chunk_texts_map.get(c["chunk_id"], "")
            model_results[model_name] = chunks
            for c in chunks:
                all_chunk_ids.append(c["chunk_id"])

        counts = Counter(all_chunk_ids)
        duplicate_ids = [cid for cid, count in counts.items() if count > 1]
        
        return JSONResponse(content={
            "query": query,
            "model_results": model_results,
            "duplicate_ids": duplicate_ids
        })
    except Exception as e:
        print(f"Chyba při retrievalu chunků: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba při vyhledávání: {e}")

@app.post("/api/generate")
async def generate_llm_response_endpoint(req: LLMRequest):
    try:
        response_text = await generate_llm_response(req.query, req.context)
        print(f"Generuji odpověď LLM pro dotaz: {req.query}...")
        print("Odpověď LLM vygenerována.")
        return JSONResponse(content={"answer": response_text})

    except Exception as e:
        print(f"Chyba při generování odpovědi LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba při generování odpovědi: {e}")
    
@app.post("/api/abtest/get_pair")
async def get_abtest_pair_endpoint(req: ABTestGetPairRequest):
    df = load_results()
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=400, detail="Nedostatek evaluovaných modelů pro A/B testování.")

    sorted_models = df["model"].tolist()
    
    if len(sorted_models) == 2:
        idx = 0
    else:
        idx = random.randint(0, len(sorted_models) - 2)
    
    model_a_name_internal = sorted_models[idx]
    model_b_name_internal = sorted_models[idx + 1]

    if random.random() < 0.5:
        display_order = ['model_a', 'model_b']
    else:
        display_order = ['model_b', 'model_a']

    chunks_a = retrieve_chunks(model_a_name_internal, req.query, req.top_k)
    context_a = "\n\n".join([c["text"] for c in chunks_a])
    response_a = await generate_llm_response(req.query, context_a)

    chunks_b = retrieve_chunks(model_b_name_internal, req.query, req.top_k)
    context_b = "\n\n".join([c["text"] for c in chunks_b])
    response_b = await generate_llm_response(req.query, context_b)
    
    return JSONResponse(content={
        "model_a_name_internal": model_a_name_internal,
        "model_b_name_internal": model_b_name_internal,
        "response_a": response_a,
        "response_b": response_b,
        "display_order": display_order
    })

@app.post("/api/abtest/submit_result")
async def submit_abtest_result(req: ABTestSubmitRequest):
    try:
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "query": req.query,
            "model_a_name": req.model_a_name,
            "model_b_name": req.model_b_name,
            "winner": req.winner
        }
        with open(AB_TEST_RESULTS_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return JSONResponse(content={"message": "Výsledek A/B testu uložen."})
    except Exception as e:
        print(f"Chyba při ukládání výsledku A/B testu: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba při ukládání: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
