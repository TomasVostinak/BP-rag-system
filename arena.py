########################
### Soubor pro arénu ###
########################

import streamlit as st
import pandas as pd
import os
import json
import google.genai as genai
from collections import Counter
from embedding import evaluate_model, retrieve_chunks

RESULTS_FILE = "data/eval-results.jsonl"

def save_results(model_name, results):
    score = (
        0.4 * results["recall"][10]
        + 0.3 * results["ndcg"]
        + 0.3 * results["mrr"]
    )

    record = {
        "model": model_name,
        "score": score,
        "recall@10": results["recall"][10],
        "recall@20": results["recall"][20],
        "recall@30": results["recall"][30],
        "mrr": results["mrr"],
        "ndcg": results["ndcg"]
    }

    existing_models = set()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                existing_models.add(data["model"])

    if model_name in existing_models:
        return

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame()
    
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def highlight_shared(row):
    if row["chunk_id"] in duplicate_ids:
        return ["background-color: rgba(255,200,0,0.35)"] * len(row)
    return [""] * len(row)

####################
### Streamlit UI ###
####################

st.title("RAG Evaluation Arena")
st.write("Porovnání embedding modelů na datasetu otázek z webů města Jablonec nad Nisou.")

model_name = st.text_input(
    "Zadejte název embedding modelu (z HuggingFace)",
    value="intfloat/multilingual-e5-base"
)

if st.button("Spustit evaluaci"):
    try:
        with st.spinner("Počítám embeddingy a evaluaci..."):
            results = evaluate_model(model_name)

        save_results(model_name, results)

        st.success("Evaluace dokončena")
        st.rerun()

    except Exception as e:
        st.error("Chyba při evaluaci.")
        st.exception(e)

st.subheader("Leaderboard")

df = load_results()

if not df.empty:
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "placement", df.index + 1)

    st.dataframe(
        df,
        width='stretch',
        hide_index=True
    )
    st.download_button(
        "Stáhnout výsledky",
        df.to_csv(index=False),
        "results.csv"
    )
else:
    st.info("Zatím nejsou uložené žádné výsledky.")

st.divider()
st.header("Retrieval Arena")
st.write("Zadejte otázku a porovnejte výsledky různých embedding modelů. Chunky se žlutým pozadím se opakují ve více modelech.")

query = st.text_input("Otázka")

top_k = st.slider(
    "Kolik chunků zobrazit a použít",
    min_value=1,
    max_value=20,
    value=10
)

models = df["model"].tolist() if not df.empty else []

if st.button("Porovnat modely"):
    if not query:
        st.warning("Zadejte otázku.")
    else:
        with st.spinner("Vyhledávám relevantní chunky..."):
            model_results = {}
            all_chunk_ids = []

            for model in models:
                chunks = retrieve_chunks(model, query, top_k)
                model_results[model] = chunks
                for c in chunks:
                    all_chunk_ids.append(c["chunk_id"])

            counts = Counter(all_chunk_ids)

        duplicate_ids = {
            cid for cid, count in counts.items()
            if count > 1
        }

        st.session_state["model_results"] = model_results
        st.session_state["duplicate_ids"] = duplicate_ids
        st.session_state["query"] = query

if "model_results" in st.session_state:
    st.subheader("Retrieved chunks")

    model_results = st.session_state["model_results"]
    duplicate_ids = st.session_state["duplicate_ids"]
    query = st.session_state["query"]

    for model, chunks in model_results.items():
        st.markdown(f"#### {model}")
        rows = []

        for c in chunks:
            rows.append({
                "rank": c["rank"],
                "chunk_id": c["chunk_id"],
                "text": c["text"]
            })

        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.apply(highlight_shared, axis=1),
            width='stretch',
            hide_index=True
        )
    
    st.subheader("Generovaná odpověď od LLM (Gemini)")
    st.write("Chcete vygenerovat odpověď na otázku?")

    if st.button("Vygenerovat odpověď"):
        best_model = models[0]

        context = "\n\n".join(
            [c["text"] for c in model_results[best_model]]
        )

        with st.spinner("Generuji odpověď pomocí LLM..."):
            client = genai.Client()
            prompt = f"""
                    Seš občan města Jablonec nad Nisou. Krátce odpověz na otázku pouze na základě následujícího kontextu.

                    OTÁZKA:
                    {query}

                    KONTEXT:
                    {context}
                    """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.caption(f"Odpověď generována z chunků modelu: {best_model}")
        st.write(response.text)
