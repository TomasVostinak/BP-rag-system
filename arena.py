########################
### Soubor pro arénu ###
########################

import streamlit as st
import pandas as pd
import os
import json
from embedding import evaluate_model

RESULTS_FILE = "data/eval-results.jsonl"

def save_results(model_name, results):
    record = {
        "model": model_name,
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

### Streamlit UI ###

st.title("RAG Evaluation Arena")
st.write(
"""
Porovnání embedding modelů na datasetu otázek z webů města Jablonec nad Nisou.
"""
)

model_name = st.text_input(
    "Zadej název embedding modelu (z HuggingFace)",
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
    df = df.sort_values("recall@10", ascending=False)
    st.dataframe(
        df,
        use_container_width=True
    )
    st.download_button(
        "Stáhnout výsledky",
        df.to_csv(index=False),
        "results.csv"
    )
else:
    st.info("Zatím nejsou uložené žádné výsledky.")
