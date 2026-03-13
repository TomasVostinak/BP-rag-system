########################
### Soubor pro arénu ###
########################

import streamlit as st
from embedding import evaluate_model

CHUNK_FILE = "data/final-chunks.jsonl"

MODELS = {
    "intfloat/multilingual-e5-base"
}

st.title("RAG Evaluation Arena")
st.write(
"""
Porovnání embedding modelů na datasetu otázek z webů českých měst.
"""
)

model_name = st.selectbox(
    "Vyber embedding model",
    MODELS
)

if st.button("Spustit evaluaci"):

    with st.spinner("Počítám embeddingy a evaluaci..."):

        results = evaluate_model(model_name)

    st.success("Evaluace dokončena")

    st.subheader("Výsledky")

    for k, value in results["recall"].items():
        st.write(f"Recall@{k}: {value:.3f}")

    st.write(f"MRR: {results['mrr']:.3f}")
    st.write(f"nDCG: {results['ndcg']:.3f}")