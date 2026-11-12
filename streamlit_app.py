"""Atikah's Digital Clone: Streamlit UI. Run locally:  streamlit run streamlit_app.py"""
import os

import streamlit as st

try:  # on Streamlit Cloud the key lives in Secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

import rag

st.set_page_config(page_title="Atikah's Digital Clone", page_icon="🤖")
st.title("Ask Atikah's Digital Clone")
st.warning("⚠️ This is an AI clone, not the real person.")
st.write("It answers only from my own notes. If it doesn't know, it will say so.")

question = st.text_input("Your question", placeholder="e.g. What was your final year project?")
mode = st.radio("Retrieval mode", ["BM25 only", "Dense only", "Hybrid"], index=2, horizontal=True)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        hits = rag.retrieve(question.strip(), mode)
        answer = rag.generate(question.strip(), hits)
    st.subheader("Answer")
    st.write(answer)
    if rag.LAST_MODEL and not answer.startswith("("):
        st.caption(f"Answered by {rag.LAST_MODEL} via Groq")
    st.subheader("Retrieved chunks")
    for rank, h in enumerate(hits, 1):
        c = rag.CHUNKS[h["idx"]]
        with st.expander(f"#{rank} | {c['source']} | {c['chunk_id']} | score: {h['score']:.4f}", expanded=(rank == 1)):
            if h["note"]:
                st.caption(h["note"])
            st.write(c["text"])
elif st.button("Ask", key="empty") and not question.strip():
    st.info("Please type a question.")

st.sidebar.markdown("### How it works")
st.sidebar.write("BM25 keyword search, dense embedding search (all-MiniLM-L6-v2) and Reciprocal Rank Fusion, "
                 "then an LLM answers in the first person using only the retrieved chunks.")
st.sidebar.caption(rag.EMB_STATUS)
