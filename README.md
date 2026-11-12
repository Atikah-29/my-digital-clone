# Atikah's Digital Clone (Hybrid RAG)

> **This is an AI clone, not the real person.**

A chatbot that answers questions as me, using only documents I wrote about myself. It uses Retrieval-Augmented Generation with BM25 keyword search, dense embedding search, and Reciprocal Rank Fusion (RRF). The UI is built with Streamlit.

**Live demo:** https://my-clone-mhk8unzxnw5qpbhracuurd.streamlit.app/

## How to run locally

```bash
pip install -r requirements-local.txt
python build_index.py          # chunks data/ and saves chunks.json + embeddings.npy
$env:GROQ_API_KEY="gsk_your_real_key_here"
python -m streamlit run streamlit_app.py