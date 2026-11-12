# Atikah's Digital Clone (Hybrid RAG)

> **This is an AI clone, not the real person.**

A chatbot that answers questions as me, using only documents I wrote about myself. It uses Retrieval-Augmented Generation with BM25 keyword search, dense embedding search, and Reciprocal Rank Fusion (RRF). The UI is built with Gradio.

**Live demo:** PASTE_YOUR_RENDER_URL_HERE (hosted on Render; the free server sleeps when idle, so the first load can take about a minute).

**Note on hosting:** My Hugging Face account is new, and Hugging Face does not allow Gradio Spaces on new free accounts, so, as suggested by the instructor, the source code is kept in this repository and the Gradio app is deployed on Render. Render's free tier has only 512 MB of RAM, so the live app encodes questions with the same `all-MiniLM-L6-v2` model through ONNX (`fastembed`) instead of PyTorch; the document embeddings were precomputed with sentence-transformers and saved in `embeddings.npy`.

## How to run

```
pip install -r requirements-local.txt
python build_index.py          # chunks data/ and saves chunks.json + embeddings.npy
set GITHUB_TOKEN=...           # or GEMINI_API_KEY (never commit keys)
python app.py                  # opens http://127.0.0.1:7860
```

Without an API key the app still works and shows the best retrieved passage instead of an LLM answer.

## Pipeline

1. **Chunk:** about 250-word windows with 40-word overlap; each chunk keeps its source file and chunk ID.
2. **Index twice:** BM25 (`rank_bm25`) and `all-MiniLM-L6-v2` embeddings (precomputed and saved in `embeddings.npy`, cosine similarity with numpy, no vector database).
3. **Retrieve:** BM25 only, Dense only, or Hybrid.
4. **Fuse:** Reciprocal Rank Fusion, `score = sum(1 / (60 + rank))`.
5. **Generate:** an LLM gets the top chunks and a persona prompt: first person, only from context, and "I don't have that in my notes." when the answer is missing.
6. **Show your work:** the UI lists each retrieved chunk with its source, chunk ID and score.

The API key is read from an environment variable and is never stored in the code.

---

# Report

## 1. Knowledge base and why

[EDIT] I used 6 documents (about X words): `bio.md` and `cv.md` (facts), `opinions.md` (personality and views), `faq.md` (first-person FAQ), `project.md` (long-form write-up of PhishShield) and `study_habits.md`. I chose them because they cover facts, opinions and long-form text, so the clone can answer different kinds of questions and I can judge the answers myself. The project name and tool names give BM25 exact keywords to match, and the plain descriptive passages give dense search something to match on paraphrases.

## 2. Test questions: BM25 vs Dense vs Hybrid

[Paste the contents of `report_table.md` here after running `python eval_report.py`, then fill in the last column.]

Observations to write (one or two sentences each):
- Q1 (exact keyword): [which mode won and why, e.g. BM25 matches the rare word directly]
- Q2 (paraphrase): [which mode won and why, e.g. dense matches meaning without shared words]
- Hybrid: [did it change the ranking? when did it help or hurt?]

## 3. Failure case

[EDIT] Question asked: ... What the clone answered: ... Why it went wrong: ... (examples: the answer was split across two chunks; the fact was missing from my documents; a chunk started mid-answer because splitting is by word count.)

## 4. Improvement for next time

[EDIT] e.g. a cross-encoder reranker over the hybrid results, splitting chunks by FAQ question instead of by word count, or a small router that detects out-of-scope questions.

## 5. Fusion method note

I implemented Reciprocal Rank Fusion. The alternative is a weighted score, `alpha * dense + (1 - alpha) * bm25`, after min-max normalizing each score. RRF needs no tuning and ignores the different score scales of BM25 and cosine similarity, which is why I chose it.
