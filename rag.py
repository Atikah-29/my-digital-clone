"""
Core RAG code for Atikah's digital clone: chunks -> BM25 + dense -> RRF fusion -> Groq LLM.
Needs chunks.json and embeddings.npy in the same folder (made by build_index.py).
Secret: GROQ_API_KEY (environment variable or Streamlit secret, never in code).
"""
import json
import os
import re

os.environ.setdefault("USE_TF", "0")  # avoid TensorFlow/Keras clashes

import numpy as np
import requests
from rank_bm25 import BM25Okapi

# ---------------- Config ----------------
OWNER = "Atikah"
TONE = "friendly, calm and a little witty"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [m for m in [
    os.environ.get("GROQ_MODEL"),
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
] if m]
TOP_K = 4
RRF_K = 60

# ---------------- Load data ----------------
with open("chunks.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)
EMB = np.load("embeddings.npy")  # L2-normalized, precomputed with sentence-transformers
EMB_STATUS = "Using the saved embeddings.npy."

STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was were be been
am do does did i me my you your he she it we they this that these those what which who how why when
where about can could would should will have has had not no yes so than then there their its""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


BM25 = BM25Okapi([tokenize(c["text"]) for c in CHUNKS])

# ---------------- Query encoder ----------------
_encoder = None


def _verify_embeddings(enc):
    """If the query encoder disagrees with the saved embeddings (different backend), re-embed the chunks
    with the same encoder so that query and chunk vectors always match."""
    global EMB, EMB_STATUS
    v = enc(CHUNKS[0]["text"])
    sim = float(v @ EMB[0])
    if sim >= 0.98:
        EMB_STATUS = f"Saved embeddings verified (match {sim:.3f})."
    else:
        EMB = np.array([enc(c["text"]) for c in CHUNKS], dtype="float32")
        EMB_STATUS = f"Saved embeddings differed (match {sim:.3f}), re-embedded chunks at startup."


def get_encoder():
    """question -> normalized vector. sentence-transformers if installed (laptop), else fastembed (ONNX,
    same all-MiniLM-L6-v2 model, much lower memory for free cloud hosting)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    enc = None
    if os.environ.get("EMBED_BACKEND", "auto") != "onnx":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBED_MODEL)
            enc = lambda q: model.encode([q], normalize_embeddings=True)[0]
        except ImportError:
            pass
    if enc is None:
        from fastembed import TextEmbedding
        onnx_model = TextEmbedding(EMBED_MODEL)

        def enc(q):
            v = np.asarray(list(onnx_model.embed([q]))[0], dtype="float32")
            return v / np.linalg.norm(v)
    _verify_embeddings(enc)
    _encoder = enc
    return _encoder


# ---------------- Retrieval ----------------
def bm25_search(query, k=TOP_K):
    scores = BM25.get_scores(tokenize(query))
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def dense_search(query, k=TOP_K, encode=None):
    encode = encode or get_encoder()
    scores = EMB @ encode(query)  # cosine similarity (normalized vectors)
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def rrf_fuse(rank_lists, k=RRF_K):
    """Reciprocal Rank Fusion: score = sum 1/(k + rank), rank starts at 1."""
    fused = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def hybrid_search(query, k=TOP_K, encode=None, pool=10):
    b = bm25_search(query, pool)
    d = dense_search(query, pool, encode)
    fused = rrf_fuse([[i for i, _ in b], [i for i, _ in d]])[:k]
    b_rank = {i: r for r, (i, _) in enumerate(b, 1)}
    d_rank = {i: r for r, (i, _) in enumerate(d, 1)}
    return [(i, s, b_rank.get(i), d_rank.get(i)) for i, s in fused]


def retrieve(query, mode):
    """Returns a list of dicts: idx, score, note."""
    if mode == "BM25 only":
        return [{"idx": i, "score": s, "note": ""} for i, s in bm25_search(query)]
    if mode == "Dense only":
        return [{"idx": i, "score": s, "note": ""} for i, s in dense_search(query)]
    return [{"idx": i, "score": s, "note": f"BM25 rank: {br or '-'} | Dense rank: {dr or '-'}"}
            for i, s, br, dr in hybrid_search(query)]


# ---------------- Generation (Groq) ----------------
PERSONA = f"""You are the AI clone of {OWNER}, a 7th semester Computer Science student at the
University of Management and Technology (UMT) from Sialkot, Pakistan. Cyber security is my side interest
and cyber forensics is my long term goal.
Speak in the first person, in a {TONE} tone. Keep answers short and natural.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of {OWNER}.

CONTEXT:
{{context}}

QUESTION:
{{question}}"""


LAST_MODEL = ""
_SKIP = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard", "prompt-guard", "moderation")


def list_groq_models(key):
    """Ask Groq which chat models this key can use right now."""
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
    ids = [i for i in ids if not any(w in i.lower() for w in _SKIP)]
    pref = ("llama-3.3", "llama-3.1", "gpt-oss", "qwen", "llama")
    return sorted(ids, key=lambda i: next((n for n, p in enumerate(pref) if p in i.lower()), len(pref)))


def _groq_call(model, prompt, key):
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.3, "max_tokens": 400},
        timeout=60)
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"HTTP {r.status_code}, non-JSON body: {r.text[:100]!r}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {str(data)[:160]}")
    return data["choices"][0]["message"]["content"].strip()


def groq_generate(prompt, key):
    """Try the preferred models; if none exist for this key, ask Groq for its live model list."""
    global LAST_MODEL
    errors = []
    for model in GROQ_MODELS:
        try:
            text = _groq_call(model, prompt, key)
            LAST_MODEL = model
            return text
        except Exception as e:
            errors.append(f"{model}: {str(e)[:120]}")
    try:
        for model in [m for m in list_groq_models(key) if m not in GROQ_MODELS][:4]:
            try:
                text = _groq_call(model, prompt, key)
                LAST_MODEL = model
                return text
            except Exception as e:
                errors.append(f"{model}: {str(e)[:120]}")
    except Exception as e:
        errors.append(f"model list: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(" || ".join(errors))


def templated_answer(hits, reason):
    if not hits:
        return "I don't have that in my notes."
    return f"({reason})\n\nFrom my notes: {CHUNKS[hits[0]['idx']]['text'][:500]}..."


def generate(question, hits):
    context = "\n\n".join(f"[{CHUNKS[h['idx']]['chunk_id']}]\n{CHUNKS[h['idx']]['text']}" for h in hits)
    prompt = PERSONA.format(context=context, question=question)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return templated_answer(hits, "No GROQ_API_KEY set, showing the best passage instead.")
    try:
        return groq_generate(prompt, key)
    except Exception as e:
        return templated_answer(hits, "LLM call failed: " + str(e)[:300])
