"""
Step 2: chunk your documents and precompute dense embeddings.
Run once on your laptop:  python build_index.py
Outputs: chunks.json and embeddings.npy (commit both to your Space).
"""
import glob
import json
import os
import re

os.environ["USE_TF"] = "0"  # skip TensorFlow/Keras (avoids the Keras 3 error)

import numpy as np

DATA_DIR = "data"
CHUNK_WORDS = 250      # target chunk size (assignment: about 200 to 300)
OVERLAP_WORDS = 40     # words shared between neighbouring chunks
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir=DATA_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if path.lower().endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs


def clean(text):
    text = re.sub(r"^>\s*(DUMMY DATA|TEMPLATE).*$", "", text, flags=re.MULTILINE)  # drop leftover notes
    return text


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks():
    all_chunks = []
    for filename, text in load_documents():
        for i, piece in enumerate(chunk_text(clean(text))):
            all_chunks.append({
                "chunk_id": f"{filename}::{i}",
                "source": filename,
                "text": piece,
            })
    return all_chunks


def warn_placeholders():
    found = 0
    for filename, text in load_documents():
        for n, line in enumerate(text.splitlines(), 1):
            if "[EDIT" in line or "____" in line:
                found += 1
                print(f"  PLACEHOLDER {filename} line {n}: {line.strip()[:70]}")
    if found:
        print(f"WARNING: {found} placeholder line(s) still need your real answers (see above).")
    else:
        print("No placeholders left.")


if __name__ == "__main__":
    warn_placeholders()
    chunks = build_chunks()
    print(f"Documents: {len({c['source'] for c in chunks})}")
    print(f"Chunks:    {len(chunks)}")
    for src in sorted({c['source'] for c in chunks}):
        n = sum(1 for c in chunks if c["source"] == src)
        print(f"  {src}: {n} chunk(s)")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks],
                       normalize_embeddings=True, show_progress_bar=True)
    np.save("embeddings.npy", np.asarray(emb, dtype="float32"))
    print("Saved chunks.json and embeddings.npy, embeddings shape:", emb.shape)
