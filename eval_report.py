"""
Runs your test questions through BM25, Dense and Hybrid and writes two files:
  report_table.md    -> table skeleton for your README
  report_details.md  -> the top chunk text per mode, to help you judge which mode was best
Retrieval only: no API key needed.  Run:  python eval_report.py
Edit QUESTIONS below so they match YOUR documents.
"""
import os
os.environ.setdefault("USE_TF", "0")

import rag as app

QUESTIONS = [
    ("Q1 (exact rare keyword)", "What is PhishShield?"),
    ("Q2 (paraphrase, no shared words)", "What kind of things do you like making that protect everyday people?"),
    ("Q3 (factual)", "Which university do you study at and in which semester are you?"),
    ("Q4 (opinion)", "What do you think about using the same password everywhere?"),
    ("Q5 (out of scope)", "What is your favorite football team?"),
]
K = 3


def short(idx):
    return app.CHUNKS[idx]["chunk_id"]


def run(encode=None):
    rows, details = [], []
    for label, q in QUESTIONS:
        bm = app.bm25_search(q, K)
        de = app.dense_search(q, K, encode)
        hy = app.hybrid_search(q, K, encode)
        fmt = lambda pairs: "<br>".join(f"{short(p[0])} ({p[1]:.3f})" for p in pairs)
        rows.append(f"| {label}: {q} | {fmt(bm)} | {fmt(de)} | {fmt(hy)} | TODO: which mode was best and why |")
        details.append(f"## {label}\n**{q}**\n")
        for name, res in (("BM25", bm), ("Dense", de), ("Hybrid", hy)):
            top = res[0][0]
            details.append(f"- **{name} top-1** `{short(top)}`: {app.CHUNKS[top]['text'][:220]}...")
        details.append("")
    header = ("| Question | BM25 result | Dense result | Hybrid result | Best mode and why |\n"
              "|---|---|---|---|---|\n")
    open("report_table.md", "w", encoding="utf-8").write(header + "\n".join(rows) + "\n")
    open("report_details.md", "w", encoding="utf-8").write("\n".join(details))
    print("Wrote report_table.md and report_details.md")


if __name__ == "__main__":
    run()
