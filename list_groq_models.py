"""Shows which models your Groq key can use. Run:  python list_groq_models.py"""
import os
import rag

key = os.environ.get("GROQ_API_KEY")
print("Key set:", bool(key))
try:
    for m in rag.list_groq_models(key):
        print(" ", m)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300])
