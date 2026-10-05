import os
from transformers import pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf = pipeline("text-classification",
               model=os.path.join(BASE, "models", "distilbert_model"),
               top_k=None)

texte = [
    "This product is excellent, highly recommended!",
    "Terrible quality, it broke after two days.",
    "It's okay, nothing special but it works.",
]

for t in texte:
    res = clf(t)
    if isinstance(res[0], list):
        res = res[0]
    best = max(res, key=lambda r: r["score"])
    print(f"{t}\n  -> {best['label']} ({best['score']:.2f})\n")