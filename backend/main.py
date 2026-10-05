import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

from backend import database

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "distilbert_model")

app = FastAPI(title="Sentiment Analysis API")
classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
database.init_db()


class TextIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: TextIn):
    text = item.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Textul nu poate fi gol.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Textul este prea lung (maxim 5000 de caractere).")

    start = time.perf_counter()
    res = classifier(text, truncation=True, max_length=256)
    time_ms = int((time.perf_counter() - start) * 1000)

    if isinstance(res[0], list):
        res = res[0]
    scores = {r["label"]: round(r["score"], 4) for r in res}
    best = max(scores, key=scores.get)

    database.save_analysis(text, best, scores[best], time_ms)
    return {"label": best, "confidence": scores[best], "scores": scores, "time_ms": time_ms}


@app.get("/history")
def history(limit: int = 20):
    return database.get_history(min(max(limit, 1), 200))


@app.delete("/history")
def delete_history():
    database.clear_history()
    return {"status": "cleared"}


# Trebuie sa fie ULTIMA linie, altfel acopera celelalte rute
app.mount("/", StaticFiles(directory=os.path.join(BASE, "frontend"), html=True), name="frontend")