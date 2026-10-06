import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
import io
import pandas as pd
from fastapi import UploadFile, File

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

MAX_ROWS = 500


@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Fișierul trebuie să fie de tip .csv")

    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Nu am putut citi fișierul CSV.")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="Fișierul este gol.")

    col = "text" if "text" in df.columns else df.columns[0]
    texts = df[col].dropna().astype(str).str.strip()
    texts = [t[:5000] for t in texts if t]

    if not texts:
        raise HTTPException(status_code=400, detail="Nu am găsit texte de analizat în fișier.")
    if len(texts) > MAX_ROWS:
        raise HTTPException(status_code=400,
                            detail=f"Fișierul are {len(texts)} texte. Maximum permis: {MAX_ROWS}.")

    start = time.perf_counter()
    results = classifier(texts, batch_size=32, truncation=True, max_length=256)
    time_ms = int((time.perf_counter() - start) * 1000)

    rows, counts = [], {"pozitiv": 0, "neutru": 0, "negativ": 0}
    for t, r in zip(texts, results):
        best = max(r, key=lambda x: x["score"])
        counts[best["label"]] += 1
        rows.append({"text": t, "label": best["label"], "confidence": round(best["score"], 4)})

    return {"total": len(rows), "column": col, "counts": counts, "time_ms": time_ms, "rows": rows}

# Trebuie sa fie ULTIMA linie, altfel acopera celelalte rute
app.mount("/", StaticFiles(directory=os.path.join(BASE, "frontend"), html=True), name="frontend")