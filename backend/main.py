import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.staticfiles import StaticFiles

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "distilbert_model")

app = FastAPI(title="Sentiment Analysis API")
classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)


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

    res = classifier(text, truncation=True, max_length=256)
    if isinstance(res[0], list):
        res = res[0]
    scores = {r["label"]: round(r["score"], 4) for r in res}
    best = max(scores, key=scores.get)
    return {"label": best, "confidence": scores[best], "scores": scores}

app.mount("/", StaticFiles(directory=os.path.join(BASE, "frontend"), html=True), name="frontend")