import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("models", exist_ok=True)

# Incarcare date
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
print("Train:", len(train), "| Validation:", len(val))

# Model: TF-IDF + Regresie Logistica
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),      # cuvinte singure si perechi de cuvinte
        max_features=200000,
        min_df=3,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # compenseaza clasa "neutru", mai mica
    )),
])

print("\nAntrenare in curs...")
start = time.time()
model.fit(train["text"], train["label"])
print(f"Antrenare terminata in {time.time() - start:.1f} secunde")

# Evaluare pe setul de validare
pred = model.predict(val["text"])
print("\nAcuratete:", round(accuracy_score(val["label"], pred), 4))
print("F1 macro:", round(f1_score(val["label"], pred, average="macro"), 4))

print("\nRaport pe clase:")
print(classification_report(val["label"], pred, digits=3))

labels = ["negativ", "neutru", "pozitiv"]
print("Matricea de confuzie (randuri = real, coloane = prezis):")
print(pd.DataFrame(confusion_matrix(val["label"], pred, labels=labels),
                   index=labels, columns=labels))

# Acuratete separat pe fiecare sursa
print("\nAcuratete pe sursa:")
for src in val["source"].unique():
    mask = val["source"] == src
    print(f"  {src}: {accuracy_score(val.loc[mask, 'label'], pred[mask]):.4f}")

# Salvare model
joblib.dump(model, "models/baseline_tfidf_lr.joblib")
print("\nModel salvat in models/baseline_tfidf_lr.joblib")