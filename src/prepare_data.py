import os
import pandas as pd
from datasets import load_dataset
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)


os.makedirs("data/raw/imdb", exist_ok=True)
os.makedirs("data/raw/amazon", exist_ok=True)
os.makedirs("data/combined", exist_ok=True)

# IMDB (0 = negativ, 1 = pozitiv)
imdb = load_dataset("stanfordnlp/imdb")
imdb_df = pd.concat([imdb["train"].to_pandas(), imdb["test"].to_pandas()])
imdb_df.to_csv("data/raw/imdb/imdb_raw.csv", index=False)
imdb_df = pd.DataFrame({
    "text": imdb_df["text"],
    "source": "imdb",
    "label": imdb_df["label"].map({0: "negativ", 1: "pozitiv"}),
})

# Amazon (1-5 stele)
amz = load_dataset("SetFit/amazon_reviews_multi_en")
amz_df = pd.concat([amz["train"].to_pandas(), amz["test"].to_pandas()])
amz_df.to_csv("data/raw/amazon/amazon_raw.csv", index=False)
stars = amz_df["label"] + 1
amz_df = pd.DataFrame({
    "text": amz_df["text"],
    "source": "amazon",
    "label": stars.map(lambda s: "negativ" if s <= 2 else "neutru" if s == 3 else "pozitiv"),
})
amz_df = amz_df.sample(n=50000, random_state=42)

# Unificare si verificari
df = pd.concat([imdb_df, amz_df], ignore_index=True)
print("Inainte de curatare:", len(df))
print("Valori lipsa:\n", df.isna().sum())
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]
dup = df.duplicated(subset=["text"]).sum()
df = df.drop_duplicates(subset=["text"])
print("Duplicate eliminate:", dup)
print("Dupa curatare:", len(df))
print(df.groupby(["source", "label"]).size())

df.to_csv("data/combined/reviews_combined.csv", index=False)