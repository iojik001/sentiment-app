import os
import re
import html
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)


def clean_text(text: str) -> str:
    """Curata un text: HTML, linkuri, spatii in exces."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)           # taguri HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # linkuri
    text = re.sub(r"\s+", " ", text)               # spatii multiple
    return text.strip()


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/combined/reviews_combined.csv")
    print("Recenzii incarcate:", len(df))
    print("Recenzii cu HTML inainte de curatare:",
          df["text"].str.contains("<br", regex=False).sum())

    # Curatare
    df["text"] = df["text"].apply(clean_text)
    print("Recenzii cu HTML dupa curatare:",
          df["text"].str.contains("<br", regex=False).sum())

    # Eliminam textele prea scurte (sub 3 cuvinte) si duplicatele aparute dupa curatare
    df["words"] = df["text"].str.split().str.len()
    scurte = (df["words"] < 3).sum()
    df = df[df["words"] >= 3]
    dup = df.duplicated(subset=["text"]).sum()
    df = df.drop_duplicates(subset=["text"])
    print("Texte prea scurte eliminate:", scurte)
    print("Duplicate noi eliminate:", dup)
    print("Total dupa curatare:", len(df))

    # Statistici despre lungimea textelor
    print("\nLungimea textelor (nr. cuvinte):")
    print(df["words"].describe().round(1))

    # Impartire 80 / 10 / 10, pastrand proportia claselor si a surselor
    strat = df["source"] + "_" + df["label"]
    train, rest = train_test_split(df, test_size=0.2, stratify=strat, random_state=42)
    strat_rest = rest["source"] + "_" + rest["label"]
    val, test = train_test_split(rest, test_size=0.5, stratify=strat_rest, random_state=42)

    cols = ["text", "source", "label"]
    train[cols].to_csv("data/processed/train.csv", index=False)
    val[cols].to_csv("data/processed/val.csv", index=False)
    test[cols].to_csv("data/processed/test.csv", index=False)

    print("\nImpartire:")
    print("train:", len(train), "| val:", len(val), "| test:", len(test))
    print("\nDistributia claselor in train:")
    print(train["label"].value_counts())