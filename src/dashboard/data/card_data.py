import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CARDS_FILE = PROJECT_ROOT / "data" / "raw" / "cards_info.csv"
VECTORS_FILE = PROJECT_ROOT / "models" / "card2vec_vectors.txt"


def load_cards() -> pd.DataFrame:
    """
    Charge les métadonnées cartes + embeddings card2vec.
    """
    meta = pd.read_csv(CARDS_FILE)

    vectors = []
    with open(VECTORS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            vectors.append([parts[0]] + list(map(float, parts[1:])))

    vec_df = pd.DataFrame(vectors)
    vec_df.rename(columns={0: "reference"}, inplace=True)

    df = meta.merge(vec_df, on="reference", how="inner")

    # projection simple pour le proto
    df["x"] = df[1]
    df["y"] = df[2]

    return df