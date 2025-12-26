import sys
from pathlib import Path
import pandas as pd

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.preprocessing.deck_preprocessing import preprocess_deck

# Path des input et output
INPUT = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"

def main():
    df = pd.read_csv(INPUT)

    decks = (
        df["cards"]
        .dropna()
        .apply(preprocess_deck)
    )

    decks = decks[decks.apply(len) > 0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for deck in decks:
            f.write(" ".join(deck) + "\n")


if __name__ == "__main__":
    main()