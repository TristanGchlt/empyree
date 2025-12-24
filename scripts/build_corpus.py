import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.deck_preprocessing import preprocess_deck

INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
OUTPUT_TXT = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"


def main():
    df = pd.read_csv(INPUT_CSV)

    decks = (
        df["cards"]
        .dropna()
        .apply(preprocess_deck)
    )

    decks = decks[decks.apply(len) > 0]

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for deck in decks:
            f.write(" ".join(deck) + "\n")


if __name__ == "__main__":
    main()