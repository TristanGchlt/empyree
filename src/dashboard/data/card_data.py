import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CARDS_TSNE_FILE = PROJECT_ROOT / "data" / "final" / "cards_tsne.csv"
CARD_METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "cards_info.csv"

def load_cards() -> pd.DataFrame:
    tsne_df = pd.read_csv(CARDS_TSNE_FILE, index_col=0)
    meta_df = pd.read_csv(CARD_METADATA_FILE)
    df = tsne_df.merge(meta_df, left_on="card_id", right_on="reference", how="left")
    df = df.rename(columns={"name": "nom", "faction": "faction", "card_type": "type"})
    return df