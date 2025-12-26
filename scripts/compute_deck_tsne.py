from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering.tsne_utils import compute_tsne, save_tsne

# --- Chemins ---
DECK_DATA_FILE = PROJECT_ROOT / "data" / "final" / "deck_dataset.csv"
TSNE_OUTPUT_FILE = PROJECT_ROOT / "data" / "final" / "deck_dataset_tsne.csv"

# --- Charger les embeddings ---
deck_df = pd.read_csv(DECK_DATA_FILE, index_col=0)

# --- Calcul du t-SNE ---
tsne_df = compute_tsne(deck_df)

# --- Sauvegarde ---
save_tsne(tsne_df, TSNE_OUTPUT_FILE)
print(f"t-SNE saved to {TSNE_OUTPUT_FILE}")