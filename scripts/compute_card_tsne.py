from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering.deck_embeddings import load_card_embeddings
from src.clustering.tsne_utils import compute_tsne_card_embeddings, save_tsne

EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "card2vec_vectors.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "final" / "cards_tsne.csv"

# --- 1️⃣ Charger les embeddings ---
card_embeddings = load_card_embeddings(EMBEDDINGS_FILE)

# --- 2️⃣ Calculer TSNE ---
tsne_df = compute_tsne_card_embeddings(card_embeddings)

# --- 3️⃣ Sauvegarder ---
save_tsne(tsne_df, OUTPUT_FILE)
print(f"TSNE cartes sauvegardé dans {OUTPUT_FILE}")