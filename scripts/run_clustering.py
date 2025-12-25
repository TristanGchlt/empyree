from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering.deck_embeddings import compute_all_deck_embeddings, load_card_embeddings
# from src.clustering.faction_clustering import assign_factions
from src.clustering.deck_clustering import cluster_decks

# --- Chemins ---
CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"
EMBEDDINGS_FILE = PROJECT_ROOT / "models"/ "card2vec_vectors.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "final" / "deck_dataset.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "cards_info.csv"

# --- 1️⃣ Charger les decklists et embeddings ---
card_embeddings = load_card_embeddings(EMBEDDINGS_FILE)
deck_lists = compute_all_deck_embeddings(CORPUS_FILE, card_embeddings)  
# deck_lists : DataFrame avec colonnes 'deck_id', 'cards', et embedding vectorisé

# --- 2️⃣ Attribution des factions (6 zones) ---
# factions = assign_factions(deck_lists)  # retourne pd.Series indexé par deck_id
# deck_lists["faction"] = factions
metadata = pd.read_csv(METADATA_FILE)
card_to_faction = metadata.set_index("reference")["faction"].to_dict()
deck_lists["faction"] = deck_lists["cards"].apply(
    lambda cards: card_to_faction.get(cards[0], "Unknown")
)

# --- 3️⃣ Clustering intra-faction ---
deck_lists["cluster"] = cluster_decks(deck_lists,)  # retourne pd.Series indexé par deck_id

# --- 4️⃣ Sauvegarde du dataset final ---
deck_lists.to_csv(OUTPUT_FILE, index=True)

print(f"Final deck dataset saved to {OUTPUT_FILE}")