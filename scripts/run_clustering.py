from pathlib import Path
import sys
import yaml
import pandas as pd
import ast

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.deck_embeddings import compute_all_deck_embeddings, load_card_embeddings
from src.clustering.deck_clustering import cluster_decks

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"
with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
INPUT_INFO = PROJECT_ROOT / "data" / "raw" / "cards_info.csv"
INPUT_CORPUS = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"
INPUT_EMBEDDINGS = PROJECT_ROOT / "runs"/ model_name / "embeddings" / "decks_vectors.csv"
OUTPUT_CLUSTER = PROJECT_ROOT / "runs" / model_name / "clustering" / "decks_clusters.csv"

# Chargement des embeddings
deck_embeddings = pd.read_csv(INPUT_EMBEDDINGS)

# Récupération des factions
metadata = pd.read_csv(INPUT_INFO)
card_to_faction = metadata.set_index("reference")["faction"].to_dict()
deck_embeddings["faction"] = deck_embeddings["cards"].apply(
    lambda cards: card_to_faction.get(ast.literal_eval(cards)[0], "Unknown")
)

# Clustering intra faction
deck_embeddings["cluster"] = cluster_decks(deck_embeddings)

# Sauvegarde des clusters
deck_embeddings[["deck_id", "faction", "cluster"]].to_csv(OUTPUT_CLUSTER, index=False)