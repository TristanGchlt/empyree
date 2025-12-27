from pathlib import Path
import sys
import yaml
import pandas as pd

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.deck_clustering_pipeline import run_deck_clustering

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
card_metadata = pd.read_csv(INPUT_INFO)

clusters = run_deck_clustering(
    deck_embeddings=deck_embeddings,
    card_metadata=card_metadata,
)

OUTPUT_CLUSTER.parent.mkdir(parents=True, exist_ok=True)
clusters.to_csv(OUTPUT_CLUSTER, index=False)