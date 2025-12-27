from pathlib import Path
import sys
import yaml
import pandas as pd

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.embedding_utils import embeddings_from_dataframe
from src.clustering.umap_utils import load_umap_model, transform_umap

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"
with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
RUN_DIR = PROJECT_ROOT / "runs" / model_name
DECK_EMBEDDINGS = RUN_DIR / "embeddings" / "decks_vectors.csv"
MODEL_DIR = RUN_DIR / "projection_models"
PROJ_DIR = RUN_DIR / "projections"

# Chargement des embeddings de deck
deck_df = pd.read_csv(DECK_EMBEDDINGS)
deck_ids, X = embeddings_from_dataframe(deck_df, id_column="deck_id")

for dim in [2, 3]:
    # Récupération du modèle de réduction de dimension
    umap = load_umap_model(MODEL_DIR / f"umap_cards_{dim}d.joblib")
    # Réduction de dimension
    coords = transform_umap(umap, X, ids=deck_ids)
    # Sauvegarde des coordonnées
    coords.to_csv(PROJ_DIR / f"decks_umap_{dim}d.csv", index=False)