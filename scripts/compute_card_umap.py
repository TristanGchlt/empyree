from pathlib import Path
import sys
import yaml
import numpy as np

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.deck_embeddings import load_card_embeddings
from src.clustering.umap_utils import fit_umap, transform_umap, save_umap_model

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"
with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
RUN_DIR = PROJECT_ROOT / "runs" / model_name
MODEL_DIR = RUN_DIR / "projection_models"
PROJ_DIR = RUN_DIR / "projections"
EMBEDDINGS_FILE = RUN_DIR / "embeddings" / "cards_vectors.txt"

# Chargement des embeddings
card_embeddings = load_card_embeddings(EMBEDDINGS_FILE)
card_ids = list(card_embeddings.keys())
X = np.vstack([card_embeddings[cid] for cid in card_ids])

for dim in [2, 3]:
    # Réduction de dimension
    umap = fit_umap(X, n_components=dim, n_neighbors=15, min_dist=0.1)
    coords = transform_umap(umap, X, ids=card_ids)
    # Sauvegarde du modèle
    save_umap_model(umap, MODEL_DIR / f"umap_cards_{dim}d.joblib")
    coords.to_csv(PROJ_DIR / f"cards_umap_{dim}d.csv", index=False)