from pathlib import Path
import sys
import yaml
import numpy as np

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.embeddings.deck_embeddings import load_card_embeddings
from src.projection.umap_pipeline import run_reference_umap

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

run_reference_umap(
    X=X,
    ids=card_ids,
    output_model_dir=MODEL_DIR,
    output_proj_dir=PROJ_DIR,
    prefix="cards_umap",
    n_neighbors=15,
    min_dist=0.1,
)