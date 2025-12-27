from pathlib import Path
import sys
import yaml
import pandas as pd

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.embedding_utils import embeddings_from_dataframe
from src.clustering.umap_pipeline import run_projection_umap

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

deck_df = pd.read_csv(DECK_EMBEDDINGS)
deck_ids, X = embeddings_from_dataframe(deck_df, id_column="deck_id")

run_projection_umap(
    X=X,
    ids=deck_ids,
    model_dir=MODEL_DIR,
    output_proj_dir=PROJ_DIR,
    prefix="decks_umap",
)