from pathlib import Path
import sys
import yaml
import pandas as pd

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.tsne_utils import compute_tsne, save_tsne

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"

with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
INPUT = PROJECT_ROOT / "runs" / model_name / "embeddings" / "decks_vectors.csv"
OUTPUT_DIR = PROJECT_ROOT / "runs" / model_name / "projections"

# Chargement des embeddings à transformer
deck_embeddings = pd.read_csv(INPUT, index_col=0)

for c in [2,3] :
    # Transormation des embeddings en dimension réduite
    tsne = compute_tsne(deck_embeddings, n_components=c)
    # Sauvegarde des coordonnées en dimension réduite
    output = OUTPUT_DIR / f"decks_tsne_{c}d.csv"
    save_tsne(tsne, output)