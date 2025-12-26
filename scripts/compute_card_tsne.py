from pathlib import Path
import sys
import yaml

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.deck_embeddings import load_card_embeddings
from src.clustering.tsne_utils import compute_tsne_card_embeddings, save_tsne

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"

with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
INPUT = PROJECT_ROOT / "runs" / model_name / "embeddings" / "cards_vectors.txt"
OUTPUT = PROJECT_ROOT / "runs" / model_name / "projections" / "cards_tsne.csv"

# Chargement des embeddings à transformer
card_embeddings = load_card_embeddings(INPUT)

# Transformation des embeddings en dimension réduite
tsne_df = compute_tsne_card_embeddings(card_embeddings)

# Sauvegarde des coordonnées en dimension réduite
save_tsne(tsne_df, OUTPUT)