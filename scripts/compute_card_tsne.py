from pathlib import Path
import sys
import yaml

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.clustering.embedding_utils import embeddings_from_dict
from src.clustering.tsne_utils import compute_tsne, save_tsne
from src.clustering.deck_embeddings import load_card_embeddings

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"

with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
INPUT = PROJECT_ROOT / "runs" / model_name / "embeddings" / "cards_vectors.txt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / model_name / "projections"

# Chargement des embeddings à transformer
card_embeddings = load_card_embeddings(INPUT)
ids, X = embeddings_from_dict(card_embeddings, expected_dim=100)

for c in [2,3] :
    # Transformation des embeddings en dimension réduite
    tsne = compute_tsne(X, ids, n_components=c)
    # Sauvegarde des coordonnées en dimension réduite
    output = OUTPUT_DIR / f"cards_tsne_{c}d.csv"
    save_tsne(tsne, output)