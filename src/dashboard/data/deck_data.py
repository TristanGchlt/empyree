import pandas as pd
from pathlib import Path
import yaml

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"
with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

PROJECTIONS_DIR = PROJECT_ROOT / "runs" / model_name / "projections"
DECKS_CLUSTER_FILE = PROJECT_ROOT / "runs" / model_name / "clustering" / "decks_clusters.csv"
PROJECTION_TYPE = "umap"

def load_decks(dim: int) -> pd.DataFrame:
    """
    Charge les projections t-SNE et merge avec les informations de faction et cluster des decks.
    """
    tsne_path = PROJECTIONS_DIR / f"decks_{PROJECTION_TYPE}_{dim}d.csv"
    tsne_df = pd.read_csv(tsne_path)
    tsne_df = tsne_df.rename(columns={tsne_df.columns[0]: "deck_id"})

    clusters_df = pd.read_csv(DECKS_CLUSTER_FILE)

    df = tsne_df.merge(clusters_df, on="deck_id", how="left")

    return df