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
CARDS_INFO = PROJECT_ROOT / "data" / "raw" / "cards_info.csv"

def load_cards(dim) -> pd.DataFrame:
    path = PROJECTIONS_DIR / f"cards_tsne_{dim}d.csv"
    tsne = pd.read_csv(path)
    tsne = tsne.rename(columns={tsne.columns[0]: "card_id"})
    meta_df = pd.read_csv(CARDS_INFO)
    df = tsne.merge(meta_df, left_on="card_id", right_on="reference", how="left")
    df = df.rename(columns={"name": "nom", "faction": "faction", "card_type": "type"})
    return df