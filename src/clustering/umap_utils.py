import numpy as np
import pandas as pd
from umap import UMAP
from pathlib import Path
import joblib


def fit_umap(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> UMAP:
    """
    Entraîne un modèle UMAP sur un jeu de données de référence.
    """
    umap = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_jobs=1,
        **kwargs,
    )
    umap.fit(X)
    return umap


def transform_umap(
    umap_model: UMAP,
    X: np.ndarray,
    ids=None,
) -> pd.DataFrame:
    """
    Projette des embeddings dans un espace UMAP existant.
    """
    coords = umap_model.transform(X)

    df = pd.DataFrame(coords, columns=[c for c in ["x", "y", "z"][: coords.shape[1]]])

    if ids is not None:
        df.insert(0, "id", ids)

    return df


def save_umap_model(umap_model: UMAP, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(umap_model, path)


def load_umap_model(path: Path) -> UMAP:
    return joblib.load(path)