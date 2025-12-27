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
    ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Projette des embeddings dans un espace UMAP existant.

    Args:
        umap_model : modèle UMAP déjà entraîné (fit)
        X          : array (n_samples, n_features) à projeter
        ids        : identifiants optionnels (n_samples)

    Returns:
        DataFrame avec colonnes :
            - id (optionnel)
            - x, y (, z)
    """
    # --- Sécurité dimensionnelle (UMAP ne le fait pas)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    expected_dim = umap_model.embedding_.shape[1]
    if X.shape[1] != umap_model._raw_data.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"UMAP was fit on {umap_model._raw_data.shape[1]} dims, "
            f"got {X.shape[1]}"
        )

    # --- Projection
    coords = umap_model.transform(X)

    # --- Colonnes selon dimension
    coord_cols = ["x", "y", "z"][: coords.shape[1]]
    df = pd.DataFrame(coords, columns=coord_cols)

    # --- Ajout des ids si fournis
    if ids is not None:
        if len(ids) != len(df):
            raise ValueError(
                f"ids length ({len(ids)}) does not match X ({len(df)})"
            )
        df.insert(0, "id", list(ids))

    return df


def save_umap_model(umap_model: UMAP, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(umap_model, path)


def load_umap_model(path: Path) -> UMAP:
    return joblib.load(path)