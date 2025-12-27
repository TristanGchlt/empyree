import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Union
import warnings


def embeddings_from_dataframe(
    df: pd.DataFrame,
    id_column: str,
    vector_prefix: str = "vector_"
) -> Tuple[List[str], np.ndarray]:
    """
    Extrait une matrice de vecteurs depuis un DataFrame.

    Returns:
        ids : liste des identifiants
        X   : np.ndarray (n_samples, n_features)
    """
    vector_cols = [c for c in df.columns if c.startswith(vector_prefix)]
    if not vector_cols:
        raise ValueError(f"Aucune colonne commençant par '{vector_prefix}'")

    if id_column not in df.columns:
        raise ValueError(f"Colonne '{id_column}' absente du DataFrame")

    ids = df[id_column].astype(str).tolist()
    X = df[vector_cols].to_numpy()

    return ids, X


def embeddings_from_dict(
    embeddings: Dict[str, np.ndarray],
    expected_dim: int | None = None
) -> Tuple[List[str], np.ndarray]:
    """
    Extrait une matrice de vecteurs depuis un dict {id: vector}.
    """
    ids, vectors = [], []

    for key, vec in embeddings.items():
        if not isinstance(vec, np.ndarray) or vec.ndim != 1:
            warnings.warn(f"Embedding invalide pour {key}, ignoré")
            continue

        if expected_dim and vec.shape[0] != expected_dim:
            warnings.warn(
                f"Embedding {key} dimension {vec.shape[0]} ≠ {expected_dim}, ignoré"
            )
            continue

        ids.append(str(key))
        vectors.append(vec)

    if not vectors:
        raise ValueError("Aucun embedding valide trouvé")

    return ids, np.vstack(vectors)