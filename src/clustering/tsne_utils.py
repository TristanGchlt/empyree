import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
import warnings

def compute_tsne(
    X: np.ndarray,
    ids: list[str],
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 1
) -> pd.DataFrame:
    """
    Calcule un TSNE à partir d'une matrice X.

    Args:
        X : np.ndarray (n_samples, n_features)
        ids : identifiants associés aux lignes de X
    """
    if n_components not in (2, 3):
        raise ValueError("n_components doit être 2 ou 3")

    if len(X) != len(ids):
        raise ValueError("X et ids doivent avoir la même longueur")

    max_perplexity = max(1, (len(X) - 1) // 3)
    if perplexity > max_perplexity:
        warnings.warn(
            f"Perplexity {perplexity} trop grande, ajustée à {max_perplexity}"
        )
        perplexity = max_perplexity

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    coords = tsne.fit_transform(X)

    columns = ["x", "y"] if n_components == 2 else ["x", "y", "z"]

    df = pd.DataFrame(coords, columns=columns)
    df.insert(0, "id", ids)

    return df


def save_tsne(df: pd.DataFrame, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)