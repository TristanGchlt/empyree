import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _find_best_k(X: np.ndarray, k_min: int = 1, k_max: int = 5) -> int:
    """
    Détermine le meilleur k via le score de silhouette.
    """
    n_samples = X.shape[0]

    # Cas trivial : pas assez de données
    if n_samples <= 1:
        return 1

    k_max = min(k_max, n_samples - 1)
    best_k = 1
    best_score = -1.0

    for k in range(max(k_min, 2), k_max + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=1, n_init="auto").fit_predict(X)
            score = silhouette_score(X, labels)

            if score > best_score:
                best_k = k
                best_score = score
        except Exception:
            continue

    # Garde-fou : si le clustering n'apporte rien
    if best_score < 0.05:
        return 1

    return best_k


def cluster_decks(
    deck_vectors: pd.DataFrame,
    group_column: str = "faction",
) -> pd.Series:
    """
    Effectue un clustering intra-groupe (faction ou héros).

    Returns:
        pd.Series indexée comme deck_vectors, labels numériques
    """
    embedding_cols = [c for c in deck_vectors.columns if c.startswith("vector_")]
    cluster_labels = pd.Series(index=deck_vectors.index, dtype=int)

    for group in deck_vectors[group_column].unique():
        mask = deck_vectors[group_column] == group
        X = deck_vectors.loc[mask, embedding_cols].to_numpy()

        # Groupes trop petits
        if X.shape[0] < 3:
            cluster_labels.loc[mask] = "0"
            continue

        k_opt = _find_best_k(X)

        if k_opt == 1:
            cluster_labels.loc[mask] = "0"
            continue

        kmeans = KMeans(
            n_clusters=k_opt,
            random_state=1,
            n_init="auto"
        )
        labels = kmeans.fit_predict(X)
        cluster_labels.loc[mask] = labels

    return cluster_labels