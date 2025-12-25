import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _find_best_k(X: np.ndarray, k_min: int = 1, k_max: int = 10) -> int:
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


def cluster_decks(deck_df: pd.DataFrame) -> pd.Series:
    """
    Effectue le clustering des decks **au sein de chaque faction réelle**.

    Prérequis :
        - deck_df contient une colonne 'faction'
        - deck_df contient les colonnes vector_*

    Returns:
        pd.Series des labels de clusters intra-faction, indexée par deck_id
    """
    embedding_cols = [c for c in deck_df.columns if c.startswith("vector_")]
    cluster_labels = pd.Series(index=deck_df.index, dtype=int)

    for faction in deck_df["faction"].unique():
        mask = deck_df["faction"] == faction
        X = deck_df.loc[mask, embedding_cols].to_numpy()

        # Cas très petits groupes
        if X.shape[0] < 3:
            cluster_labels.loc[mask] = 0
            continue

        k_opt = _find_best_k(X)

        if k_opt == 1:
            cluster_labels.loc[mask] = 0
            continue

        kmeans = KMeans(n_clusters=k_opt, random_state=1, n_init="auto")
        labels = kmeans.fit_predict(X)

        cluster_labels.loc[mask] = labels

    return cluster_labels