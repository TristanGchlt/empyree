import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _find_best_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
    """
    Détermine le meilleur k via le score de silhouette.

    Args:
        X : array (n_samples, n_features)
        k_min : nombre minimum de clusters testés
        k_max : nombre maximum de clusters testés

    Returns:
        k optimal
    """
    n_samples = X.shape[0]

    # Pas assez de données pour clusteriser
    if n_samples < k_min + 1:
        return 1

    k_max = min(k_max, n_samples - 1)

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=1).fit_predict(X)
            score = silhouette_score(X, labels)

            if score > best_score:
                best_k = k
                best_score = score
        except Exception:
            continue

    # Garde-fou : si le clustering est très mauvais, on reste simple
    if best_score < 0.05:
        return 1

    return best_k


def cluster_decks(deck_df: pd.DataFrame, faction_labels: pd.Series) -> pd.Series:
    """
    Effectue le clustering des decks **au sein de chaque faction**,
    avec détermination automatique du nombre de clusters par faction.

    Args:
        deck_df : DataFrame contenant les embeddings (colonnes vector_*)
        faction_labels : pd.Series des labels de faction, indexée par deck_id

    Returns:
        pd.Series des labels de clusters intra-faction, indexée par deck_id
    """
    embedding_cols = [c for c in deck_df.columns if c.startswith("vector_")]
    cluster_labels = pd.Series(index=deck_df.index, dtype=int)

    for faction in faction_labels.unique():
        mask = faction_labels == faction
        X = deck_df.loc[mask, embedding_cols].to_numpy()

        # Cas très petits groupes
        if X.shape[0] < 3:
            cluster_labels.loc[mask] = 0
            continue

        k_opt = _find_best_k(X)

        kmeans = KMeans(n_clusters=k_opt, random_state=1)
        labels = kmeans.fit_predict(X)

        cluster_labels.loc[mask] = labels

    return cluster_labels