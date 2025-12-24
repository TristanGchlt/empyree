import pandas as pd
from sklearn.cluster import KMeans
from typing import Union


def cluster_decks(deck_df: pd.DataFrame, faction_labels: pd.Series, n_clusters: int = 3) -> pd.Series:
    """
    Effectue le clustering des decks **au sein de chaque faction**.

    Args:
        deck_df : DataFrame contenant les embeddings (vector_*)
        faction_labels : pd.Series des labels de faction, indexé par deck_id
        n_clusters : nombre de clusters par faction

    Returns:
        pd.Series des labels de sous-clusters, indexée par deck_id
    """
    embedding_cols = [col for col in deck_df.columns if col.startswith("vector_")]
    subcluster_labels = pd.Series(index=deck_df.index, dtype=int)

    # Pour chaque faction, on clusterise ses decks
    for faction in faction_labels.unique():
        mask = faction_labels == faction
        X = deck_df.loc[mask, embedding_cols].to_numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=1)
        labels = kmeans.fit_predict(X)

        # On stocke les labels dans la série finale
        subcluster_labels.loc[mask] = labels

    return subcluster_labels