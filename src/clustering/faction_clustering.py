from typing import Optional
import pandas as pd
from sklearn.cluster import KMeans

def assign_factions(deck_df: pd.DataFrame, n_clusters: int = 6) -> pd.Series:
    """
    Assigne une faction à chaque deck en fonction de son embedding.

    Args:
        deck_df : DataFrame contenant les embeddings des decks
        n_clusters : nombre de clusters (factions)

    Returns:
        pd.Series de labels, indexée par deck_id
    """

    # Sélectionner uniquement les colonnes vector_*
    embedding_cols = [col for col in deck_df.columns if col.startswith("vector_")]
    X = deck_df[embedding_cols].to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    labels = kmeans.fit_predict(X)

    return pd.Series(labels, index=deck_df.index)