from typing import Optional, Dict
import pandas as pd
from sklearn.cluster import KMeans


def cluster_decks(
    deck_embeddings: pd.DataFrame,
    factions: pd.Series,
    n_clusters_per_faction: Optional[Dict[int, int]] = None,
    default_n_clusters: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """
    Effectue un clustering des decks **au sein de chaque faction**.

    Parameters
    ----------
    deck_embeddings : pd.DataFrame
        DataFrame indexé par deck_id, contenant les embeddings
    factions : pd.Series
        Labels de factions pour chaque deck
    n_clusters_per_faction : dict, optional
        Dictionnaire {faction_id: n_clusters} pour chaque faction
    default_n_clusters : int
        Nombre de clusters par défaut si non spécifié
    random_state : int
        Graine aléatoire pour reproductibilité

    Returns
    -------
    pd.Series
        Série indexée par deck_id avec le label de cluster intra-faction
        Format: "faction_cluster" (ex: "0_2")
    """

    cluster_labels = pd.Series(index=deck_embeddings.index, dtype=str)

    # Assurez-vous qu'on a bien un dict vide si None
    if n_clusters_per_faction is None:
        n_clusters_per_faction = {}

    for faction_id in factions.unique():
        mask = factions == faction_id
        X = deck_embeddings.loc[mask].values

        n_clusters = n_clusters_per_faction.get(faction_id, default_n_clusters)

        if len(X) < n_clusters:
            # On ne peut pas créer plus de clusters que de decks
            n_clusters = max(1, len(X))

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        # Création de labels combinés "faction_cluster"
        combined_labels = [f"{faction_id}_{l}" for l in labels]

        cluster_labels.loc[mask] = combined_labels

    return cluster_labels