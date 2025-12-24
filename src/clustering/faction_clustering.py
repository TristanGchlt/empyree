from typing import Optional
import pandas as pd
from sklearn.cluster import KMeans

def assign_factions(
    deck_embeddings: pd.DataFrame,
    n_factions: Optional[int] = 6,
    external_labels: Optional[pd.Series] = None,
    random_state: int = 42,
) -> pd.Series:
    """
    Assigne une faction à chaque deck.

    Deux modes possibles :
    - si la faction (external_labels) est fournie, on l'utilise directement
    - sinon, on applique un clustering (KMeans) sur les embeddings

    Args:
        - DataFrame indexé par deck_id, contenant les embeddings des decks
        - Nombre de factions à créer (obligatoire si external_labels est None)
        - Labels de factions fournis par une source externe
        - Graine aléatoire pour la reproductibilité

    Returns:
        Série indexée comme deck_embeddings, contenant le label de faction
    """

    # --- Cas 1 : labels externes fournis ---
    if external_labels is not None:
        if not deck_embeddings.index.equals(external_labels.index):
            raise ValueError("Les index de deck_embeddings et external_labels ne correspondent pas")

        return external_labels.rename("faction")

    # --- Cas 2 : clustering automatique ---
    X = deck_embeddings.values

    kmeans = KMeans(
        n_clusters=n_factions,
        random_state=random_state,
        n_init=10,
    )

    labels = kmeans.fit_predict(X)

    return pd.Series(labels, index=deck_embeddings.index, name="faction")