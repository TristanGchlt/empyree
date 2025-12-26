import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict

def compute_tsne(embeddings: pd.DataFrame, vector_prefix: str = "vector_", n_components: int = 2, random_state: int = 1) -> pd.DataFrame:
    """
    Calcule un t-SNE à partir d'un DataFrame contenant des embeddings.

    Args:
        embeddings : DataFrame avec colonnes vector_0, vector_1, ...
        vector_prefix : préfixe des colonnes contenant les vecteurs
        n_components : dimension du t-SNE (2 ou 3)
        random_state : pour reproductibilité

    Returns:
        DataFrame avec colonnes 'x', 'y' (et 'z' si n_components=3), indexées comme embeddings
    """
    vector_cols = [c for c in embeddings.columns if c.startswith(vector_prefix)]
    X = embeddings[vector_cols].to_numpy()

    tsne = TSNE(n_components=n_components, random_state=random_state)
    coords = tsne.fit_transform(X)

    df = pd.DataFrame({
        "deck_id": embeddings.index,
        "x": coords[:, 0],
        "y": coords[:, 1]
    })
    if n_components==3:
        df["z"] = coords[:, 2]

    return df

def compute_tsne_card_embeddings(card_embeddings: Dict[str, np.ndarray],
                                 n_components: int = 2,
                                 perplexity: int = 30,
                                 random_state: int = 1) -> pd.DataFrame:
    """
    Calcule le TSNE des embeddings de cartes.
    Filtre automatiquement les cartes dont l'embedding n'a pas la bonne dimension.
    """
    card_ids = []
    vectors = []
    for cid, vec in card_embeddings.items():
        if vec.ndim == 1 and vec.shape[0] == 100:
            card_ids.append(cid)
            vectors.append(vec)
    X = np.vstack(vectors)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    coords = tsne.fit_transform(X)

    df = pd.DataFrame({
        "card_id": card_ids,
        "x": coords[:, 0],
        "y": coords[:, 1]
    })
    if n_components==3:
        df["z"] = coords[:, 2]
    return df



def save_tsne(df_tsne: pd.DataFrame, output_file: Path):
    """
    Sauvegarde le DataFrame t-SNE en CSV.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_tsne.to_csv(output_file, index=False)