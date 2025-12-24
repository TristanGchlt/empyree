import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union


def compute_deck_embedding(deck: List[str], card_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calcule l'embedding d'un deck en prenant la moyenne des embeddings de ses cartes.
    """
    vectors = [card_embeddings[card_id] for card_id in deck if card_id in card_embeddings]

    if not vectors:
        raise ValueError("Aucune carte du deck n'a d'embedding connu.")

    return np.mean(vectors, axis=0)


def load_card_embeddings(embeddings_file: Path) -> Dict[str, np.ndarray]:
    """
    Lit le fichier de vecteurs et retourne un dictionnaire {card_id: np.array}.
    Chaque ligne du fichier doit être : card_id val1 val2 ... valN
    """
    embeddings = {}
    with open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            card_id, vector = parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[card_id] = vector
    return embeddings


def compute_all_deck_embeddings(
    decks: Union[List[List[str]], Path],
    card_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Calcule les embeddings pour tous les decks.

    Args:
        decks : soit un Path vers un fichier txt (deck par ligne, cartes séparées par espace),
                soit une liste de decks (chaque deck est une liste de cartes)
        card_embeddings : dictionnaire {card_id: np.array}

    Returns:
        DataFrame avec colonnes :
            - 'deck_id' : index du deck
            - 'cards'   : liste de cartes (string ou list)
            - embeddings : colonnes vector_0, vector_1, ..., vector_{dim-1}
    """
    # Lire le fichier si nécessaire
    if isinstance(decks, (str, Path)):
        path = Path(decks)
        with path.open("r", encoding="utf-8") as f:
            decks = [line.strip().split() for line in f if line.strip()]

    data = []
    for idx, deck in enumerate(decks):
        embedding = compute_deck_embedding(deck, card_embeddings)
        data.append({
            "deck_id": idx,
            "cards": deck,
            **{f"vector_{i}": val for i, val in enumerate(embedding)}
        })

    return pd.DataFrame(data)