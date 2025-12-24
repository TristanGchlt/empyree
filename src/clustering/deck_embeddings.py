import numpy as np
from typing import List, Dict


def compute_deck_embedding(deck: List[str], card_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calcule l'embedding d'un deck.

    Args:
        - Liste des identifiants de cartes du deck
        - Embeddings des cartes

    Returns:
        Vecteur représentant le deck
    """
    vectors = [
        card_embeddings[card_id]
        for card_id in deck
        if card_id in card_embeddings
    ]

    if not vectors:
        raise ValueError("Aucune carte du deck n'a d'embedding connu.")

    return np.mean(vectors, axis=0)


def compute_all_deck_embeddings(decks: List[List[str]], card_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calcule les embeddings pour tous les decks.

    Args:
        - Liste de decks (chaque deck est une liste de cartes)
        - Embeddings des cartes

    Returns:
        Table des decks embeddings
    """
    return np.vstack(
        [
            compute_deck_embedding(deck, card_embeddings)
            for deck in decks
        ]
    )