import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -----------------------
# Configurations
# -----------------------

@pytest.fixture
def card2vec_config():
    """Configuration minimale mais valide pour card2vec."""
    return {
        "vector_size": 8,
        "window": 2,
        "min_count": 1,
        "workers": 1,
        "sg": 1,
        "epochs": 5,
    }


# -----------------------
# Données de test
# -----------------------

@pytest.fixture
def small_decks():
    """
    Mini corpus de decks cohérent :
    - pas de mélange de factions
    - cartes répétées
    """
    return [
        ["A", "B", "C"],
        ["A", "B"],
        ["C", "D"],
        ["A", "C"],
    ]


@pytest.fixture
def small_card_embeddings():
    """
    Faux embeddings de cartes (déterministes).
    """
    rng = np.random.default_rng(seed=42)
    return {
        "A": rng.random(8),
        "B": rng.random(8),
        "C": rng.random(8),
        "D": rng.random(8),
    }


@pytest.fixture
def small_deck_embeddings_df(small_decks, small_card_embeddings):
    """
    DataFrame d'embeddings de decks réaliste.
    """
    from src.embeddings.deck_embeddings import compute_all_deck_embeddings

    return compute_all_deck_embeddings(
        decks=small_decks,
        card_embeddings=small_card_embeddings,
    )

@pytest.fixture
def small_card_metadata():
    return pd.DataFrame({
        "reference": ["A", "B", "C", "D"],
        "faction": ["Ordis", "Ordis", "Lyra", "Lyra"],
        "card_type": ["Héros", "Sort", "Héros", "Sort"],
    })

@pytest.fixture
def card_metadata_with_heroes():
    return pd.DataFrame({
        "reference": ["A", "B", "C", "H1", "H2"],
        "card_type": ["Sort", "Personnage", "Permanent", "Héros", "Héros"],
        "faction": ["Ordis", "Ordis", "Lyra", "Ordis", "Lyra"],
    })

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)