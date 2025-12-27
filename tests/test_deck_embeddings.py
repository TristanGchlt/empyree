import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.embeddings.deck_embeddings import compute_deck_embedding,  compute_all_deck_embeddings

def test_deck_embedding_is_mean():
    card_embeddings = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([2.0, 2.0]),
        "C": np.array([4.0, 4.0]),
    }

    deck = ["A", "B", "C"]
    emb = compute_deck_embedding(deck, card_embeddings)

    assert np.allclose(emb, np.array([2.0, 2.0]))

def test_deck_embedding_inside_card_bounds():
    rng = np.random.default_rng(42)

    cards = {
        f"C{i}": rng.uniform(-10, 10, size=5)
        for i in range(10)
    }

    deck = list(cards.keys())
    emb = compute_deck_embedding(deck, cards)

    all_vectors = np.vstack(list(cards.values()))

    assert emb.min() >= all_vectors.min()
    assert emb.max() <= all_vectors.max()

def test_no_nan_in_deck_embeddings():
    cards = {
        "A": np.ones(3),
        "B": np.zeros(3),
    }

    deck = ["A", "B"]
    emb = compute_deck_embedding(deck, cards)

    assert np.isfinite(emb).all()

def test_all_deck_embeddings_shape():
    card_embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
    }

    decks = [["A", "B"], ["A"]]
    df = compute_all_deck_embeddings(decks, card_embeddings)

    assert df.shape[0] == 2
    assert "vector_0" in df.columns
    assert "vector_1" in df.columns