import numpy as np

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

def test_deck_embedding_inside_card_bounds(rng):
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

def test_deck_embedding_ignores_unknown_cards():
    cards = {
        "A": np.array([1.0, 1.0]),
        "B": np.array([3.0, 3.0]),
    }

    deck = ["A", "B", "UNKNOWN"]
    emb = compute_deck_embedding(deck, cards)

    assert np.allclose(emb, np.array([2.0, 2.0]))

def test_all_deck_embeddings_shape():
    card_embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
    }

    decks = [["A", "B"], ["A"]]
    df = compute_all_deck_embeddings(decks, card_embeddings)

    assert df.shape[0] == 2
    assert {"vector_0", "vector_1"}.issubset(df.columns)
    assert df[["vector_0", "vector_1"]].notna().all().all()