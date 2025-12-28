import numpy as np

from src.projection.umap_utils import fit_umap, transform_umap


def test_card_embeddings_to_umap(small_card_embeddings):
    # Arrange
    card_ids = list(small_card_embeddings.keys())
    X = np.vstack([small_card_embeddings[c] for c in card_ids])

    # Act
    umap = fit_umap(X, n_components=2, random_state=42)
    coords = transform_umap(umap, X, ids=card_ids)

    # Assert
    assert coords.shape == (len(card_ids), 3)  # id + x + y
    assert list(coords.columns) == ["id", "x", "y"]
    assert coords["id"].tolist() == card_ids