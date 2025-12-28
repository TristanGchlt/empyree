import numpy as np

from src.embeddings.card2vec import train_card2vec, extract_card_embeddings
from src.embeddings.deck_embeddings import compute_all_deck_embeddings
from src.projection.umap_utils import fit_umap, transform_umap


def test_mini_pipeline_end_to_end(card2vec_config, small_decks):
    # Entrainement
    model = train_card2vec(small_decks, card2vec_config)
    card_embeddings = extract_card_embeddings(model)

    # Embeddings
    deck_df = compute_all_deck_embeddings(
        decks=small_decks,
        card_embeddings=card_embeddings,
    )

    vector_cols = [c for c in deck_df.columns if c.startswith("vector_")]
    X = deck_df[vector_cols].to_numpy()

    # Projection
    umap = fit_umap(X, n_components=2, random_state=42)
    coords = transform_umap(umap, X)

    # Assertions finales
    assert coords.shape[0] == len(deck_df)
    assert coords.shape[1] == 2
    assert np.isfinite(coords.to_numpy()).all()