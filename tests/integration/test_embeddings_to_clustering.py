import pandas as pd

from src.clustering.deck_clustering_pipeline import run_deck_clustering


def test_deck_embeddings_to_clusters(small_deck_embeddings_df):
    # Fake metadata cohérente
    card_metadata = pd.DataFrame({
        "reference": ["A", "B", "C", "D"],
        "faction": ["Ordis", "Ordis", "Lyra", "Lyra"],
    })

    # Act
    result = run_deck_clustering(
        deck_embeddings=small_deck_embeddings_df,
        card_metadata=card_metadata,
    )

    # Assert
    assert set(result.columns) == {"deck_id", "faction", "cluster"}
    assert result["cluster"].notna().all()
    assert result["faction"].isin(["Ordis", "Lyra"]).all()