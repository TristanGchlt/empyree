import pandas as pd

from src.clustering.deck_clustering_pipeline import run_deck_clustering


def test_deck_embeddings_to_clusters(small_deck_embeddings_df):
    small_deck_embeddings_df = small_deck_embeddings_df.copy()
    small_deck_embeddings_df["cards"] = small_deck_embeddings_df["cards"].apply(str)

    card_metadata = pd.DataFrame({
        "reference": ["A", "B", "C", "D"],
        "faction": ["Ordis", "Ordis", "Lyra", "Lyra"],
        "card_type": ["Héros", "Sort", "Héros", "Sort"],
    })

    result = run_deck_clustering(
        deck_embeddings=small_deck_embeddings_df,
        card_metadata=card_metadata,
    )

    assert set(result.columns) == {"deck_id", "faction", "hero", "cluster"}
    assert result["cluster"].notna().all()
    assert result["faction"].isin(["Ordis", "Lyra"]).all()