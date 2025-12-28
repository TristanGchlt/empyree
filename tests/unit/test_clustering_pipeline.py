import pandas as pd

from src.clustering.deck_clustering_pipeline import run_deck_clustering

def test_run_deck_clustering_basic():
    deck_embeddings = pd.DataFrame({
        "deck_id": [0, 1, 2],
        "cards": [
            "['CARD_A']",
            "['CARD_A']",
            "['CARD_B']"
        ],
        "vector_0": [0.0, 0.1, 10.0],
        "vector_1": [0.0, 0.1, 10.0],
    })

    card_metadata = pd.DataFrame({
        "reference": ["CARD_A", "CARD_B"],
        "faction": ["Ordis", "Lyra"]
    })

    result = run_deck_clustering(deck_embeddings, card_metadata)

    # Colonnes attendues
    assert set(result.columns) == {"deck_id", "faction", "cluster"}

    # Même faction pour les deux premiers decks
    assert result.loc[result["deck_id"] == 0, "faction"].iloc[0] == "Ordis"
    assert result.loc[result["deck_id"] == 1, "faction"].iloc[0] == "Ordis"

    # Cluster intra-faction : valeurs numériques
    assert result["cluster"].notna().all()