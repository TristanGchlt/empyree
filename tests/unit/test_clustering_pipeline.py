import pandas as pd

from src.clustering.deck_clustering_pipeline import run_deck_clustering

def test_run_deck_clustering_basic(small_card_metadata):
    deck_embeddings = pd.DataFrame({
        "deck_id": [0, 1, 2],
        "cards": [
            "['A']",
            "['A']",
            "['C']"
        ],
        "vector_0": [0.0, 0.1, 10.0],
        "vector_1": [0.0, 0.1, 10.0],
    })

    result = run_deck_clustering(
        deck_embeddings=deck_embeddings,
        card_metadata=small_card_metadata,
    )

    # Structure de sortie
    assert set(result.columns) == {"deck_id", "faction", "cluster"}
    assert len(result) == 3

    faction_by_deck = result.set_index("deck_id")["faction"].to_dict()

    # Factions assignées par deck
    assert faction_by_deck[0] == "Ordis"
    assert faction_by_deck[1] == "Ordis"
    assert faction_by_deck[2] == "Lyra"


    # Cluster intra-faction valides
    assert result["cluster"].notna().all()
    assert result["cluster"].apply(lambda x: isinstance(x, int)).all()