import pandas as pd

from src.clustering.deck_clustering_pipeline import assign_deck_factions


def test_assign_deck_factions_basic():
    deck_df = pd.DataFrame({
        "deck_id": [0, 1],
        "cards": [
            "['CARD_A', 'CARD_B']",
            "['CARD_C']"
        ]
    })

    card_to_faction = {
        "CARD_A": "Ordis",
        "CARD_C": "Lyra",
    }

    result = assign_deck_factions(deck_df, card_to_faction)

    assert "faction" in result.columns
    assert list(result["faction"]) == ["Ordis", "Lyra"]

def test_assign_deck_factions_unknown_card():
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["['UNKNOWN_CARD']"]
    })

    card_to_faction = {}

    result = assign_deck_factions(deck_df, card_to_faction)

    assert result.loc[0, "faction"] == "Unknown"

def test_assign_deck_factions_malformed_cards():
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["not_a_list"]
    })

    card_to_faction = {"CARD_A": "Ordis"}

    result = assign_deck_factions(deck_df, card_to_faction)

    assert result.loc[0, "faction"] == "Unknown"