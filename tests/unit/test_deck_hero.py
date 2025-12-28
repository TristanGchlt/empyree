import pandas as pd
from src.clustering.deck_clustering_pipeline import assign_deck_hero
import pytest

def test_assign_deck_hero_basic(card_metadata_with_heroes):
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["['H1', 'A', 'B']"]
    })

    result = assign_deck_hero(deck_df, card_metadata_with_heroes)

    assert result.loc[0, "hero"] == "H1"

def test_assign_deck_hero_multiple_decks(card_metadata_with_heroes):
    deck_df = pd.DataFrame({
        "deck_id": [0, 1],
        "cards": [
            "['H1', 'A']",
            "['H2', 'C']",
        ]
    })

    result = assign_deck_hero(deck_df, card_metadata_with_heroes)

    heroes = result.set_index("deck_id")["hero"].to_dict()

    assert heroes[0] == "H1"
    assert heroes[1] == "H2"

def test_assign_deck_hero_missing_hero_raises(card_metadata_with_heroes):
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["['A', 'B', 'C']"]
    })

    with pytest.raises(ValueError, match="No hero found"):
        assign_deck_hero(deck_df, card_metadata_with_heroes)

def test_assign_deck_hero_multiple_heroes_raises(card_metadata_with_heroes):
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["['H1', 'H2', 'A']"]
    })

    with pytest.raises(ValueError, match="Multiple heroes found"):
        assign_deck_hero(deck_df, card_metadata_with_heroes)

def test_assign_deck_hero_malformed_cards_raises(card_metadata_with_heroes):
    deck_df = pd.DataFrame({
        "deck_id": [0],
        "cards": ["not_a_list"]
    })

    with pytest.raises(ValueError, match="Invalid cards format"):
        assign_deck_hero(deck_df, card_metadata_with_heroes)