import pandas as pd
import ast
from typing import Dict

from .deck_clustering import cluster_decks

def assign_deck_factions(
    deck_df: pd.DataFrame,
    card_to_faction: Dict[str, str],
    cards_column: str = "cards",
) -> pd.DataFrame:
    """
    Assigne une faction à chaque deck en utilisant la faction de sa première carte.
    """
    df = deck_df.copy()

    def infer_faction(cards):
        try:
            card_list = ast.literal_eval(cards)
            if not card_list:
                return "Unknown"
            return card_to_faction.get(card_list[0], "Unknown")
        except Exception:
            return "Unknown"

    df["faction"] = df[cards_column].apply(infer_faction)
    return df

def assign_deck_hero(
    deck_df: pd.DataFrame,
    card_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assigne le héros unique de chaque deck.

    Prérequis :
    - deck_df contient une colonne 'cards' (liste sérialisée)
    - card_metadata contient 'reference' et 'card_type'
    - card_type == 'Héros' identifie les héros

    Raises:
        ValueError si aucun ou plusieurs héros sont trouvés
    """

    hero_cards = set(
        card_metadata.loc[
            card_metadata["card_type"] == "Héros", "reference"
        ]
    )

    def extract_hero(cards_raw):
        try:
            cards = ast.literal_eval(cards_raw)
        except Exception:
            raise ValueError("Invalid cards format")

        heroes = [c for c in cards if c in hero_cards]

        if len(heroes) == 0:
            raise ValueError("No hero found in deck")
        if len(heroes) > 1:
            raise ValueError("Multiple heroes found in deck")

        return heroes[0]

    df = deck_df.copy()
    df["hero"] = df["cards"].apply(extract_hero)

    return df


def run_deck_clustering(
    deck_embeddings: pd.DataFrame,
    card_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pipeline complet :
    - détection du héros
    - récupération de la faction via le héros
    - clustering intra-héros
    """

    df = deck_embeddings.copy()

    df = assign_deck_hero(df, card_metadata)

    hero_to_faction = (
        card_metadata
        .set_index("reference")["faction"]
        .to_dict()
    )

    df["faction"] = df["hero"].map(hero_to_faction)

    if df["faction"].isna().any():
        raise ValueError("Faction missing for at least one hero")

    df["cluster"] = cluster_decks(
        df,
        group_column="hero",
    )

    return df[["deck_id", "faction", "hero", "cluster"]]