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

def run_deck_clustering(
    deck_embeddings: pd.DataFrame,
    card_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Effectue un clustering intra-faction des decks.
    """
    card_to_faction = card_metadata.set_index("reference")["faction"].to_dict()

    df = assign_deck_factions(deck_embeddings, card_to_faction)
    df["cluster"] = cluster_decks(df)

    return df[["deck_id", "faction", "cluster"]]