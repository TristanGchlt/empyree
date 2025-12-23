def split_cards(cards_string: str) -> list[str]:
    """
    Transforme une decklist string en liste de string par cartes.

    Args:
        cards_string: String de la decklist.

    Returns:
        Liste d'identifiants de cartes.
    """
    return cards_string[1:-1].split("\\n")


def remove_uniques(cards_list: list[str]) -> list[str]:
    """
    Retire les cartes de rareté Unique d'une liste de cartes.

    Args:
        cards_list: Liste des cartes.

    Returns:
        Même liste sans les cartes de rareté Unique.
    """
    return [c for c in cards_list if "_U_" not in c]


def rename_ks_ids(cards_list: list[str]) -> list[str]:
    """
    Modifie les IDs des cartes issues du set Kickstarter pour l'uniformiser avec le set 1.

    Args:
        cards_list: Liste des cartes.

    Returns:
        Même liste avec les KS remplacés par du simple Set 1.
    """
    return [c.replace("COREKS", "CORE") for c in cards_list]


def expand_quantities(cards_list: list[str]) -> list[str]:
    """
    Remplace l'indication des quantités de cartes en mettant autant d'élément dans la liste que la quantité indiquée.

    Args:
        cards_list: Liste des cartes avec indicateurs de quantité.

    Returns:
        Même liste avec répétitions et sans indicateur de quantité.
    """
    deck = []
    for c in cards_list:
        qty = int(c[0])
        card_id = c[2:]
        deck.extend([card_id] * qty)
    return deck

def preprocess_deck(cards_string: str) -> list[str]:
    """
    Effectue la préparation complète des données.

    Steps:
        - Transformation en liste
        - Retrait des cartes de rareté Uniques
        - Regroupement des cartes KS et Set 1
        - Répétition des quantités

    Args:
        cards_string: La string de decklist issue des données décodées.

    Returns:
        La liste des cartes entièrement traitées.
    """
    cards = split_cards(cards_string)
    cards = remove_uniques(cards)
    cards = rename_ks_ids(cards)
    cards = expand_quantities(cards)
    return cards