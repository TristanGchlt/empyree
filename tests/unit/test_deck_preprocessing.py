from src.preprocessing.deck_preprocessing import preprocess_deck

def test_preprocess_basic():
    deck_string = "[3 CORE_ABC\\n2 CORE_DEF\\n1 CORE_U_XYZ]"
    processed = preprocess_deck(deck_string)

    # CORE_U_XYZ est unique -> doit être retiré
    # Quantités doivent être correctement étendues
    assert processed == ["CORE_ABC","CORE_ABC","CORE_ABC","CORE_DEF","CORE_DEF"]

def test_preprocess_empty_unique():
    deck_string = "[1 CORE_U_XYZ]"
    processed = preprocess_deck(deck_string)

    # Un deck composé uniquement d'uniques -> liste vide
    assert processed == []

def test_preprocess_rename_ks():
    deck_string = "[1 COREKS_ABC]"
    processed = preprocess_deck(deck_string)

    # COREKS doit être renommé en CORE
    assert processed == ["CORE_ABC"]