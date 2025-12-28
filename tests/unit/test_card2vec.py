from src.embeddings.card2vec import train_card2vec, extract_card_embeddings

def test_train_card2vec_dimension(card2vec_config, small_decks):

    model = train_card2vec(small_decks, card2vec_config)

    assert model.wv.vector_size == card2vec_config["vector_size"]
    assert "A" in model.wv

def test_extract_card_embeddings_shape(card2vec_config, small_decks):

    model = train_card2vec(small_decks, card2vec_config)
    vectors = extract_card_embeddings(model)

    assert len(vectors) == 4
    assert all(
        v.shape == (card2vec_config["vector_size"],)
        for v in vectors.values()
    )