import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.embeddings.card2vec import train_card2vec, extract_card_embeddings

@pytest.fixture
def card2vec_config():
    return {
        "vector_size": 10,
        "window": 2,
        "min_count": 1,
        "workers": 1,
        "sg": 1,
        "epochs": 5,
    }

def test_train_card2vec_dimension(card2vec_config):
    sentences = [["A", "B", "C"], ["A", "C"], ["B", "C"]]

    model = train_card2vec(sentences, card2vec_config)

    assert model.wv.vector_size == 10
    assert "A" in model.wv

def test_extract_card_embeddings_shape(card2vec_config):
    sentences = [["A", "B"], ["A", "C"]]

    model = train_card2vec(sentences, card2vec_config)
    vectors = extract_card_embeddings(model)

    assert len(vectors) == 3
    assert all(v.shape == (10,) for v in vectors.values())