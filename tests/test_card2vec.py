import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.embeddings.card2vec import train_card2vec, extract_card_embeddings

def test_train_card2vec_dimension():
    sentences = [["A", "B", "C"], ["A", "C"], ["B", "C"]]
    cfg = {
        "vector_size": 10,
        "window": 2,
        "min_count": 1,
        "workers": 1,
        "sg": 1,
        "epochs": 5,
    }

    model = train_card2vec(sentences, cfg)

    assert model.wv.vector_size == 10
    assert "A" in model.wv

def test_extract_card_embeddings_shape():
    sentences = [["A", "B"], ["A", "C"]]
    cfg = {...}

    model = train_card2vec(sentences, cfg)
    embeddings = extract_card_embeddings(model)

    assert isinstance(embeddings, dict)
    assert all(len(v) == cfg["vector_size"] for v in embeddings.values())