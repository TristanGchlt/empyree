from gensim.models import Word2Vec
from typing import Dict
import numpy as np

def train_card2vec(sentences, config: dict) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        workers=config["workers"],
        sg=config["sg"],
        epochs=config["epochs"],
    )

def extract_card_embeddings(model: Word2Vec) -> Dict[str, np.ndarray]:
    return {card: model.wv[card] for card in model.wv.index_to_key}