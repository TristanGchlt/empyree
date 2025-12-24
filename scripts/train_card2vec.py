from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pathlib import Path
import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

corpus_file = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"

# Préparer le corpus
sentences = LineSentence(str(corpus_file))

# Récupératiou des paramètres
with open("configs/card2vec.yaml") as f:
    cfg = yaml.safe_load(f)

# Entraîner le modèle
model = Word2Vec(
    sentences=sentences,
    vector_size=cfg['vector_size'],
    window=cfg['window'],
    min_count=cfg['min_count'],
    workers=cfg['workers'],
    sg=cfg['sg'],
    epochs=cfg['epochs']
)

# Sauvegarder
model.save(str(PROJECT_ROOT / "models" / "card2vec.model"))
model.wv.save_word2vec_format(str(PROJECT_ROOT / "models" / "card2vec_vectors.txt"), binary=False)