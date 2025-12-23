from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pathlib import Path
import yaml

# Chemins
project_root = Path(__file__).resolve().parents[1]
corpus_file = project_root / "data" / "processed" / "card_corpus.txt"
model_file = project_root / "models" / "card2vec.model"

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
model.save(str(model_file))
model.wv.save_word2vec_format(str(project_root / "models" / "card2vec_vectors.txt"), binary=False)

print(f"Modèle sauvegardé dans {model_file}")