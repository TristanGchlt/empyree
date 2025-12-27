from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pathlib import Path
import yaml
import sys

# Configuration du Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import des fonctions utiles
from src.embeddings.card2vec import train_card2vec, extract_card_embeddings
from src.clustering.deck_embeddings import compute_all_deck_embeddings

# Path du modèle courant
MODEL_YAML = PROJECT_ROOT / "configs" / "current_model.yaml"
with open(MODEL_YAML) as f:
    model_yaml = yaml.safe_load(f)
    model_name = model_yaml['current_model']

# Path des input et output
INPUT_CORPUS = PROJECT_ROOT / "data" / "processed" / "card_corpus.txt"
INPUT_CONFIG = PROJECT_ROOT / "configs" / "card2vec.yaml"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / model_name
OUTPUT_MODEL = OUTPUT_FOLDER / "model" / "card2vec.model"
OUTPUT_EMBEDDINGS = OUTPUT_FOLDER / "embeddings" / "cards_vectors.txt"
OUTPUT_DECK_EMBEDDINGS = OUTPUT_FOLDER / "embeddings" / "decks_vectors.csv"

# Chargement du corpus
sentences = LineSentence(str(INPUT_CORPUS))

# Récupératiou des paramètres
cfg = yaml.safe_load(open(INPUT_CONFIG))

# Entrainement
model = train_card2vec(sentences, cfg)

# Création des embeddings de cartes
card_embeddings = {card: model.wv[card] for card in model.wv.index_to_key}

# Création des embeddings de decks : Moyenne des embeddings des cartes du deck
deck_embeddings = compute_all_deck_embeddings(INPUT_CORPUS, card_embeddings)

# Sauvegarde du modèle et des embeddings
for sub in ["model", "embeddings", "projections", "clustering"]:
    (OUTPUT_FOLDER / sub).mkdir(parents=True, exist_ok=True)
model.save(str(OUTPUT_MODEL))
model.wv.save_word2vec_format(str(OUTPUT_EMBEDDINGS), binary=False)
deck_embeddings.to_csv(OUTPUT_DECK_EMBEDDINGS, index=False)