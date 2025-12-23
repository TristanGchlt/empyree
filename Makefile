preprocess:
	poetry run python scripts/build_corpus.py

train: 
	poetry run python scripts/train_card2vec.py

install_dependencies:
	poetry install --no-root