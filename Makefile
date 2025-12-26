install_dependencies:
	poetry install --no-root

decode:
	node scripts/decode_decks.js

preprocess:
	poetry run python scripts/build_corpus.py

train: 
	poetry run python scripts/train_card2vec.py

cluster:
	poetry run python scripts/run_clustering.py

run_dashboard:
	poetry run python -m src.dashboard.app