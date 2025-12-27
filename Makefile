install_dependencies:
	poetry install --no-root

preprocess:
	poetry run python scripts/build_corpus.py

train: 
	poetry run python scripts/train_card2vec.py

cluster:
	poetry run python scripts/run_clustering.py

project:
	poetry run python scripts/run_umap.py

run_dashboard:
	poetry run python -m src.dashboard.app

all: preprocess train project cluster

clean:
	rm -rf runs/*
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache