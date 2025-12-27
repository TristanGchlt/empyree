import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering.deck_clustering import cluster_decks

def test_cluster_labels_exist():
    df = pd.DataFrame({
        "faction": ["A"] * 10,
        **{f"vector_{i}": np.random.rand(10) for i in range(5)}
    })

    labels = cluster_decks(df)

    assert len(labels) == 10
    assert labels.notna().all()