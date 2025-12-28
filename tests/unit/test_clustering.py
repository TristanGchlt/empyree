import numpy as np
import pandas as pd

from src.clustering.deck_clustering import cluster_decks

def test_cluster_labels_exist():
    df = pd.DataFrame({
        "faction": ["A"] * 10,
        **{f"vector_{i}": np.random.rand(10) for i in range(5)}
    })

    labels = cluster_decks(df)

    assert len(labels) == 10
    assert labels.notna().all()