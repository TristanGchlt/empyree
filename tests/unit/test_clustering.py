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

def test_cluster_small_group_returns_single_cluster():
    """
    Pour un groupe trop petit, tous les decks doivent
    être assignés au même cluster (0).
    """

    df = pd.DataFrame({
        "faction": ["A", "A"],
        "vector_0": [0.0, 1.0],
        "vector_1": [0.0, 1.0],
    })

    labels = cluster_decks(df)

    assert labels.nunique() == 1
    assert labels.iloc[0] == 0