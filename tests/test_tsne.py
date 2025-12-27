import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering.tsne_utils import compute_tsne

def test_tsne_output_shape():
    X = np.random.rand(20, 10)
    ids = list(range(20))

    tsne = compute_tsne(X, ids, n_components=2, perplexity=6)

    assert tsne.shape == (20, 3)  # id + x + y
    assert tsne[["x", "y"]].isna().sum().sum() == 0