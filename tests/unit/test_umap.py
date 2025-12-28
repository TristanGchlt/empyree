import numpy as np
from umap import UMAP
import pytest

from src.projection.umap_utils import fit_umap, transform_umap

def test_fit_umap_runs():
    X = np.random.rand(50, 10)

    umap = fit_umap(
        X,
        n_components=2,
        random_state=42,
    )

    assert isinstance(umap, UMAP)

def test_umap_transform_shape():
    X_train = np.random.rand(50, 10)
    X_test = np.random.rand(10, 10)

    umap = fit_umap(X_train, n_components=2)
    coords = transform_umap(umap, X_test)

    assert coords.shape == (10, 2)
    assert list(coords.columns) == ["x", "y"]

def test_umap_no_nan():
    X = np.random.rand(30, 8)

    umap = fit_umap(X, n_components=3)
    coords = transform_umap(umap, X)

    values = coords.to_numpy()

    assert not np.isnan(values).any()
    assert np.isfinite(values).all()

def test_umap_dimension_mismatch_raises():
    X_train = np.random.rand(50, 10)
    X_bad = np.random.rand(10, 12)

    umap = fit_umap(X_train)

    with pytest.raises(ValueError):
        transform_umap(umap, X_bad)