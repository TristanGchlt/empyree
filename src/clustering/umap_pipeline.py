from pathlib import Path
import numpy as np
from typing import Iterable, List

from .umap_utils import fit_umap, transform_umap, save_umap_model, load_umap_model

def run_reference_umap(
    X: np.ndarray,
    ids: Iterable,
    output_model_dir: Path,
    output_proj_dir: Path,
    prefix: str,
    dims: List[int] = [2, 3],
    **umap_params,
):
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_proj_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        umap = fit_umap(X, n_components=dim, **umap_params)
        coords = transform_umap(umap, X, ids=ids)

        save_umap_model(umap, output_model_dir / f"{prefix}_{dim}d.joblib")
        coords.to_csv(output_proj_dir / f"{prefix}_{dim}d.csv", index=False)

def run_projection_umap(
    X: np.ndarray,
    ids: Iterable,
    model_dir: Path,
    output_proj_dir: Path,
    prefix: str,
    dims: List[int] = [2, 3],
):
    output_proj_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        umap = load_umap_model(model_dir / f"umap_cards_{dim}d.joblib")
        coords = transform_umap(umap, X, ids=ids)

        coords.to_csv(output_proj_dir / f"{prefix}_{dim}d.csv", index=False)