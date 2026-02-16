from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from scipy import sparse


def load_multimodal_embeddings(path: Path) -> Tuple[np.ndarray, list]:
    df = pd.read_parquet(path)
    ids = df['item_idx'].astype(int).tolist()
    X = np.vstack(df['embedding'].values).astype(np.float32)
    return X, ids


def load_als_item_factors() -> Tuple[Optional[np.ndarray], Optional[Path]]:
    # try common paths
    cand = [Path('data/amazon/processed/als/model'), Path('data/amazon/processed_small/als/model')]
    for c in cand:
        itf = c / 'item_factors.npy'
        if itf.exists():
            return np.load(itf), c
    return None, None


def _load_popularity(model_dir: Optional[Path]) -> Optional[np.ndarray]:
    """Load normalised item popularity from the CSR interaction matrix.

    Returns a 1-d array of shape (n_items,) with values in [0, 1].
    """
    if model_dir is None:
        return None

    csr_path = model_dir.parent / "train_csr.npz"
    if not csr_path.exists():
        return None

    try:
        X = sparse.load_npz(csr_path).tocsr()
        item_pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        mx = item_pop.max()
        if mx > 0:
            return (item_pop - item_pop.min()) / (mx - item_pop.min() + 1e-12)
        return item_pop
    except Exception:
        return None


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-d array to [0, 1] — mirrors _normalize_scores in the API."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def build_training_data(
    mm_X: np.ndarray,
    mm_ids: list,
    als_item_factors: Optional[np.ndarray] = None,
    pop_scores: Optional[np.ndarray] = None,
    n_samples: int = 20000,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    pool_size: int = 200,
):
    """Generate training data that mirrors the real hybrid engine.

    For each sample we pick a random query item and a pool of neighbours,
    compute the three raw signals, **normalise each to [0,1]** (just like the
    API's ``_normalize_scores``), and combine with the hybrid weights to get
    the target score.  This ensures the surrogate learns the same relative
    importance that the real engine uses.
    """
    rng = np.random.default_rng(42)
    n_items = mm_X.shape[0]

    # Pre-compute norms for fast cosine
    norms = np.linalg.norm(mm_X, axis=1, keepdims=True) + 1e-12
    mm_normed = mm_X / norms

    all_feats = []
    all_targets = []

    n_queries = max(1, n_samples // pool_size)

    for _ in range(n_queries):
        q = rng.integers(0, n_items)
        pool_idx = rng.choice(n_items, size=min(pool_size, n_items), replace=False)

        # ── Raw cosine similarities (query vs pool) ──
        q_vec = mm_normed[q]
        cos_raw = (mm_normed[pool_idx] @ q_vec).astype(np.float32)

        # ── Raw ALS dot products ──
        als_raw = np.zeros(len(pool_idx), dtype=np.float32)
        if als_item_factors is not None:
            try:
                q_id = mm_ids[q]
                q_factor = als_item_factors[q_id].astype(np.float32)
                for i, pidx in enumerate(pool_idx):
                    try:
                        p_id = mm_ids[pidx]
                        als_raw[i] = float(als_item_factors[p_id].astype(np.float32) @ q_factor)
                    except (IndexError, KeyError):
                        als_raw[i] = 0.0
            except (IndexError, KeyError):
                pass

        # ── Raw popularity of each candidate ──
        pop_raw = np.zeros(len(pool_idx), dtype=np.float32)
        if pop_scores is not None:
            for i, pidx in enumerate(pool_idx):
                try:
                    pop_raw[i] = float(pop_scores[mm_ids[pidx]])
                except (IndexError, KeyError):
                    pop_raw[i] = 0.0

        # ── Normalise each signal to [0,1] per pool (same as API) ──
        cos_n = _minmax_norm(cos_raw)
        als_n = _minmax_norm(als_raw)
        pop_n = _minmax_norm(pop_raw)

        # ── Target = hybrid combination of normalised signals ──
        targets = alpha * cos_n + beta * als_n + gamma * pop_n

        # ── Store features (normalised) and targets ──
        for i in range(len(pool_idx)):
            all_feats.append([float(cos_n[i]), float(als_n[i]), float(pop_n[i])])
            all_targets.append(float(targets[i]))

    X = np.asarray(all_feats, dtype=np.float32)
    y = np.asarray(all_targets, dtype=np.float32)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm[:n_samples]], y[perm[:n_samples]]


def train_and_save_surrogate(mm_path: Path, out_path: Path, n_samples: int = 20000):
    mm_X, mm_ids = load_multimodal_embeddings(mm_path)
    als_item_factors, model_dir = load_als_item_factors()

    # Load popularity from CSR matrix
    pop_scores = _load_popularity(model_dir)

    X, y = build_training_data(
        mm_X, mm_ids,
        als_item_factors=als_item_factors,
        pop_scores=pop_scores,
        n_samples=n_samples,
    )

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Report importances
    imp = rf.feature_importances_
    names = ['multimodal_cosine', 'als_dot', 'popularity']
    print("Feature importances:")
    for n, v in zip(names, imp):
        print(f"  {n}: {v * 100:.2f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': rf, 'mm_ids': mm_ids}, out_path)
    print(f"Saved surrogate model to {out_path}")


def load_surrogate(path: Path):
    return joblib.load(path)
