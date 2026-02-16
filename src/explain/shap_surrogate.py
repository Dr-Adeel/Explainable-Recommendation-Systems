from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import joblib


def load_multimodal_embeddings(path: Path) -> Tuple[np.ndarray, list]:
    df = pd.read_parquet(path)
    ids = df['item_idx'].astype(int).tolist()
    X = np.vstack(df['embedding'].values).astype(np.float32)
    return X, ids


def load_als_item_factors() -> Tuple[np.ndarray, None]:
    # try common paths
    cand = [Path('data/amazon/processed/als/model'), Path('data/amazon/processed_small/als/model')]
    for c in cand:
        itf = c / 'item_factors.npy'
        if itf.exists():
            return np.load(itf), c
    return None, None


def build_training_data(mm_X: np.ndarray, mm_ids: list, als_item_factors: np.ndarray | None, n_samples: int = 20000):
    rng = np.random.default_rng(42)
    n_items = mm_X.shape[0]
    pairs = rng.integers(0, n_items, size=(n_samples, 2))

    feats = []
    targets = []
    for a, b in pairs:
        va = mm_X[a]
        vb = mm_X[b]
        # cosine similarity
        sim = float((va @ vb) / ((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12)))

        als_sim = 0.0
        if als_item_factors is not None:
            try:
                ai = als_item_factors[mm_ids[a]]
                bi = als_item_factors[mm_ids[b]]
                als_sim = float(np.dot(ai.astype(np.float32), bi.astype(np.float32)))
            except Exception:
                als_sim = 0.0

        pop = 0.0
        # synthetic target using default hybrid weights
        target = 0.6 * sim + 0.35 * als_sim + 0.05 * pop

        feats.append([sim, als_sim, pop])
        targets.append(target)

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y


def train_and_save_surrogate(mm_path: Path, out_path: Path, n_samples: int = 20000):
    mm_X, mm_ids = load_multimodal_embeddings(mm_path)
    als_item_factors, model_dir = load_als_item_factors()

    X, y = build_training_data(mm_X, mm_ids, als_item_factors, n_samples=n_samples)

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': rf, 'mm_ids': mm_ids}, out_path)
    print(f"Saved surrogate model to {out_path}")


def load_surrogate(path: Path):
    return joblib.load(path)
