from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import scipy.sparse as sp

# pip install implicit
from implicit.als import AlternatingLeastSquares


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csr(npz_path: Path) -> sp.csr_matrix:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing CSR: {npz_path}")
    X = sp.load_npz(npz_path).tocsr()
    X.sort_indices()
    return X


def build_popularity_from_csr(X: sp.csr_matrix, topn: int = 50) -> List[int]:
    # item popularity = sum over users
    item_scores = np.asarray(X.sum(axis=0)).ravel()
    top_items = np.argsort(-item_scores)[:topn]
    return top_items.astype(int).tolist()


def main() -> None:
    # --- Paths ---
    processed_dir = Path("data/amazon/processed_small")
    als_dir = processed_dir / "als"
    mappings_dir = processed_dir / "mappings"

    csr_path = als_dir / "train_csr.npz"
    meta_path = als_dir / "als_meta.json"  # optional
    model_dir = als_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Hyperparams (good baseline) ---
    factors = 64
    iterations = 20
    regularization = 0.05
    alpha = 20.0  # confidence scaling for implicit ALS
    random_state = 42

    # --- Load data ---
    X = load_csr(csr_path)

    # implicit expects item-user matrix
    Cui = (X * alpha).T.tocsr()

    # --- Train ---
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
        use_gpu=False,
    )
    model.fit(Cui)

    # --- Save embeddings (npz) ---
    # user_factors: (users, factors)
    # item_factors: (items, factors)
    np.save(model_dir / "user_factors.npy", model.user_factors.astype("float32"))
    np.save(model_dir / "item_factors.npy", model.item_factors.astype("float32"))

    # --- Save config/meta ---
    out_meta = {
        "processed_dir": str(processed_dir),
        "csr_path": str(csr_path),
        "shape_users_items": [int(X.shape[0]), int(X.shape[1])],
        "nnz": int(X.nnz),
        "alpha": float(alpha),
        "factors": int(factors),
        "iterations": int(iterations),
        "regularization": float(regularization),
        "random_state": int(random_state),
    }
    if meta_path.exists():
        out_meta["csr_stats"] = load_json(meta_path)

    save_json(model_dir / "train_meta.json", out_meta)

    # --- Quick test ---
    # Recommend for a user_idx (0..users-1)
    user_idx = 0
    N = 10

    # seen items for that user
    seen = set(X[user_idx].indices.tolist())

    recs, scores = model.recommend(
        userid=user_idx,
        user_items=X[user_idx],
        N=N,
        filter_already_liked_items=True,
    )

    pop = build_popularity_from_csr(X, topn=10)

    print("ALS trained and saved.")
    print(f"Model dir: {model_dir}")
    print(f"Example user_idx={user_idx}")
    print(f"Seen items count: {len(seen)}")
    print(f"Recs (item_idx): {recs.tolist()}")
    print(f"Popularity fallback top-10 (item_idx): {pop}")


if __name__ == "__main__":
    main()
