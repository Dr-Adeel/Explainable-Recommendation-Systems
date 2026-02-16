from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data/image_hf/processed")
CATALOG_PATH = DATA_DIR / "catalog.parquet"
EMB_PATH = DATA_DIR / "image_embeddings.parquet"


def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing: {CATALOG_PATH}")
    df = pd.read_parquet(CATALOG_PATH)

    if "product_id" not in df.columns:
        raise ValueError("catalog.parquet must contain 'product_id'")

    df["product_id"] = df["product_id"].astype(int)
    return df


def load_embeddings() -> pd.DataFrame:
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing: {EMB_PATH}")
    df = pd.read_parquet(EMB_PATH)

    required = {"product_id", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"image_embeddings.parquet missing columns: {missing}")

    df["product_id"] = df["product_id"].astype(int)
    return df


def build_index(X: np.ndarray, k: int) -> NearestNeighbors:
    # cosine distance = 1 - cosine similarity (vectors should already be normalized)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
    nn.fit(X)
    return nn


def get_neighbors_indices(nn: NearestNeighbors, X: np.ndarray, q_idx: int, k: int) -> List[int]:
    distances, indices = nn.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=k + 1)
    # remove self
    neigh = [i for i in indices[0].tolist() if i != q_idx][:k]
    return neigh


def precision_at_k_category(query_label: str, neighbor_labels: List[str]) -> float:
    if not neighbor_labels:
        return 0.0
    hits = sum(1 for x in neighbor_labels if x == query_label)
    return hits / len(neighbor_labels)


def hitrate_at_k_category(query_label: str, neighbor_labels: List[str]) -> float:
    if not neighbor_labels:
        return 0.0
    return 1.0 if any(x == query_label for x in neighbor_labels) else 0.0


def evaluate(
    df_eval: pd.DataFrame,
    labels: Dict[int, str],
    nn: NearestNeighbors,
    X: np.ndarray,
    product_ids: List[int],
    k: int,
    n_queries: int,
    seed: int,
) -> Tuple[float, float]:
    rng = random.Random(seed)

    idx_by_pid = {pid: i for i, pid in enumerate(product_ids)}
    pids = [pid for pid in df_eval["product_id"].astype(int).tolist() if pid in idx_by_pid and pid in labels]

    if len(pids) == 0:
        raise RuntimeError("No overlapping product_id between catalog labels and embeddings.")

    if n_queries is not None and len(pids) > n_queries:
        pids = rng.sample(pids, n_queries)

    precisions = []
    hitrates = []

    for pid in pids:
        q_idx = idx_by_pid[pid]
        q_label = labels[pid]

        neigh_idx = get_neighbors_indices(nn, X, q_idx, k)
        neigh_pids = [product_ids[i] for i in neigh_idx]
        neigh_labels = [labels.get(npid, "") for npid in neigh_pids if labels.get(npid, "") != ""]

        precisions.append(precision_at_k_category(q_label, neigh_labels))
        hitrates.append(hitrate_at_k_category(q_label, neigh_labels))

    return float(np.mean(precisions)), float(np.mean(hitrates))


def evaluate_random_baseline(
    df_eval: pd.DataFrame,
    labels: Dict[int, str],
    all_pids: List[int],
    k: int,
    n_queries: int,
    seed: int,
) -> Tuple[float, float]:
    rng = random.Random(seed)

    pids = [pid for pid in df_eval["product_id"].astype(int).tolist() if pid in labels]
    if n_queries is not None and len(pids) > n_queries:
        pids = rng.sample(pids, n_queries)

    eligible = [pid for pid in all_pids if pid in labels]
    if len(eligible) <= k:
        raise RuntimeError("Not enough eligible products for random baseline.")

    precisions = []
    hitrates = []

    for pid in pids:
        q_label = labels[pid]
        # sample neighbors at random excluding the query pid
        pool = [x for x in eligible if x != pid]
        neigh_pids = rng.sample(pool, k)
        neigh_labels = [labels[npid] for npid in neigh_pids]

        precisions.append(precision_at_k_category(q_label, neigh_labels))
        hitrates.append(hitrate_at_k_category(q_label, neigh_labels))

    return float(np.mean(precisions)), float(np.mean(hitrates))


def main():
    # Parameters
    K = 10
    N_QUERIES = 1000      # increase later (e.g., 5000) for more stable numbers
    SEED = 42

    # Choose label level for evaluation:
    # 'articleType' is usually the best (more specific than masterCategory).
    LABEL_COL = "articleType"  # fallback order handled below

    catalog = load_catalog()
    emb = load_embeddings()

    df = emb.merge(catalog, on="product_id", how="left")

    # pick a usable label column
    label_col = None
    for col in [LABEL_COL, "subCategory", "masterCategory", "gender"]:
        if col in df.columns and df[col].notna().any():
            label_col = col
            break
    if label_col is None:
        raise ValueError("No usable label column found in catalog for evaluation.")

    df = df.dropna(subset=["embedding", label_col]).copy()
    df[label_col] = df[label_col].astype(str)

    product_ids = df["product_id"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype("float32")

    labels = {int(pid): lab for pid, lab in zip(df["product_id"].astype(int), df[label_col].astype(str))}

    print(f"label_col={label_col}")
    print(f"rows={len(df)} dim={X.shape[1]} K={K} n_queries={min(N_QUERIES, len(df))}")

    nn = build_index(X, k=K)

    prec, hit = evaluate(df, labels, nn, X, product_ids, k=K, n_queries=N_QUERIES, seed=SEED)
    bprec, bhit = evaluate_random_baseline(df, labels, product_ids, k=K, n_queries=N_QUERIES, seed=SEED)

    print("\nImage reco evaluation (category-based)")
    print(f"Model   Precision@{K}={prec:.4f}  HitRate@{K}={hit:.4f}")
    print(f"Random  Precision@{K}={bprec:.4f}  HitRate@{K}={bhit:.4f}")


if __name__ == "__main__":
    main()
