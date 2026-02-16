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


def build_index(X: np.ndarray) -> NearestNeighbors:
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X)
    return nn


def neighbors_pool(nn: NearestNeighbors, X: np.ndarray, q_idx: int, pool: int) -> List[int]:
    distances, indices = nn.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=pool + 1)
    return [i for i in indices[0].tolist() if i != q_idx][:pool]


def precision_at_k(query_label: str, neighbor_labels: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = neighbor_labels[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x == query_label) / len(top)


def hitrate_at_k(query_label: str, neighbor_labels: List[str], k: int) -> float:
    top = neighbor_labels[:k]
    if not top:
        return 0.0
    return 1.0 if any(x == query_label for x in top) else 0.0


def eval_unfiltered(
    pids: List[int],
    product_ids: List[int],
    labels: Dict[int, str],
    nn: NearestNeighbors,
    X: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    idx_by_pid = {pid: i for i, pid in enumerate(product_ids)}
    precisions, hitrates = [], []

    for pid in pids:
        q_idx = idx_by_pid[pid]
        neigh_idx = neighbors_pool(nn, X, q_idx, pool=k)  # pool=k means direct top-k
        neigh_pids = [product_ids[i] for i in neigh_idx]
        neigh_labels = [labels.get(npid, "") for npid in neigh_pids]

        q_label = labels[pid]
        precisions.append(precision_at_k(q_label, neigh_labels, k))
        hitrates.append(hitrate_at_k(q_label, neigh_labels, k))

    return float(np.mean(precisions)), float(np.mean(hitrates))


def eval_filtered(
    pids: List[int],
    product_ids: List[int],
    labels: Dict[int, str],
    nn: NearestNeighbors,
    X: np.ndarray,
    k: int,
    pool: int,
) -> Tuple[float, float]:
    idx_by_pid = {pid: i for i, pid in enumerate(product_ids)}
    precisions, hitrates = [], []

    for pid in pids:
        q_idx = idx_by_pid[pid]
        pool_idx = neighbors_pool(nn, X, q_idx, pool=pool)
        pool_pids = [product_ids[i] for i in pool_idx]

        q_label = labels[pid]

        same = [x for x in pool_pids if labels.get(x) == q_label]
        other = [x for x in pool_pids if labels.get(x) != q_label]

        rec = same[:k]
        if len(rec) < k:
            rec.extend(other[: (k - len(rec))])

        neigh_labels = [labels.get(npid, "") for npid in rec]
        precisions.append(precision_at_k(q_label, neigh_labels, k))
        hitrates.append(hitrate_at_k(q_label, neigh_labels, k))

    return float(np.mean(precisions)), float(np.mean(hitrates))


def main():
    # Params
    K = 10
    POOL = 200
    N_QUERIES = 1000
    SEED = 42
    LABEL_COL = "articleType"

    catalog = load_catalog()
    emb = load_embeddings()

    if LABEL_COL not in catalog.columns:
        raise ValueError("catalog.parquet must contain 'articleType'.")

    df = emb.merge(catalog[["product_id", LABEL_COL]], on="product_id", how="left")
    df = df.dropna(subset=["embedding", LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    product_ids = df["product_id"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype("float32")
    labels = dict(zip(df["product_id"].astype(int), df[LABEL_COL].astype(str)))

    nn = build_index(X)

    rng = random.Random(SEED)
    valid_pids = [pid for pid in product_ids if pid in labels]
    if len(valid_pids) > N_QUERIES:
        valid_pids = rng.sample(valid_pids, N_QUERIES)

    prec_u, hit_u = eval_unfiltered(valid_pids, product_ids, labels, nn, X, k=K)
    prec_f, hit_f = eval_filtered(valid_pids, product_ids, labels, nn, X, k=K, pool=POOL)

    print(f"label_col={LABEL_COL}")
    print(f"rows={len(df)} K={K} pool={POOL} n_queries={len(valid_pids)}")
    print("\nBefore/After filtering on articleType")
    print(f"Unfiltered  Precision@{K}={prec_u:.4f}  HitRate@{K}={hit_u:.4f}")
    print(f"Filtered    Precision@{K}={prec_f:.4f}  HitRate@{K}={hit_f:.4f}")


if __name__ == "__main__":
    main()
