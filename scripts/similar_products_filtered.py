from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data/image_hf/processed")
CATALOG_PATH = DATA_DIR / "catalog.parquet"
EMB_PATH = DATA_DIR / "image_embeddings.parquet"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing: {CATALOG_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing: {EMB_PATH}")

    catalog = pd.read_parquet(CATALOG_PATH)
    emb = pd.read_parquet(EMB_PATH)

    catalog["product_id"] = catalog["product_id"].astype(int)
    emb["product_id"] = emb["product_id"].astype(int)

    return catalog, emb


def build_index(df: pd.DataFrame) -> Tuple[NearestNeighbors, np.ndarray, List[int]]:
    product_ids = df["product_id"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype("float32")

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X)

    return nn, X, product_ids


def get_neighbors_pool(nn: NearestNeighbors, X: np.ndarray, q_idx: int, pool: int) -> List[int]:
    distances, indices = nn.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=pool + 1)
    return [i for i in indices[0].tolist() if i != q_idx][:pool]


def recommend_similar_filtered(
    query_pid: int,
    k: int,
    pool: int,
    nn: NearestNeighbors,
    X: np.ndarray,
    product_ids: List[int],
    pid_to_type: Dict[int, str],
) -> Tuple[str, List[int]]:
    idx_by_pid = {pid: i for i, pid in enumerate(product_ids)}
    if query_pid not in idx_by_pid:
        raise ValueError(f"product_id not found in embeddings: {query_pid}")

    q_type = pid_to_type.get(query_pid)
    if q_type is None:
        raise ValueError(f"articleType missing for query product_id: {query_pid}")

    q_idx = idx_by_pid[query_pid]
    pool_idxs = get_neighbors_pool(nn, X, q_idx, pool=pool)
    pool_pids = [product_ids[i] for i in pool_idxs]

    same_type = [pid for pid in pool_pids if pid_to_type.get(pid) == q_type]
    other_type = [pid for pid in pool_pids if pid_to_type.get(pid) != q_type]

    rec = same_type[:k]
    if len(rec) < k:
        rec.extend(other_type[: (k - len(rec))])

    return q_type, rec


def main():
    # Parameters
    K = 10
    POOL = 200  # candidate pool size

    catalog, emb = load_data()

    # Merge to keep only rows with embeddings + articleType
    if "articleType" not in catalog.columns:
        raise ValueError("catalog.parquet must contain 'articleType' for C1 filtering.")

    df = emb.merge(catalog[["product_id", "articleType", "productDisplayName"]], on="product_id", how="left")
    df = df.dropna(subset=["embedding", "articleType"]).copy()
    df["articleType"] = df["articleType"].astype(str)

    pid_to_type = dict(zip(df["product_id"].astype(int), df["articleType"].astype(str)))
    pid_to_name = dict(zip(df["product_id"].astype(int), df.get("productDisplayName", pd.Series([""] * len(df))).astype(str)))

    nn, X, product_ids = build_index(df)

    # Example query: pick a product id from the dataset
    query_pid = product_ids[0]
    q_type, rec_ids = recommend_similar_filtered(
        query_pid=query_pid,
        k=K,
        pool=POOL,
        nn=nn,
        X=X,
        product_ids=product_ids,
        pid_to_type=pid_to_type,
    )

    print(f"query_product_id={query_pid}")
    print(f"query_articleType={q_type}")
    print("recommendations:")
    for pid in rec_ids:
        print(f"- {pid} | {pid_to_type.get(pid)} | {pid_to_name.get(pid, '')}")


if __name__ == "__main__":
    main()
