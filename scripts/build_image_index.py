from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data/image_hf")
EMB_PATH = DATA_DIR / "processed" / "image_embeddings.parquet"


def load_embeddings() -> pd.DataFrame:
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings: {EMB_PATH}")
    df = pd.read_parquet(EMB_PATH)

    required = {"product_id", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"image_embeddings.parquet missing columns: {missing}")

    return df.reset_index(drop=True)


def build_index(embeddings: np.ndarray) -> NearestNeighbors:
    nn = NearestNeighbors(
        n_neighbors=11,          # 1 self + 10 neighbors
        metric="cosine",
        algorithm="auto"
    )
    nn.fit(embeddings)
    return nn


def find_similar(
    nn: NearestNeighbors,
    embeddings: np.ndarray,
    product_ids: List[int],
    query_pid: int,
    k: int = 10
) -> List[int]:
    idx_map = {pid: i for i, pid in enumerate(product_ids)}
    if query_pid not in idx_map:
        raise ValueError(f"product_id not found: {query_pid}")

    q_idx = idx_map[query_pid]
    distances, indices = nn.kneighbors(embeddings[q_idx].reshape(1, -1), n_neighbors=k + 1)

    # Remove self (distance 0)
    neigh_idxs = [i for i in indices[0] if i != q_idx][:k]
    return [product_ids[i] for i in neigh_idxs]


def main():
    print("Loading embeddings...")
    df = load_embeddings()

    product_ids = df["product_id"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype("float32")

    print(f"Rows={len(df)}, dim={X.shape[1]}")

    print("Building NearestNeighbors index...")
    nn = build_index(X)

    # ---- Test query ----
    query_pid = product_ids[52]  # change this to test other products
    k = 10

    similar_ids = find_similar(nn, X, product_ids, query_pid, k=k)

    print(f"Query product_id={query_pid}")
    print("Top similar product_ids:")
    for pid in similar_ids:
        print(pid)


if __name__ == "__main__":
    main()
