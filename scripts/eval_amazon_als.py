from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def load_parquet(path: Path, cols: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path, columns=cols)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def precision_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = len(set(topk) & relevant)
    return hits / float(len(topk))


def recall_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = len(set(topk) & relevant)
    return hits / float(len(relevant))


def build_seen_from_csr(train_csr: sparse.csr_matrix) -> Dict[int, np.ndarray]:
    seen: Dict[int, np.ndarray] = {}
    indptr = train_csr.indptr
    indices = train_csr.indices
    n_users = train_csr.shape[0]
    for u in range(n_users):
        start, end = indptr[u], indptr[u + 1]
        if end > start:
            seen[u] = indices[start:end]
        else:
            seen[u] = np.array([], dtype=np.int32)
    return seen


def recommend_topk(
    user_idx: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items: np.ndarray,
    k: int,
    pool: int = 2000,
) -> List[int]:
    # scores: (n_items,)
    uvec = user_factors[user_idx].astype(np.float32, copy=False)
    scores = item_factors.astype(np.float32, copy=False) @ uvec

    if seen_items.size > 0:
        scores[seen_items] = -1e9

    pool = max(pool, k)
    if pool >= scores.shape[0]:
        top = np.argsort(-scores)
        return top[:k].astype(int).tolist()

    # fast top-pool then sort
    cand = np.argpartition(-scores, pool)[:pool]
    cand = cand[np.argsort(-scores[cand])]
    return cand[:k].astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--als_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--relevance_threshold", type=float, default=4.0)
    ap.add_argument("--max_users", type=int, default=2000)
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    als_dir = Path(args.als_dir)
    model_dir = als_dir / "model"

    # ---- Load interactions ----
    split_path = processed_dir / f"interactions_{args.split}.parquet"

    try:
        eval_df = load_parquet(split_path, cols=["user_id", "parent_asin", "rating"])
        mode = "id"
    except Exception:
        eval_df = load_parquet(split_path, cols=["user_idx", "item_idx", "rating"])
        mode = "idx"

    if mode == "id":
        # ---- Load mappings ----
        # Expected from your pipeline:
        # users.parquet: user_id, user_idx
        # items.parquet: parent_asin, item_idx
        users_map = load_parquet(processed_dir / "users.parquet", cols=["user_id", "user_idx"])
        items_map = load_parquet(processed_dir / "items.parquet", cols=["parent_asin", "item_idx"])

        u2i = dict(zip(users_map["user_id"].astype(str), users_map["user_idx"].astype(int)))
        it2i = dict(zip(items_map["parent_asin"].astype(str), items_map["item_idx"].astype(int)))

        # Map eval_df to indices (drop unknowns)
        eval_df["user_id"] = eval_df["user_id"].astype(str)
        eval_df["parent_asin"] = eval_df["parent_asin"].astype(str)

        eval_df["user_idx"] = eval_df["user_id"].map(u2i)
        eval_df["item_idx"] = eval_df["parent_asin"].map(it2i)
        eval_df = eval_df.dropna(subset=["user_idx", "item_idx"]).copy()
        eval_df["user_idx"] = eval_df["user_idx"].astype(int)
        eval_df["item_idx"] = eval_df["item_idx"].astype(int)
    else:
        eval_df["user_idx"] = eval_df["user_idx"].astype(int)
        eval_df["item_idx"] = eval_df["item_idx"].astype(int)

    # Define relevance based on rating threshold
    eval_df = eval_df[eval_df["rating"].astype(float) >= float(args.relevance_threshold)].copy()

    if eval_df.empty:
        raise RuntimeError(
            f"No relevant interactions found in {args.split} with threshold >= {args.relevance_threshold}."
        )

    # Group relevant items by user
    relevant_by_user = eval_df.groupby("user_idx")["item_idx"].apply(lambda s: set(s.astype(int)))

    # Optionally subsample users for speed
    user_indices = relevant_by_user.index.to_numpy()
    if args.max_users is not None and len(user_indices) > args.max_users:
        rng = np.random.default_rng(42)
        user_indices = rng.choice(user_indices, size=args.max_users, replace=False)

    # ---- Load ALS artifacts ----
    csr_path = als_dir / "train_csr.npz"
    if not csr_path.exists():
        raise FileNotFoundError(f"Missing CSR: {csr_path}")
    train_csr = sparse.load_npz(csr_path).tocsr()

    uf_path = model_dir / "user_factors.npy"
    itf_path = model_dir / "item_factors.npy"
    if not uf_path.exists() or not itf_path.exists():
        raise FileNotFoundError(
            f"Missing factors in {model_dir}. Expected user_factors.npy and item_factors.npy."
        )

    user_factors = np.load(uf_path)
    item_factors = np.load(itf_path)

    expected = (user_factors.shape[0], item_factors.shape[0])
    swapped = (item_factors.shape[0], user_factors.shape[0])
    if train_csr.shape != expected:
        if train_csr.shape == swapped:
            user_factors, item_factors = item_factors, user_factors
        else:
            raise RuntimeError(
                f"Shape mismatch: CSR={train_csr.shape}, user_factors={user_factors.shape}, item_factors={item_factors.shape}"
            )

    seen_by_user = build_seen_from_csr(train_csr)

    # ---- Popularity baseline (from train CSR nnz per item) ----
    item_pop = np.asarray(train_csr.sum(axis=0)).ravel()
    pop_rank = np.argsort(-item_pop).astype(int).tolist()

    # ---- Evaluate ----
    precisions_als: List[float] = []
    recalls_als: List[float] = []
    precisions_pop: List[float] = []
    recalls_pop: List[float] = []

    for u in user_indices:
        u = int(u)
        rel = relevant_by_user.get(u, set())
        if not rel:
            continue

        seen = seen_by_user.get(u, np.array([], dtype=np.int32))

        rec_als = recommend_topk(
            user_idx=u,
            user_factors=user_factors,
            item_factors=item_factors,
            seen_items=seen,
            k=int(args.k),
            pool=int(args.pool),
        )

        # popularity rec excluding seen
        rec_pop = []
        used = set(seen.tolist()) if seen.size > 0 else set()
        for it in pop_rank:
            if it in used:
                continue
            rec_pop.append(int(it))
            if len(rec_pop) >= int(args.k):
                break

        precisions_als.append(precision_at_k(rec_als, rel, int(args.k)))
        recalls_als.append(recall_at_k(rec_als, rel, int(args.k)))
        precisions_pop.append(precision_at_k(rec_pop, rel, int(args.k)))
        recalls_pop.append(recall_at_k(rec_pop, rel, int(args.k)))

    def avg(x: List[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    print(f"Evaluated users: {len(precisions_als)}")
    print(f"Split: {args.split} | K={args.k} | relevance_threshold>={args.relevance_threshold}")

    print("\nALS")
    print(f"Precision@{args.k}: {avg(precisions_als):.4f}")
    print(f"Recall@{args.k}:    {avg(recalls_als):.4f}")

    print("\nPopularity baseline")
    print(f"Precision@{args.k}: {avg(precisions_pop):.4f}")
    print(f"Recall@{args.k}:    {avg(recalls_pop):.4f}")


if __name__ == "__main__":
    main()
