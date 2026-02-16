from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sklearn.neighbors import NearestNeighbors

from src.recommenders.fashion_user import recommend_user_item_item, build_popularity_rank


DATA_DIR = Path("data/image_hf/processed")
CATALOG_PATH = DATA_DIR / "catalog.parquet"
EMB_PATH = DATA_DIR / "image_embeddings.parquet"

FASHION_TRAIN_PATH = DATA_DIR / "interactions_train.parquet"
FASHION_NEIGH_PATH = DATA_DIR / "fashion_item_neighbors.parquet"

IMAGES_DIR_DEFAULT = Path("data/image_hf/images")

app = FastAPI(title="Recommender API", version="1.0")


# ===== In-memory artifacts (image) =====
_df: Optional[pd.DataFrame] = None
_nn: Optional[NearestNeighbors] = None
_X: Optional[np.ndarray] = None
_product_ids: Optional[List[int]] = None
_idx_by_pid: Optional[Dict[int, int]] = None
_pid_to_type: Optional[Dict[int, str]] = None
_pid_to_name: Optional[Dict[int, str]] = None
_pid_to_image_path: Optional[Dict[int, str]] = None


# ===== In-memory artifacts (fashion user-based) =====
_fashion_train_df: Optional[pd.DataFrame] = None
_fashion_pop_rank: Optional[List[int]] = None
_fashion_item_neighbors: Optional[Dict[int, List[Tuple[int, float]]]] = None


def _neighbors_pool(nn: NearestNeighbors, X: np.ndarray, q_idx: int, pool: int) -> List[int]:
    distances, indices = nn.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=pool + 1)
    return [i for i in indices[0].tolist() if i != q_idx][:pool]


def _recommend_image_filtered(query_pid: int, k: int, pool: int, filter_type: bool) -> List[int]:
    assert _nn is not None and _X is not None and _product_ids is not None and _idx_by_pid is not None
    assert _pid_to_type is not None

    if query_pid not in _idx_by_pid:
        raise ValueError("product_id not found")

    q_idx = _idx_by_pid[query_pid]
    pool = max(pool, k)

    pool_idxs = _neighbors_pool(_nn, _X, q_idx, pool=pool)
    pool_pids = [_product_ids[i] for i in pool_idxs]

    if not filter_type:
        return pool_pids[:k]

    q_type = _pid_to_type.get(query_pid)
    if not q_type:
        return pool_pids[:k]

    same = [pid for pid in pool_pids if _pid_to_type.get(pid) == q_type]
    other = [pid for pid in pool_pids if _pid_to_type.get(pid) != q_type]

    rec = same[:k]
    if len(rec) < k:
        rec.extend(other[: (k - len(rec))])
    return rec[:k]


def _load_item_neighbors(path: Path) -> Dict[int, List[Tuple[int, float]]]:
    df = pd.read_parquet(path, columns=["product_id", "neighbor_id", "score"])
    df["product_id"] = df["product_id"].astype(int)
    df["neighbor_id"] = df["neighbor_id"].astype(int)
    df["score"] = df["score"].astype(float)

    m: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for pid, nid, s in df.itertuples(index=False):
        m[int(pid)].append((int(nid), float(s)))
    return m


@app.on_event("startup")
def load_artifacts() -> None:
    global _df, _nn, _X, _product_ids, _idx_by_pid
    global _pid_to_type, _pid_to_name, _pid_to_image_path
    global _fashion_train_df, _fashion_pop_rank, _fashion_item_neighbors

    if not CATALOG_PATH.exists():
        raise RuntimeError(f"Missing catalog: {CATALOG_PATH}")
    if not EMB_PATH.exists():
        raise RuntimeError(f"Missing embeddings: {EMB_PATH}")

    catalog = pd.read_parquet(CATALOG_PATH)
    emb = pd.read_parquet(EMB_PATH)

    catalog["product_id"] = catalog["product_id"].astype(int)
    emb["product_id"] = emb["product_id"].astype(int)

    if "articleType" not in catalog.columns or "productDisplayName" not in catalog.columns:
        raise RuntimeError("catalog.parquet must contain 'articleType' and 'productDisplayName'")

    cols = ["product_id", "articleType", "productDisplayName"]
    if "image_path" in catalog.columns:
        cols.append("image_path")

    df = emb.merge(catalog[cols], on="product_id", how="left")
    df = df.dropna(subset=["embedding", "articleType"]).copy()

    df["articleType"] = df["articleType"].astype(str)
    df["productDisplayName"] = df["productDisplayName"].astype(str)

    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype(str)
    else:
        df["image_path"] = ""

    product_ids = df["product_id"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype("float32")

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X)

    _df = df
    _nn = nn
    _X = X
    _product_ids = product_ids
    _idx_by_pid = {pid: i for i, pid in enumerate(product_ids)}
    _pid_to_type = dict(zip(df["product_id"], df["articleType"]))
    _pid_to_name = dict(zip(df["product_id"], df["productDisplayName"]))
    _pid_to_image_path = dict(zip(df["product_id"], df["image_path"]))

    # Fashion user-based artifacts (optional but expected in H2)
    if FASHION_TRAIN_PATH.exists():
        train_df = pd.read_parquet(FASHION_TRAIN_PATH, columns=["user_id", "product_id", "event"])
        train_df["user_id"] = train_df["user_id"].astype(int)
        train_df["product_id"] = train_df["product_id"].astype(int)
        train_df["event"] = train_df["event"].astype(str)

        _fashion_train_df = train_df
        _fashion_pop_rank = build_popularity_rank(train_df)

    if FASHION_NEIGH_PATH.exists():
        _fashion_item_neighbors = _load_item_neighbors(FASHION_NEIGH_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok" if _df is not None else "not_ready",
        "image_rows": int(len(_df)) if _df is not None else 0,
        "fashion_users_loaded": _fashion_train_df is not None,
        "fashion_neighbors_loaded": _fashion_item_neighbors is not None,
    }


@app.get("/image/{product_id}")
def get_image(product_id: int):
    if _pid_to_image_path is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    p = _pid_to_image_path.get(int(product_id), "")
    if not p:
        raise HTTPException(status_code=404, detail="No image_path for this product_id")

    img_path = Path(p)
    if not img_path.is_absolute():
        # allow relative paths stored in parquet
        img_path = Path(p)

    if not img_path.exists():
        # fallback to default images directory, common pattern: <dir>/<product_id>.jpg
        fallback = IMAGES_DIR_DEFAULT / f"{int(product_id)}.jpg"
        if fallback.exists():
            img_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Image file not found: {img_path}")

    return FileResponse(img_path)


@app.get("/similar-products")
def similar_products(
    product_id: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=50),
    pool: int = Query(200, ge=10, le=5000),
    filter_type: bool = Query(True),
):
    if _df is None or _nn is None or _X is None or _product_ids is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    try:
        rec_ids = _recommend_image_filtered(int(product_id), k=int(k), pool=int(pool), filter_type=bool(filter_type))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    query = {
        "product_id": int(product_id),
        "articleType": _pid_to_type.get(int(product_id)) if _pid_to_type else None,
        "name": _pid_to_name.get(int(product_id)) if _pid_to_name else None,
        "image_path": _pid_to_image_path.get(int(product_id)) if _pid_to_image_path else None,
    }

    recs = [
        {
            "product_id": int(pid),
            "articleType": _pid_to_type.get(int(pid)) if _pid_to_type else None,
            "name": _pid_to_name.get(int(pid)) if _pid_to_name else None,
            "image_path": _pid_to_image_path.get(int(pid)) if _pid_to_image_path else None,
        }
        for pid in rec_ids
    ]

    return {"query": query, "k": int(k), "pool": int(pool), "filter_type": bool(filter_type), "recommendations": recs}


@app.get("/recommend-user-fashion")
def recommend_user_fashion(
    user_id: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=50),
):
    if _fashion_train_df is None or _fashion_pop_rank is None or _fashion_item_neighbors is None:
        raise HTTPException(
            status_code=503,
            detail="Fashion user artifacts not loaded. Missing interactions_train.parquet or fashion_item_neighbors.parquet",
        )

    rec_ids = recommend_user_item_item(
        user_id=int(user_id),
        train_df=_fashion_train_df,
        item_neighbors=_fashion_item_neighbors,
        pop_rank=_fashion_pop_rank,
        k=int(k),
    )

    return {
        "user_id": int(user_id),
        "k": int(k),
        "recommendations": [
            {
                "product_id": int(pid),
                "articleType": _pid_to_type.get(int(pid)) if _pid_to_type else None,
                "name": _pid_to_name.get(int(pid)) if _pid_to_name else None,
                "image_path": _pid_to_image_path.get(int(pid)) if _pid_to_image_path else None,
            }
            for pid in rec_ids
        ],
    }


@app.get("/recommend-hybrid")
def recommend_hybrid(
    user_id: int = Query(..., ge=0),
    product_id: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=50),
    pool: int = Query(200, ge=10, le=5000),
    filter_type: bool = Query(True),
    alpha: float = Query(0.6, ge=0.0, le=1.0),
    beta: float = Query(0.35, ge=0.0, le=1.0),
    gamma: float = Query(0.05, ge=0.0, le=1.0),
):
    if _df is None or _nn is None or _X is None or _product_ids is None:
        raise HTTPException(status_code=503, detail="Image artifacts not loaded")

    if _fashion_train_df is None or _fashion_pop_rank is None or _fashion_item_neighbors is None:
        raise HTTPException(status_code=503, detail="Fashion user artifacts not loaded")

    pid = int(product_id)
    uid = int(user_id)
    k = int(k)
    pool = int(pool)

    # 1) image candidates
    try:
        img_cands = _recommend_image_filtered(pid, k=max(k, 50), pool=pool, filter_type=bool(filter_type))
    except ValueError:
        raise HTTPException(status_code=404, detail="product_id not found in image index")

    # 2) user candidates
    user_cands = recommend_user_item_item(
        user_id=uid,
        train_df=_fashion_train_df,
        item_neighbors=_fashion_item_neighbors,
        pop_rank=_fashion_pop_rank,
        k=max(k, 50),
    )

    # 3) popularity candidates
    pop_cands = _fashion_pop_rank[: max(k, 200)]

    # Candidate pool
    cand_set = []
    seen = set()
    for lst in (img_cands, user_cands, pop_cands):
        for x in lst:
            if x not in seen:
                cand_set.append(int(x))
                seen.add(int(x))

    # Build fast ranks for scoring
    img_rank = {p: i for i, p in enumerate(img_cands)}
    user_rank = {p: i for i, p in enumerate(user_cands)}
    pop_rank = {p: i for i, p in enumerate(pop_cands)}

    def rank_score(rank_map: Dict[int, int], p: int, scale: int) -> float:
        # higher is better; if missing, 0
        r = rank_map.get(p)
        if r is None:
            return 0.0
        return 1.0 - (float(r) / float(max(scale - 1, 1)))

    scored = []
    for p in cand_set:
        s_img = rank_score(img_rank, p, scale=len(img_cands))
        s_user = rank_score(user_rank, p, scale=len(user_cands))
        s_pop = rank_score(pop_rank, p, scale=len(pop_cands))

        s = alpha * s_img + beta * s_user + gamma * s_pop

        parts = {"image": float(alpha * s_img), "user": float(beta * s_user), "popularity": float(gamma * s_pop)}
        reason_bits = []
        if parts["image"] >= max(parts["user"], parts["popularity"]) and parts["image"] > 0:
            reason_bits.append("visually similar")
        if parts["user"] >= max(parts["image"], parts["popularity"]) and parts["user"] > 0:
            reason_bits.append("matches user behavior")
        if parts["popularity"] > 0 and len(reason_bits) == 0:
            reason_bits.append("popular fallback")

        scored.append((p, float(s), parts, ", ".join(reason_bits)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    query = {
        "user_id": uid,
        "product_id": pid,
        "articleType": _pid_to_type.get(pid) if _pid_to_type else None,
        "product_name": _pid_to_name.get(pid) if _pid_to_name else None,
    }

    recs = [
        {
            "product_id": int(p),
            "articleType": _pid_to_type.get(int(p)) if _pid_to_type else None,
            "name": _pid_to_name.get(int(p)) if _pid_to_name else None,
            "image_path": _pid_to_image_path.get(int(p)) if _pid_to_image_path else None,
            "score": float(s),
            "parts": parts,
            "reason": reason,
        }
        for (p, s, parts, reason) in top
    ]

    return {
        "query": query,
        "k": k,
        "pool": pool,
        "filter_type": bool(filter_type),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "recommendations": recs,
    }
