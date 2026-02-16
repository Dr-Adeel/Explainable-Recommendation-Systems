from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sklearn.neighbors import NearestNeighbors
import joblib
import json
from pathlib import Path as _Path
from typing import Any


DATA_DIR = Path("data/amazon/processed")

ITEMS_PATH = DATA_DIR / "items_with_images.parquet"
EMB_PATH = DATA_DIR / "amazon_image_embeddings.parquet"
CAT_PATH = DATA_DIR / "amazon_clip_catalog.parquet"

app = FastAPI(title="Amazon Fashion API", version="1.1")


_items: Optional[pd.DataFrame] = None

# CLIP artifacts
_cat: Optional[pd.DataFrame] = None
_nn: Optional[NearestNeighbors] = None
_X: Optional[np.ndarray] = None
_item_ids: Optional[List[int]] = None
_idx_by_item: Optional[Dict[int, int]] = None
_item_to_cat: Optional[Dict[int, str]] = None
_item_to_title: Optional[Dict[int, str]] = None
_item_to_path: Optional[Dict[int, str]] = None

# ALS artifacts
_als_user_factors: Optional[np.ndarray] = None
_als_item_factors: Optional[np.ndarray] = None
_als_seen_by_user: Optional[Dict[int, np.ndarray]] = None
_als_model_dir: Optional[Path] = None
_als_pop_scores: Optional[np.ndarray] = None

# Multimodal / fused embeddings + index
_multimodal_X: Optional[np.ndarray] = None
_multimodal_nn: Optional[NearestNeighbors] = None
_multimodal_ids: Optional[List[int]] = None
_multimodal_idx_map: Optional[Dict[int, int]] = None
_faiss_index = None
_multimodal_index_path = Path("data/embeddings/multimodal_embeddings.parquet")
_faiss_index_path = Path("data/embeddings/multimodal.faiss")

# surrogate path
_surrogate_path = Path("data/models/surrogate_rf.joblib")
_surrogate_model: Any = None

# Lazy CLIP model for on-demand embedding computation
_clip_model = None
_clip_processor = None
_clip_device = None


def _require_loaded_items() -> pd.DataFrame:
    if _items is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")
    return _items


def _require_clip_ready() -> None:
    if _nn is None or _X is None or _item_ids is None or _idx_by_item is None:
        raise HTTPException(status_code=503, detail="CLIP artifacts not loaded")


def _load_clip_model_once() -> bool:
    """Lazily load a CLIP model and processor into globals. Returns True on success."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None and _clip_processor is not None:
        return True
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        model.to(device)

        _clip_model = model
        _clip_processor = processor
        _clip_device = device
        return True
    except Exception:
        _clip_model = None
        _clip_processor = None
        _clip_device = None
        return False


def _compute_clip_embedding_for_item(item_idx: int) -> Optional[np.ndarray]:
    """Compute a CLIP image embedding for `item_idx` using the item's image_path.
    Returns a float32 numpy array normalized, or None on failure."""
    global _clip_model, _clip_processor, _clip_device, _items
    if _clip_model is None or _clip_processor is None:
        ok = _load_clip_model_once()
        if not ok:
            return None

    # find item row and image path
    if _items is None:
        return None
    row = _items.loc[_items["item_idx"] == int(item_idx)]
    if row.empty:
        return None
    img_path = str(row.iloc[0].get("image_path", "") or "")
    if not img_path:
        return None

    try:
        from PIL import Image
        import torch

        img = Image.open(img_path).convert("RGB")
        inputs = _clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(_clip_device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = _clip_model.get_image_features(**inputs)
            if hasattr(feats, "detach"):
                feats = feats.detach()
            feats = torch.nn.functional.normalize(feats, dim=1)
            vec = feats.cpu().numpy().reshape(-1).astype(np.float32)
            return vec
    except Exception:
        return None


def ensure_clip_embedding(item_idx: int) -> bool:
    """Ensure there is a CLIP embedding for `item_idx` in memory index. If missing,
    attempt to compute it on-demand from the image and append to the in-memory index.
    Returns True if the embedding is present after this call."""
    global _X, _item_ids, _idx_by_item, _nn, _cat, _item_to_path, _item_to_title, _item_to_cat

    if _idx_by_item is not None and item_idx in _idx_by_item:
        return True

    # try compute
    vec = _compute_clip_embedding_for_item(item_idx)
    if vec is None:
        return False

    try:
        # append to arrays/lists
        if _X is None:
            _X = np.asarray([vec], dtype=np.float32)
            _item_ids = [int(item_idx)]
            _idx_by_item = {int(item_idx): 0}
        else:
            _X = np.vstack([_X, vec.astype(np.float32)])
            if _item_ids is None:
                _item_ids = [int(item_idx)]
            else:
                _item_ids.append(int(item_idx))
            # rebuild index mapping
            _idx_by_item = {iid: i for i, iid in enumerate(_item_ids)}

        # try updating catalog mappings
        if _cat is not None:
            if item_idx in _cat["item_idx"].values:
                r = _cat.loc[_cat["item_idx"] == item_idx].iloc[0]
                _item_to_path = _item_to_path or {}
                _item_to_title = _item_to_title or {}
                _item_to_cat = _item_to_cat or {}
                _item_to_path[item_idx] = str(r.get("image_path", ""))
                _item_to_title[item_idx] = str(r.get("title", ""))
                _item_to_cat[item_idx] = str(r.get("main_category", ""))

        # rebuild nearest-neighbors index
        try:
            _nn = NearestNeighbors(metric="cosine", algorithm="auto")
            _nn.fit(_X)
        except Exception:
            _nn = None

        return True
    except Exception:
        return False


@app.on_event("startup")
def load_artifacts() -> None:
    global _items, _cat, _nn, _X, _item_ids, _idx_by_item
    global _item_to_cat, _item_to_title, _item_to_path

    if not ITEMS_PATH.exists():
        raise RuntimeError(f"Missing: {ITEMS_PATH}")

    items = pd.read_parquet(ITEMS_PATH)
    required = {"item_idx", "item_raw_id"}
    missing = required - set(items.columns)
    if missing:
        raise RuntimeError(f"items_with_images.parquet missing columns: {sorted(missing)}")

    items["item_idx"] = items["item_idx"].astype(int)
    items["title"] = items.get("title", "").fillna("").astype(str)
    items["main_category"] = items.get("main_category", "").fillna("").astype(str)
    items["image_path"] = items.get("image_path", "").fillna("").astype(str)

    _items = items

    # Load CLIP artifacts if present
    if EMB_PATH.exists() and CAT_PATH.exists():
        # debug: report which embedding file we load
        print(f"Loading CLIP embeddings from: {EMB_PATH}")
        emb = pd.read_parquet(EMB_PATH)
        print(f"emb parquet rows: {len(emb)}")
        cat = pd.read_parquet(CAT_PATH)
        print(f"cat parquet rows: {len(cat)}")

        if "item_idx" not in emb.columns or "embedding" not in emb.columns:
            raise RuntimeError("amazon_image_embeddings.parquet must contain item_idx + embedding")

        cat["item_idx"] = cat["item_idx"].astype(int)
        emb["item_idx"] = emb["item_idx"].astype(int)

        df = emb.merge(cat, on="item_idx", how="left")
        df = df.dropna(subset=["embedding"]).copy()

        df["title"] = df.get("title", "").fillna("").astype(str)
        df["main_category"] = df.get("main_category", "").fillna("").astype(str)
        df["image_path"] = df.get("image_path", "").fillna("").astype(str)

        item_ids = df["item_idx"].astype(int).tolist()
        X = np.vstack(df["embedding"].values).astype("float32")

        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(X)

        _cat = df
        _nn = nn
        _X = X
        _item_ids = item_ids
        _idx_by_item = {iid: i for i, iid in enumerate(item_ids)}
        _item_to_cat = dict(zip(df["item_idx"], df["main_category"]))
        _item_to_title = dict(zip(df["item_idx"], df["title"]))
        _item_to_path = dict(zip(df["item_idx"], df["image_path"]))

    # Try loading ALS artifacts from either processed or processed_small
    # Expected layout: <processed_dir>/als/model/{user_factors.npy,item_factors.npy}
    candidates = [
        DATA_DIR / "als" / "model",
        DATA_DIR.parent / "processed_small" / "als" / "model",
    ]

    for cand in candidates:
        try:
            uf = cand / "user_factors.npy"
            itf = cand / "item_factors.npy"
            csr = cand.parent / "train_csr.npz"
            if uf.exists() and itf.exists() and csr.exists():
                user_factors = np.load(uf)
                item_factors = np.load(itf)

                # Load CSR to build seen items per user
                X = sparse.load_npz(csr).tocsr()

                # Ensure shapes align, swap if needed
                expected = (user_factors.shape[0], item_factors.shape[0])
                swapped = (item_factors.shape[0], user_factors.shape[0])
                if X.shape != expected:
                    if X.shape == swapped:
                        user_factors, item_factors = item_factors, user_factors
                    else:
                        # skip this candidate if shapes mismatch
                        continue

                # build seen dict
                indptr = X.indptr
                indices = X.indices
                n_users = X.shape[0]
                seen: Dict[int, np.ndarray] = {}
                for u in range(n_users):
                    start, end = indptr[u], indptr[u + 1]
                    if end > start:
                        seen[u] = indices[start:end]
                    else:
                        seen[u] = np.array([], dtype=np.int32)

                global _als_user_factors, _als_item_factors, _als_seen_by_user, _als_model_dir, _als_pop_scores
                _als_user_factors = user_factors.astype(np.float32)
                _als_item_factors = item_factors.astype(np.float32)
                _als_seen_by_user = seen
                _als_model_dir = cand
                # compute popularity per item (sum over users)
                try:
                    item_pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
                    # store normalized pop scores in [0,1]
                    if item_pop.max() > 0:
                        _als_pop_scores = (item_pop - item_pop.min()) / (item_pop.max() - item_pop.min())
                    else:
                        _als_pop_scores = item_pop
                except Exception:
                    _als_pop_scores = None
                break
        except Exception:
            continue
    # Try to load multimodal embeddings (fused image+text) if available
    try:
        from src.recommenders.multimodal import load_parquet
    except Exception:
        load_parquet = None

    try:
        if _multimodal_index_path.exists() and load_parquet is not None:
            mm = load_parquet(_multimodal_index_path)
            mm_ids = mm['item_idx'].astype(int).tolist()
            mm_X = np.vstack(mm['embedding'].values).astype(np.float32)
            _multimodal_X = mm_X
            _multimodal_ids = mm_ids
            _multimodal_idx_map = {iid: i for i, iid in enumerate(mm_ids)}

            # try loading FAISS index if present
            try:
                import faiss

                if _faiss_index_path.exists():
                    _faiss_index = faiss.read_index(str(_faiss_index_path))
                else:
                    # build faiss index quickly
                    dim = _multimodal_X.shape[1]
                    _faiss_index = faiss.IndexFlatIP(dim)
                    # normalize vectors for inner product ~ cosine
                    norms = np.linalg.norm(_multimodal_X, axis=1, keepdims=True) + 1e-12
                    faiss_vecs = (_multimodal_X / norms).astype(np.float32)
                    _faiss_index.add(faiss_vecs)
                    faiss.write_index(_faiss_index, str(_faiss_index_path))
            except Exception:
                _faiss_index = None
                # fallback to sklearn NN
                try:
                    _multimodal_nn = NearestNeighbors(metric='cosine', algorithm='auto')
                    _multimodal_nn.fit(_multimodal_X)
                except Exception:
                    _multimodal_nn = None
    except Exception:
        # multimodal not available
        pass

    # try loading surrogate if present
    try:
        if _surrogate_path.exists():
            _surrogate_model = joblib.load(_surrogate_path)
    except Exception:
        _surrogate_model = None


@app.get("/health")
def health():
    df = _require_loaded_items()
    clip_ready = _nn is not None and _X is not None
    als_ready = _als_user_factors is not None and _als_item_factors is not None
    return {
        "status": "ok",
        "items_total": int(len(df)),
        "items_with_images": int((df["image_path"].str.len() > 0).sum()),
        "clip_ready": bool(clip_ready),
        "clip_rows": int(len(_item_ids)) if _item_ids is not None else 0,
        "als_ready": bool(als_ready),
        "als_users": int(_als_user_factors.shape[0]) if _als_user_factors is not None else 0,
        "als_items": int(_als_item_factors.shape[0]) if _als_item_factors is not None else 0,
    }


@app.get("/amazon/item/{item_idx}")
def get_item(item_idx: int):
    df = _require_loaded_items()

    row = df.loc[df["item_idx"] == int(item_idx)]
    if row.empty:
        raise HTTPException(status_code=404, detail="item_idx not found")

    r = row.iloc[0]
    return {
        "item_idx": int(r["item_idx"]),
        "item_raw_id": str(r["item_raw_id"]),
        "title": str(r.get("title", "")),
        "main_category": str(r.get("main_category", "")),
        "image_path": str(r.get("image_path", "")),
    }


@app.get("/amazon/image/{item_idx}")
def get_image(item_idx: int):
    df = _require_loaded_items()

    row = df.loc[df["item_idx"] == int(item_idx)]
    if row.empty:
        raise HTTPException(status_code=404, detail="item_idx not found")

    image_path = str(row.iloc[0].get("image_path", "") or "")

    # Prefer explicit image_path from parquet
    path = Path(image_path) if image_path else None

    # If no image_path or the file is missing, try fallback into data/amazon/images
    if path is None or not path.exists():
        images_dir = DATA_DIR.parent / "images"
        fallback_path = None
        if images_dir.exists():
            # downloaded filenames use zero-padded item_idx as prefix: 000123_<rawid>_...jpg
            pattern = f"{int(item_idx):06d}_*"
            matches = list(images_dir.glob(pattern))
            if matches:
                fallback_path = matches[0]

        if fallback_path is not None:
            return FileResponse(fallback_path)

        # no fallback found, raise explicit 404 with reason
        if not image_path:
            raise HTTPException(status_code=404, detail="No image_path for this item_idx")
        raise HTTPException(status_code=404, detail=f"Image file not found on disk: {image_path}")

    return FileResponse(path)


@app.get("/amazon/sample-items")
def sample_items(
    n: int = Query(30, ge=1, le=200),
    with_images: bool = Query(True),
    seed: int = Query(42, ge=0, le=10_000),
):
    df = _require_loaded_items()

    view = df
    if with_images:
        view = view[view["image_path"].str.len() > 0]

    if view.empty:
        raise HTTPException(status_code=404, detail="No items available for sampling")

    if len(view) > n:
        view = view.sample(n=n, random_state=seed)

    out = []
    for r in view.itertuples(index=False):
        out.append(
            {
                "item_idx": int(getattr(r, "item_idx")),
                "item_raw_id": str(getattr(r, "item_raw_id")),
                "title": str(getattr(r, "title", "")),
                "main_category": str(getattr(r, "main_category", "")),
            }
        )
    return {"n": int(len(out)), "items": out}


def _neighbors_pool(q_idx: int, pool: int) -> List[int]:
    assert _nn is not None and _X is not None
    distances, indices = _nn.kneighbors(_X[q_idx].reshape(1, -1), n_neighbors=pool + 1)
    return [int(i) for i in indices[0].tolist() if int(i) != q_idx][:pool]


def _require_als_ready() -> None:
    if _als_user_factors is None or _als_item_factors is None or _als_seen_by_user is None:
        raise HTTPException(status_code=503, detail="ALS artifacts not loaded")


def _recommend_topk(
    user_idx: int,
    k: int,
    pool: int = 2000,
) -> List[int]:
    assert _als_user_factors is not None and _als_item_factors is not None and _als_seen_by_user is not None

    if user_idx < 0 or user_idx >= _als_user_factors.shape[0]:
        raise HTTPException(status_code=404, detail="user_idx out of range for ALS model")

    uvec = _als_user_factors[user_idx].astype(np.float32, copy=False)
    scores = _als_item_factors.astype(np.float32, copy=False) @ uvec

    seen = _als_seen_by_user.get(user_idx, np.array([], dtype=np.int32))
    if seen.size > 0:
        scores[seen] = -1e9

    pool = max(pool, k)
    if pool >= scores.shape[0]:
        top = np.argsort(-scores)
        return top[:k].astype(int).tolist()

    cand = np.argpartition(-scores, pool)[:pool]
    cand = cand[np.argsort(-scores[cand])]
    return cand[:k].astype(int).tolist()


def _normalize_scores(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    a = arr.astype(np.float32, copy=False)
    mn = float(a.min())
    mx = float(a.max())
    if mx - mn <= 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def _compose_reason(parts: Dict[str, float]) -> str:
    # parts are contribution shares (0..1), pick the largest
    if not parts:
        return "No explanation available."
    top = max(parts.items(), key=lambda x: x[1])
    label, val = top
    pct = int(round(val * 100))
    if label == "image":
        return f"Visually similar product (image similarity contributes {pct}% to the score)."
    if label == "user":
        return f"People who bought this also bought that (user-item signal contributes {pct}%)."
    if label == "popularity":
        return f"Popular item (popularity contributes {pct}% to the score)."
    return "Recommended based on combined signals."


def _compose_reason_fr(parts: Dict[str, float]) -> str:
    """Compose a short French explanation from contribution shares.
    `parts` are shares in [0,1] for keys: 'image', 'user', 'popularity'.
    """
    if not parts:
        return "Aucune explication disponible."
    top = max(parts.items(), key=lambda x: x[1])
    label, val = top
    pct = int(round(val * 100))
    if label == "image":
        return f"Produit visuellement similaire (la similarité d'image contribue à hauteur de {pct}% à la recommandation)."
    if label == "user":
        return f"Les utilisateurs qui ont acheté ce produit ont aussi acheté celui-ci (le signal utilisateur contribue à hauteur de {pct}%)."
    if label == "popularity":
        return f"Article populaire (la popularité contribue à hauteur de {pct}% à la recommandation)."
    return "Recommandé en combinant plusieurs signaux."
@app.get("/amazon/similar-items")
def similar_items(
    item_idx: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=50),
    pool: int = Query(200, ge=10, le=5000),
    filter_category: bool = Query(True),
    use_multimodal: bool = Query(False),
):
    # Try to ensure query has CLIP embedding available (compute-on-demand)
    _ = ensure_clip_embedding(item_idx)

    assert _item_to_cat is not None and _item_to_title is not None and _item_to_path is not None

    # If multimodal retrieval requested and available, use it
    if use_multimodal and _multimodal_X is not None:
        pool = max(pool, k)
        pool_item_ids = []
        # prefer faiss if available
        if _faiss_index is not None and item_idx in _multimodal_idx_map:
            try:
                import faiss

                qidx = _multimodal_idx_map[item_idx]
                qvec = _multimodal_X[qidx : qidx + 1].astype(np.float32)
                # normalize
                qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
                D, I = _faiss_index.search(qvec, pool + 1)
                for i in I[0].tolist():
                    if i != qidx:
                        pool_item_ids.append(int(_multimodal_ids[i]))
            except Exception:
                pool_item_ids = []
        elif _multimodal_nn is not None and item_idx in _multimodal_idx_map:
            qidx = _multimodal_idx_map[item_idx]
            distances, indices = _multimodal_nn.kneighbors(_multimodal_X[qidx].reshape(1, -1), n_neighbors=pool + 1)
            pool_item_ids = [int(_multimodal_ids[i]) for i in indices[0].tolist() if int(i) != qidx][:pool]
        else:
            raise HTTPException(status_code=503, detail="Multimodal index not available for retrieval")

    else:
        if _idx_by_item is None or item_idx not in _idx_by_item:
            # try compute-on-demand; if still missing, return 404
            if not ensure_clip_embedding(item_idx):
                raise HTTPException(status_code=404, detail="item_idx not found in CLIP index and no image available")

        q_index = _idx_by_item[item_idx]
        pool = max(pool, k)

        pool_idxs = _neighbors_pool(q_index, pool=pool)
        pool_item_ids = [_item_ids[i] for i in pool_idxs]

    if filter_category:
        q_cat = _item_to_cat.get(item_idx, "")
        if q_cat:
            same = [iid for iid in pool_item_ids if _item_to_cat.get(iid, "") == q_cat]
            other = [iid for iid in pool_item_ids if _item_to_cat.get(iid, "") != q_cat]
            rec_ids = (same + other)[:k]
        else:
            rec_ids = pool_item_ids[:k]
    else:
        rec_ids = pool_item_ids[:k]

    query = {
        "item_idx": int(item_idx),
        "title": _item_to_title.get(item_idx, ""),
        "main_category": _item_to_cat.get(item_idx, ""),
        "image_path": _item_to_path.get(item_idx, ""),
    }

    recs = []
    for iid in rec_ids:
        recs.append(
            {
                "item_idx": int(iid),
                "title": _item_to_title.get(iid, ""),
                "main_category": _item_to_cat.get(iid, ""),
            }
        )

    return {
        "query": query,
        "k": int(k),
        "pool": int(pool),
        "filter_category": bool(filter_category),
        "recommendations": recs,
    }


@app.get("/amazon/recommend-user")
def recommend_user(
    user_idx: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=100),
    pool: int = Query(2000, ge=10, le=10000),
):
    _require_als_ready()

    rec_item_idxs = _recommend_topk(user_idx=user_idx, k=k, pool=pool)

    # Map item indices to item metadata if available
    df = _require_loaded_items()
    out = []
    for it in rec_item_idxs:
        row = df.loc[df["item_idx"] == int(it)]
        if row.empty:
            out.append({"item_idx": int(it)})
            continue
        r = row.iloc[0]
        out.append(
            {
                "item_idx": int(r["item_idx"]),
                "item_raw_id": str(r.get("item_raw_id", "")),
                "title": str(r.get("title", "")),
                "main_category": str(r.get("main_category", "")),
                "image_path": str(r.get("image_path", "")),
            }
        )

    return {"user_idx": int(user_idx), "k": int(k), "recommendations": out}


@app.get("/amazon/recommend-hybrid")
def recommend_hybrid(
    item_idx: int = Query(..., ge=0),
    k: int = Query(10, ge=1, le=100),
    pool: int = Query(500, ge=10, le=5000),
    alpha: float = Query(0.5, ge=0.0, le=10.0),
    beta: float = Query(0.4, ge=0.0, le=10.0),
    gamma: float = Query(0.1, ge=0.0, le=10.0),
    filter_category: bool = Query(True),
):
    # Hybrid: combine image (CLIP), ALS (item factors), and popularity
    # Ensure ALS ready; CLIP embeddings may be computed on-demand
    _require_als_ready()

    # Determine candidate pool from CLIP if available; compute missing embeddings on-demand
    cand_item_ids = []
    if ensure_clip_embedding(item_idx) and _idx_by_item is not None and item_idx in _idx_by_item:
        q_index = _idx_by_item[item_idx]
        pool_idxs = _neighbors_pool(q_index, pool=pool)
        cand_item_ids = [_item_ids[i] for i in pool_idxs]
        # include the query itself if missing
        if item_idx not in cand_item_ids:
            cand_item_ids = [item_idx] + cand_item_ids
    else:
        # fallback: use top-popular items
        if _als_pop_scores is not None:
            cand_item_ids = list(np.argsort(-_als_pop_scores)[:pool].astype(int).tolist())
        else:
            raise HTTPException(status_code=404, detail="Cannot build candidate pool for this item")

    # Optional category filtering
    if filter_category:
        q_cat = _item_to_cat.get(item_idx, "")
        if q_cat:
            same = [iid for iid in cand_item_ids if _item_to_cat.get(iid, "") == q_cat]
            other = [iid for iid in cand_item_ids if _item_to_cat.get(iid, "") != q_cat]
            cand_item_ids = (same + other)[: max(pool, k)]

    cand = np.array(cand_item_ids, dtype=int)

    # Image scores: cosine similarity between query embedding and candidate embeddings
    img_scores = np.zeros(cand.shape[0], dtype=np.float32)
    # compute image similarities, attempting to compute embeddings for missing candidates
    if ensure_clip_embedding(item_idx) and _idx_by_item is not None and item_idx in _idx_by_item:
        qidx = _idx_by_item[item_idx]
        qvec = _X[qidx].astype(np.float32)
        # ensure embeddings for candidates
        for i, iid in enumerate(cand.tolist()):
            if iid not in _idx_by_item:
                ensure_clip_embedding(int(iid))

        Xcand_idx = [ _idx_by_item[iid] for iid in cand if iid in _idx_by_item ]
        if Xcand_idx:
            Xcand = _X[Xcand_idx].astype(np.float32)
            qnorm = np.linalg.norm(qvec) + 1e-12
            norms = np.linalg.norm(Xcand, axis=1) + 1e-12
            sims = (Xcand @ qvec) / (norms * qnorm)
            # place sims into img_scores aligned with cand indices
            mask = [iid in _idx_by_item for iid in cand]
            img_scores[np.array(mask)] = sims

    # ALS/item-factor scores: dot product between query item factor and candidate item factors
    als_scores = np.zeros(cand.shape[0], dtype=np.float32)
    try:
        q_if = _als_item_factors[item_idx]
        itf = _als_item_factors[cand]
        als_raw = itf.astype(np.float32) @ q_if.astype(np.float32)
        als_scores = als_raw
    except Exception:
        # item not in ALS (out of range), leave zeros
        pass

    # popularity scores from ALS pop normalization if available
    pop_scores = np.zeros(cand.shape[0], dtype=np.float32)
    if _als_pop_scores is not None:
        pop_scores = _als_pop_scores[cand]

    # normalize each signal to [0,1]
    img_n = _normalize_scores(img_scores)
    als_n = _normalize_scores(als_scores)
    pop_n = _normalize_scores(pop_scores)

    # combine
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    combined_raw = alpha * img_n + beta * als_n + gamma * pop_n

    # pick top-k (exclude query item from results)
    if combined_raw.size == 0:
        return {"query_item": int(item_idx), "k": int(k), "recommendations": []}

    sorted_idx = np.argsort(-combined_raw)

    recs = []
    for ti in sorted_idx:
        iid = int(cand[ti])
        if iid == item_idx:
            continue  # skip self-recommendation
        if len(recs) >= k:
            break
        iid = int(cand[ti])
        parts_raw = {
            "image": float(alpha * img_n[ti]) if img_n.size>0 else 0.0,
            "user": float(beta * als_n[ti]) if als_n.size>0 else 0.0,
            "popularity": float(gamma * pop_n[ti]) if pop_n.size>0 else 0.0,
        }
        total = sum(parts_raw.values())
        parts_share = {k: (v / total if total > 0 else 0.0) for k, v in parts_raw.items()}
        reason = _compose_reason(parts_share)

        row = _items.loc[_items["item_idx"] == iid]
        if row.empty:
            meta = {"item_idx": iid}
        else:
            r = row.iloc[0]
            meta = {
                "item_idx": int(r["item_idx"]),
                "item_raw_id": str(r.get("item_raw_id", "")),
                "title": str(r.get("title", "")),
                "main_category": str(r.get("main_category", "")),
                "image_path": str(r.get("image_path", "")),
            }

        recs.append({"score": float(combined_raw[ti]), "parts": parts_share, "reason": reason, **meta})

    return {"query_item": int(item_idx), "k": int(k), "alpha": alpha, "beta": beta, "gamma": gamma, "recommendations": recs}


@app.get("/amazon/explain-recommendation")
def explain_recommendation(
    item_idx: int = Query(..., ge=0),
    user_idx: int | None = Query(None, ge=0),
    k: int = Query(10, ge=1, le=100),
    pool: int = Query(500, ge=10, le=5000),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    beta: float = Query(0.4, ge=0.0, le=1.0),
    gamma: float = Query(0.1, ge=0.0, le=1.0),
):
    """Return recommendations plus surrogate-based explanations (feature contributions).

    Features used by surrogate: [multimodal_cosine, als_dot, popularity]
    If SHAP is installed and surrogate is a tree model, SHAP values are returned; otherwise
    contributions are approximated by feature * model.feature_importances_.
    """
    # Ensure ALS ready; allow CLIP embeddings to be computed on-demand so visual
    # similarity is available for as many items as possible.
    _require_als_ready()

    # Try to compute the query embedding if missing so hybrid will include visual signal
    resp = recommend_hybrid(item_idx=item_idx, k=max(k, 50), pool=pool, alpha=alpha, beta=beta, gamma=gamma, filter_category=True)
    candidates = [int(r["item_idx"]) if isinstance(r, dict) and "item_idx" in r else int(r) for r in resp.get("recommendations", [])]

    # if surrogate not loaded, try to load
    global _surrogate_model
    if _surrogate_model is None and _surrogate_path.exists():
        try:
            _surrogate_model = joblib.load(_surrogate_path)
        except Exception:
            _surrogate_model = None

    explanations = []

    # helper to compute multimodal cosine
    def multimodal_cosine(a_idx: int, b_idx: int) -> float:
        # prefer multimodal fused embeddings when available
        if _multimodal_X is not None and a_idx in _multimodal_idx_map and b_idx in _multimodal_idx_map:
            va = _multimodal_X[_multimodal_idx_map[a_idx]]
            vb = _multimodal_X[_multimodal_idx_map[b_idx]]
            num = float(np.dot(va, vb))
            denom = float((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12))
            return float(num / denom)
        # fallback to image embeddings similarity if available
        if _X is not None and a_idx in _idx_by_item and b_idx in _idx_by_item:
            va = _X[_idx_by_item[a_idx]]
            vb = _X[_idx_by_item[b_idx]]
            num = float(np.dot(va, vb))
            denom = float((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12))
            return float(num / denom)
        # Attempt proxy: if a_idx not in CLIP but has same raw id as an item that is
        # in the CLIP index, use that item's embedding as a proxy.
        try:
            if _X is not None and _idx_by_item is not None and _items is not None:
                row = _items.loc[_items['item_idx'] == int(a_idx)]
                if not row.empty:
                    raw = row.iloc[0].get('item_raw_id')
                    if raw:
                        # find candidate item with same raw id that exists in clip index
                        matches = _items.loc[_items['item_raw_id'] == raw]
                        for m in matches.itertuples(index=False):
                            midx = int(getattr(m, 'item_idx'))
                            if midx in _idx_by_item and b_idx in _idx_by_item:
                                va = _X[_idx_by_item[midx]]
                                vb = _X[_idx_by_item[b_idx]]
                                num = float(np.dot(va, vb))
                                denom = float((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12))
                                return float(num / denom)
        except Exception:
            pass

        return 0.0

    # helper for ALS dot
    def als_dot(a_idx: int, b_idx: int) -> float:
        try:
            ai = _als_item_factors[a_idx]
            bi = _als_item_factors[b_idx]
            return float(np.dot(ai.astype(np.float32), bi.astype(np.float32)))
        except Exception:
            return 0.0

    for cand in candidates[:k]:
        feats = []
        mm_sim = multimodal_cosine(item_idx, cand)
        als_sim = als_dot(item_idx, cand)
        pop = float(_als_pop_scores[cand]) if (_als_pop_scores is not None and cand < len(_als_pop_scores)) else 0.0

        feats = [mm_sim, als_sim, pop]

        contribs = None
        pred = None
        # if surrogate and shap available, compute shap values
        if _surrogate_model is not None:
            try:
                import shap

                model = _surrogate_model.get('model') if isinstance(_surrogate_model, dict) and 'model' in _surrogate_model else _surrogate_model
                expl = shap.Explainer(model)
                vals = expl(np.array([feats]))
                shap_vals = vals.values[0].tolist() if hasattr(vals, 'values') else vals[0].tolist()
                contribs = {'features': ['multimodal_cosine', 'als_dot', 'popularity'], 'shap_values': shap_vals}
                pred = float(model.predict(np.array([feats]))[0])
            except Exception:
                # fallback: use feature_importances_ if available
                try:
                    model = _surrogate_model.get('model') if isinstance(_surrogate_model, dict) and 'model' in _surrogate_model else _surrogate_model
                    importances = getattr(model, 'feature_importances_', None)
                    if importances is not None:
                        raw = (np.array(feats) * np.array(importances)).tolist()
                        contribs = {'features': ['multimodal_cosine', 'als_dot', 'popularity'], 'contributions': raw, 'importances': importances.tolist()}
                        pred = float(model.predict(np.array([feats]))[0])
                except Exception:
                    contribs = {'features': ['multimodal_cosine', 'als_dot', 'popularity'], 'contributions': feats}
        else:
            # no surrogate: return raw features as proxy
            contribs = {'features': ['multimodal_cosine', 'als_dot', 'popularity'], 'contributions': feats}

        # diagnostic flags to help debug missing embeddings/mappings
        debug = {
            'query_in_multimodal': bool(_multimodal_idx_map is not None and item_idx in _multimodal_idx_map),
            'cand_in_multimodal': bool(_multimodal_idx_map is not None and cand in _multimodal_idx_map),
            'query_in_clip': bool(_idx_by_item is not None and item_idx in _idx_by_item),
            'cand_in_clip': bool(_idx_by_item is not None and cand in _idx_by_item),
            'multimodal_idx': int(_multimodal_idx_map[item_idx]) if (_multimodal_idx_map is not None and item_idx in _multimodal_idx_map) else None,
            'cand_multimodal_idx': int(_multimodal_idx_map[cand]) if (_multimodal_idx_map is not None and cand in _multimodal_idx_map) else None,
            'clip_idx': int(_idx_by_item[item_idx]) if (_idx_by_item is not None and item_idx in _idx_by_item) else None,
            'cand_clip_idx': int(_idx_by_item[cand]) if (_idx_by_item is not None and cand in _idx_by_item) else None,
            # embedding diagnostics
            'clip_query_norm': float(np.linalg.norm(_X[_idx_by_item[item_idx]])) if (_X is not None and _idx_by_item is not None and item_idx in _idx_by_item) else None,
            'clip_cand_norm': float(np.linalg.norm(_X[_idx_by_item[cand]])) if (_X is not None and _idx_by_item is not None and cand in _idx_by_item) else None,
            'clip_query_sum': float(np.sum(_X[_idx_by_item[item_idx]])) if (_X is not None and _idx_by_item is not None and item_idx in _idx_by_item) else None,
            'clip_cand_sum': float(np.sum(_X[_idx_by_item[cand]])) if (_X is not None and _idx_by_item is not None and cand in _idx_by_item) else None,
            'multimodal_query_norm': float(np.linalg.norm(_multimodal_X[_multimodal_idx_map[item_idx]])) if (_multimodal_X is not None and _multimodal_idx_map is not None and item_idx in _multimodal_idx_map) else None,
            'multimodal_cand_norm': float(np.linalg.norm(_multimodal_X[_multimodal_idx_map[cand]])) if (_multimodal_X is not None and _multimodal_idx_map is not None and cand in _multimodal_idx_map) else None,
        }

        # build contribution shares (scale each feature by requested mix weights)
        parts_raw = {
            'image': float(alpha * mm_sim) if mm_sim is not None else 0.0,
            'user': float(beta * als_sim) if als_sim is not None else 0.0,
            'popularity': float(gamma * pop) if pop is not None else 0.0,
        }
        total = sum(parts_raw.values())
        parts_share = {k: (v / total if total > 0 else 0.0) for k, v in parts_raw.items()}
        reason_fr = _compose_reason_fr(parts_share)

        explanations.append({
            'item_idx': int(cand),
            'features': {'multimodal_cosine': mm_sim, 'als_dot': als_sim, 'popularity': pop},
            'surrogate_pred': pred,
            'contribs': contribs,
            'debug': debug,
            'reason_fr': reason_fr,
        })

    return {'query_item': int(item_idx), 'k': int(k), 'explanations': explanations}


class FeedbackModel(BaseModel):
    query_item: int
    candidate_item: int
    helpful: bool
    lang: Optional[str] = "en"


@app.post("/amazon/feedback")
def feedback(feedback: FeedbackModel):
    """Receive simple user feedback about a candidate recommendation and append to disk as JSONL."""
    out_dir = Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feedback.jsonl"

    rec = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "query_item": int(feedback.query_item),
        "candidate_item": int(feedback.candidate_item),
        "helpful": bool(feedback.helpful),
        "lang": feedback.lang or "",
    }

    try:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write feedback: {e}")

    return {"status": "ok", "written": True}
