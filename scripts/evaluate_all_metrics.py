"""
Comprehensive evaluation of ALL recommendation engines.

Metrics:
  - User-interaction based (ALS, Popularity): Precision, Recall, F1, NDCG, MRR, HitRate, MAP
  - Category-coherence based (Multimodal KNN, Hybrid): Cat-Precision, Cat-HitRate
  - Surrogate model (SHAP explainability): R², RMSE, MAE, Feature importances

Engines:
  1. ALS (collaborative filtering)          → user-interaction metrics
  2. Multimodal KNN (embeddings cosine)     → category-coherence metrics
  3. Hybrid (α·image + β·ALS + γ·pop)      → both evaluation types
  4. Popularity baseline                    → user-interaction metrics
  5. Surrogate RF (SHAP explainability)     → regression metrics
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# ─── paths ───
PROCESSED = Path("data/amazon/processed_small")
ALS_DIR = PROCESSED / "als"
MODEL_DIR = ALS_DIR / "model"
EMB_PATH = Path("data/embeddings/multimodal_embeddings.parquet")
ITEMS_PATH = PROCESSED / "items.parquet"
REPORT_DIR = Path("reports")


# ═══════════════════════════  IR METRICS  ════════════════════════════════

def precision_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    topk = rec[:k]
    return sum(1 for x in topk if x in rel) / len(topk) if topk else 0.0

def recall_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    if not rel: return 0.0
    topk = rec[:k]
    return sum(1 for x in topk if x in rel) / len(rel)

def f1_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    p, r = precision_at_k(rec, rel, k), recall_at_k(rec, rel, k)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

def ndcg_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    topk = rec[:k]
    dcg = sum((1.0 / math.log2(i + 2)) for i, x in enumerate(topk) if x in rel)
    n_rel = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel)) if n_rel else 0.0
    return dcg / idcg if idcg > 0 else 0.0

def reciprocal_rank(rec: List[int], rel: Set[int]) -> float:
    for i, x in enumerate(rec):
        if x in rel: return 1.0 / (i + 1)
    return 0.0

def hit_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    return 1.0 if any(x in rel for x in rec[:k]) else 0.0

def ap_at_k(rec: List[int], rel: Set[int], k: int) -> float:
    topk = rec[:k]
    hits, total = 0, 0.0
    for i, x in enumerate(topk):
        if x in rel:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(rel), k) if rel else 0.0

def compute_all_ir_metrics(rec: List[int], rel: Set[int], k: int) -> Dict[str, float]:
    return {
        f"Precision@{k}":  precision_at_k(rec, rel, k),
        f"Recall@{k}":     recall_at_k(rec, rel, k),
        f"F1@{k}":         f1_at_k(rec, rel, k),
        f"NDCG@{k}":       ndcg_at_k(rec, rel, k),
        "MRR":             reciprocal_rank(rec, rel),
        f"HitRate@{k}":    hit_at_k(rec, rel, k),
        f"MAP@{k}":        ap_at_k(rec, rel, k),
    }


# ═══════════════════  CATEGORY COHERENCE METRICS  ═══════════════════════

def cat_precision_at_k(query_cat: str, rec_cats: List[str], k: int) -> float:
    topk = rec_cats[:k]
    return sum(1 for c in topk if c == query_cat) / len(topk) if topk else 0.0

def cat_hit_at_k(query_cat: str, rec_cats: List[str], k: int) -> float:
    return 1.0 if any(c == query_cat for c in rec_cats[:k]) else 0.0


# ═══════════════════════  DATA LOADING  ═════════════════════════════════

def load_interactions(split: str, threshold: float = 4.0) -> pd.DataFrame:
    path = PROCESSED / f"interactions_{split}.parquet"
    df = pd.read_parquet(path)
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df[df["rating"].astype(float) >= threshold].copy()

def load_als() -> Tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    uf = np.load(MODEL_DIR / "user_factors.npy")
    itf = np.load(MODEL_DIR / "item_factors.npy")
    csr = sparse.load_npz(ALS_DIR / "train_csr.npz").tocsr()
    if csr.shape != (uf.shape[0], itf.shape[0]):
        if csr.shape == (itf.shape[0], uf.shape[0]):
            uf, itf = itf, uf
    return uf, itf, csr

def load_multimodal_embeddings() -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    df = pd.read_parquet(EMB_PATH)
    ids = df["item_idx"].astype(int).tolist()
    X = np.vstack(df["embedding"].values).astype(np.float32)
    idx_map = {iid: pos for pos, iid in enumerate(ids)}
    return X, idx_map, ids

def load_items() -> pd.DataFrame:
    return pd.read_parquet(ITEMS_PATH)


# ═══════════════════  RECOMMENDATION ENGINES  ═══════════════════════════

def als_recommend(user_idx: int, uf: np.ndarray, itf: np.ndarray,
                  seen: np.ndarray, k: int) -> List[int]:
    scores = itf.astype(np.float32) @ uf[user_idx].astype(np.float32)
    if seen.size > 0: scores[seen] = -1e9
    topk = np.argpartition(-scores, min(k, len(scores)-1))[:k]
    topk = topk[np.argsort(-scores[topk])]
    return topk.astype(int).tolist()

def popularity_recommend(pop_rank: List[int], seen: Set[int], k: int) -> List[int]:
    return [it for it in pop_rank if it not in seen][:k]

def multimodal_knn(item_idx: int, X: np.ndarray, idx_map: Dict[int, int],
                   reverse_map: Dict[int, int], k: int) -> List[int]:
    if item_idx not in idx_map: return []
    q = X[idx_map[item_idx]]
    norms = np.linalg.norm(X, axis=1) + 1e-12
    sims = (X @ q) / (norms * (np.linalg.norm(q) + 1e-12))
    sims[idx_map[item_idx]] = -1
    topk = np.argpartition(-sims, k)[:k]
    topk = topk[np.argsort(-sims[topk])]
    return [reverse_map[i] for i in topk if i in reverse_map]

def hybrid_recommend(item_idx: int, user_idx: int | None,
                     X: np.ndarray, idx_map: Dict[int, int],
                     uf: np.ndarray, itf: np.ndarray,
                     pop_scores: np.ndarray, seen: Set[int],
                     k: int, alpha=0.5, beta=0.4, gamma=0.1) -> List[int]:
    n_items = itf.shape[0]
    # image cosine
    img = np.zeros(n_items, dtype=np.float32)
    if item_idx in idx_map:
        q = X[idx_map[item_idx]]
        qn = np.linalg.norm(q) + 1e-12
        norms = np.linalg.norm(X, axis=1) + 1e-12
        sims = (X @ q) / (norms * qn)
        for iid, pos in idx_map.items():
            if iid < n_items: img[iid] = sims[pos]
    # ALS
    als = np.zeros(n_items, dtype=np.float32)
    if user_idx is not None and user_idx < uf.shape[0]:
        als = itf.astype(np.float32) @ uf[user_idx].astype(np.float32)
    # pop
    pop = np.zeros(n_items, dtype=np.float32)
    pop[:len(pop_scores)] = pop_scores[:n_items]
    # min-max + combine
    def mm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)
    combined = alpha * mm(img) + beta * mm(als) + gamma * mm(pop)
    for s in seen:
        if s < n_items: combined[s] = -1e9
    if item_idx < n_items: combined[item_idx] = -1e9
    topk = np.argpartition(-combined, min(k, n_items-1))[:k]
    topk = topk[np.argsort(-combined[topk])]
    return topk.astype(int).tolist()


# ═════════════════  PART 1: USER-INTERACTION EVAL  ══════════════════════

def eval_user_interaction(split, ks, threshold, max_users, alpha, beta, gamma):
    """Evaluate ALS, Hybrid (user-side), Popularity on user-item interactions."""
    print("\n── Part 1 : Évaluation interaction-utilisateur ──")
    eval_df = load_interactions(split, threshold)
    uf, itf, csr = load_als()
    X, idx_map, mm_ids = load_multimodal_embeddings()
    reverse_map = {v: k for k, v in idx_map.items()}
    n_items = itf.shape[0]

    pop_scores = np.asarray(csr.sum(axis=0)).ravel().astype(np.float32)
    pop_rank = np.argsort(-pop_scores).tolist()

    indptr, indices = csr.indptr, csr.indices
    seen_by_user = {}
    for u in range(csr.shape[0]):
        s, e = indptr[u], indptr[u+1]
        seen_by_user[u] = indices[s:e] if e > s else np.array([], dtype=np.int32)

    rel_by_user = eval_df.groupby("user_idx")["item_idx"].apply(lambda s: set(s.astype(int)))
    users = rel_by_user.index.to_numpy()
    if max_users and len(users) > max_users:
        users = np.random.default_rng(42).choice(users, size=max_users, replace=False)

    print(f"  {len(users)} utilisateurs | split={split} | seuil≥{threshold}")

    engines = ["ALS", "Hybrid", "Popularité"]
    accum = {e: {} for e in engines}
    cov = {e: set() for e in engines}

    for u in users:
        u = int(u)
        rel = rel_by_user.get(u, set())
        if not rel: continue
        seen = seen_by_user.get(u, np.array([], dtype=np.int32))
        seen_set = set(seen.tolist()) if seen.size else set()
        mk = max(ks)

        rec_als = als_recommend(u, uf, itf, seen, mk)
        rec_pop = popularity_recommend(pop_rank, seen_set, mk)

        # Hybrid: pick a relevant item as query (simulating "user is browsing this item")
        rel_list = list(rel)
        query_item = rel_list[0]
        rel_rest = rel - {query_item}
        rec_hy = hybrid_recommend(query_item, u, X, idx_map, uf, itf, pop_scores,
                                  seen_set, mk, alpha, beta, gamma)

        cov["ALS"].update(rec_als)
        cov["Popularité"].update(rec_pop)
        cov["Hybrid"].update(rec_hy)

        for eng, rec, rel_set in [("ALS", rec_als, rel),
                                   ("Hybrid", rec_hy, rel_rest if rel_rest else rel),
                                   ("Popularité", rec_pop, rel)]:
            for k in ks:
                for mn, mv in compute_all_ir_metrics(rec, rel_set, k).items():
                    key = f"{mn}"
                    accum[eng].setdefault(key, []).append(mv)

    rows = []
    for eng in engines:
        for key, vals in accum[eng].items():
            rows.append({"Engine": eng, "Metric": key,
                         "Value": round(float(np.mean(vals)), 6), "N": len(vals)})
        rows.append({"Engine": eng, "Metric": "Coverage",
                     "Value": round(len(cov[eng]) / n_items, 6), "N": len(users)})
    return pd.DataFrame(rows)


# ═══════════════  PART 2: CATEGORY-COHERENCE EVAL  ══════════════════════

def _extract_sub_category(title: str) -> str:
    """Extract a fine-grained sub-category from a product title using keywords."""
    title_low = title.lower()
    # ordered so more specific patterns match first
    KEYWORDS = [
        ("sneakers", "sneakers"), ("boots", "boots"), ("sandals", "sandals"),
        ("socks", "socks"), ("shoes", "shoes"), ("shoe", "shoes"),
        ("earring", "earring"), ("necklace", "necklace"), ("bracelet", "bracelet"),
        ("ring", "ring"), ("sunglasses", "sunglasses"), ("glasses", "glasses"),
        ("watch", "watch"), ("hat", "hat"), ("cap", "cap"),
        ("jacket", "jacket"), ("coat", "coat"), ("hoodie", "hoodie"),
        ("dress", "dress"), ("skirt", "skirt"), ("pants", "pants"),
        ("jeans", "jeans"), ("shorts", "shorts"), ("shirt", "shirt"),
        ("blouse", "blouse"), ("tee", "tee"), ("sweater", "sweater"),
        ("vest", "vest"), ("scarf", "scarf"), ("gloves", "gloves"),
        ("belt", "belt"), ("bag", "bag"), ("purse", "purse"),
        ("wallet", "wallet"), ("backpack", "backpack"), ("legging", "legging"),
        ("bra", "bra"), ("underwear", "underwear"), ("tie", "tie"),
        ("swimsuit", "swimsuit"), ("bikini", "bikini"),
    ]
    for kw, cat in KEYWORDS:
        if kw in title_low:
            return cat
    return "other"


def eval_category_coherence(ks, max_queries=1000, alpha=0.5, beta=0.4, gamma=0.1):
    """Evaluate Multimodal KNN and Hybrid on category coherence (sub-category from titles)."""
    print("\n── Part 2 : Évaluation cohérence catégorielle ──")
    items_df = load_items()
    X, idx_map, mm_ids = load_multimodal_embeddings()
    reverse_map = {v: k for k, v in idx_map.items()}
    uf, itf, csr = load_als()
    n_items = itf.shape[0]

    pop_scores = np.asarray(csr.sum(axis=0)).ravel().astype(np.float32)

    # build category map using keyword-extracted sub-categories from titles
    cat_map: Dict[int, str] = {}
    for _, row in items_df.iterrows():
        iid = int(row["item_idx"])
        title = str(row.get("title", ""))
        cat = _extract_sub_category(title)
        cat_map[iid] = cat

    # items with both embedding and a non-"other" sub-category
    eligible = [iid for iid in mm_ids if cat_map.get(iid, "other") != "other"]
    rng = np.random.default_rng(42)
    if max_queries and len(eligible) > max_queries:
        eligible = rng.choice(eligible, size=max_queries, replace=False).tolist()

    print(f"  {len(eligible)} produits évalués | {len(set(cat_map.values()))} catégories uniques")

    engines = ["Multimodal KNN", "Hybrid"]
    accum = {e: {} for e in engines}

    for query_item in eligible:
        query_item = int(query_item)
        query_cat = cat_map[query_item]
        mk = max(ks)

        # Multimodal KNN
        rec_mm = multimodal_knn(query_item, X, idx_map, reverse_map, mk)
        rec_mm_cats = [cat_map.get(r, "") for r in rec_mm]

        # Hybrid (no user → beta=0 fallback, but we can use user 0 as proxy)
        rec_hy = hybrid_recommend(query_item, 0, X, idx_map, uf, itf, pop_scores,
                                  set(), mk, alpha, beta, gamma)
        rec_hy_cats = [cat_map.get(r, "") for r in rec_hy]

        for eng, cats in [("Multimodal KNN", rec_mm_cats), ("Hybrid", rec_hy_cats)]:
            for k in ks:
                cp = cat_precision_at_k(query_cat, cats, k)
                ch = cat_hit_at_k(query_cat, cats, k)
                accum[eng].setdefault(f"Cat-Precision@{k}", []).append(cp)
                accum[eng].setdefault(f"Cat-HitRate@{k}", []).append(ch)

    # random baseline
    all_cats = [cat_map.get(iid, "") for iid in mm_ids if iid in cat_map]
    rand_prec = {k: [] for k in ks}
    for query_item in eligible:
        query_cat = cat_map[int(query_item)]
        for k in ks:
            sample = rng.choice(all_cats, size=k, replace=False).tolist()
            rand_prec[k].append(cat_precision_at_k(query_cat, sample, k))

    rows = []
    for eng in engines:
        for key, vals in accum[eng].items():
            rows.append({"Engine": eng, "Metric": key,
                         "Value": round(float(np.mean(vals)), 6), "N": len(vals)})
    for k in ks:
        rows.append({"Engine": "Random (baseline)", "Metric": f"Cat-Precision@{k}",
                     "Value": round(float(np.mean(rand_prec[k])), 6), "N": len(eligible)})
    return pd.DataFrame(rows)


# ════════════════  PART 3: SURROGATE MODEL EVAL  ════════════════════════

def eval_surrogate():
    """Evaluate SHAP surrogate model quality."""
    print("\n── Part 3 : Évaluation du modèle surrogate (SHAP) ──")
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    surr_path = Path("data/models/surrogate_rf.joblib")
    if not surr_path.exists():
        print("  [SKIP] Modèle surrogate introuvable.")
        return pd.DataFrame()

    bundle = joblib.load(surr_path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle

    X_mm, idx_map, mm_ids = load_multimodal_embeddings()
    uf, itf, csr = load_als()
    pop_scores = np.asarray(csr.sum(axis=0)).ravel().astype(np.float32)
    reverse = {v: k for k, v in idx_map.items()}

    # generate held-out test pairs (different seed than training seed=42)
    n = X_mm.shape[0]
    rng = np.random.default_rng(99)
    pairs = rng.integers(0, n, size=(5000, 2))

    feats, targets = [], []
    for a, b in pairs:
        va, vb = X_mm[a], X_mm[b]
        cos = float(np.dot(va, vb) / ((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12)))
        aid, bid = reverse.get(a, -1), reverse.get(b, -1)
        als = 0.0
        if aid >= 0 and bid >= 0:
            try:
                als = float(np.dot(itf[min(aid, itf.shape[0]-1)].astype(np.float32),
                                   itf[min(bid, itf.shape[0]-1)].astype(np.float32)))
            except: pass
        pop = float(pop_scores[min(bid, len(pop_scores)-1)]) if bid >= 0 else 0.0
        target = 0.6 * cos + 0.35 * als + 0.05 * pop
        feats.append([cos, als, pop])
        targets.append(target)

    Xt = np.array(feats, dtype=np.float32)
    yt = np.array(targets, dtype=np.float32)
    yp = model.predict(Xt)

    rows = [
        {"Engine": "Surrogate RF", "Metric": "R²",   "Value": round(r2_score(yt, yp), 6), "N": len(yt)},
        {"Engine": "Surrogate RF", "Metric": "RMSE", "Value": round(float(np.sqrt(mean_squared_error(yt, yp))), 6), "N": len(yt)},
        {"Engine": "Surrogate RF", "Metric": "MAE",  "Value": round(float(mean_absolute_error(yt, yp)), 6), "N": len(yt)},
    ]

    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        for name, imp in zip(["multimodal_cosine", "als_dot", "popularity"], fi):
            rows.append({"Engine": "Surrogate RF", "Metric": f"Importance({name})",
                         "Value": round(float(imp), 6), "N": len(yt)})

    # SHAP explainability test
    try:
        import shap
        expl = shap.Explainer(model)
        sv = expl(Xt[:100])
        mean_abs = np.mean(np.abs(sv.values), axis=0)
        for name, val in zip(["multimodal_cosine", "als_dot", "popularity"], mean_abs):
            rows.append({"Engine": "Surrogate RF", "Metric": f"MeanAbsSHAP({name})",
                         "Value": round(float(val), 6), "N": 100})
        print("  SHAP values calculés ✓")
    except Exception as e:
        print(f"  SHAP non disponible: {e}")

    return pd.DataFrame(rows)


# ═══════════════════════════  MAIN  ═════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Évaluation complète du système de recommandation")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--max_users", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.1)
    args = ap.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    ks = [1, 5, 10, 20]

    print("=" * 65)
    print("  ÉVALUATION COMPLÈTE DU SYSTÈME DE RECOMMANDATION")
    print("=" * 65)

    # Part 1: user-interaction based
    df1 = eval_user_interaction(args.split, ks, args.threshold, args.max_users,
                                args.alpha, args.beta, args.gamma)

    # Part 2: category coherence
    df2 = eval_category_coherence(ks, max_queries=1000,
                                  alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    # Part 3: surrogate
    df3 = eval_surrogate()

    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    out = REPORT_DIR / "evaluation_metrics_all.csv"
    df_all.to_csv(out, index=False)
    print(f"\n✅ Rapport CSV sauvegardé : {out}")

    # ── pretty print ──
    print("\n" + "=" * 65)
    print("  RÉSULTATS")
    print("=" * 65)

    for eng in df_all["Engine"].unique():
        sub = df_all[df_all["Engine"] == eng].sort_values("Metric")
        print(f"\n┌── {eng} {'─' * max(1, 40 - len(eng))}┐")
        for _, row in sub.iterrows():
            bar = "█" * int(row["Value"] * 40) if row["Value"] <= 1 else ""
            print(f"│  {row['Metric']:30s}  {row['Value']:.6f}  {bar}")
        print(f"└{'─' * 46}┘")

    # ── comparison tables ──
    print("\n\n╔══ Tableau comparatif — Interactions utilisateur ══╗\n")
    ir_engines = ["ALS", "Hybrid", "Popularité"]
    ir_rows = df1[df1["Engine"].isin(ir_engines) & ~df1["Metric"].str.contains("Coverage")]
    if not ir_rows.empty:
        piv = ir_rows.pivot(index="Metric", columns="Engine", values="Value")
        col_order = [c for c in ir_engines if c in piv.columns]
        print(piv[col_order].to_string())

    print("\n\n╔══ Tableau comparatif — Cohérence catégorielle ══╗\n")
    cat_engines = ["Multimodal KNN", "Hybrid", "Random (baseline)"]
    cat_rows = df2[df2["Engine"].isin(cat_engines)]
    if not cat_rows.empty:
        piv2 = cat_rows.pivot(index="Metric", columns="Engine", values="Value")
        col_order2 = [c for c in cat_engines if c in piv2.columns]
        print(piv2[col_order2].to_string())


if __name__ == "__main__":
    main()
