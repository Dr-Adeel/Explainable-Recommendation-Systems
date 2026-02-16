from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import math

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.eval.metrics import precision_at_k, recall_at_k

DATA_DIR = Path("data/processed")


def load_parquet(name: str, columns=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Introuvable: {path}")
    return pd.read_parquet(path, columns=columns)


def load_neighbors() -> dict[int, list[tuple[int, float]]]:
    neigh_df = load_parquet("item_neighbors.parquet", columns=["product_id", "neighbor_id", "score"])
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pid, nid, s in neigh_df.itertuples(index=False):
        neighbors[int(pid)].append((int(nid), float(s)))
    return neighbors


def build_popularity_scores(train_df: pd.DataFrame) -> dict[int, float]:
    # score_pop = log(1 + count) normalisé entre 0 et 1
    counts = train_df["product_id"].value_counts()
    pop = (counts.map(lambda x: float(x)).rename("cnt"))

    # log(1+cnt)
    pop = (pop + 1.0).map(lambda x: float(math.log(x)))  # compatible simple

    # normalisation [0,1]
    min_v = float(pop.min())
    max_v = float(pop.max())
    if max_v - min_v < 1e-12:
        return {int(pid): 0.0 for pid in pop.index.astype("int64")}
    pop_norm = (pop - min_v) / (max_v - min_v)
    return {int(pid): float(v) for pid, v in pop_norm.items()}


def build_user_history(train_df: pd.DataFrame, user_ids: pd.Index) -> pd.Series:
    subset = train_df[train_df["user_id"].isin(user_ids)]
    return subset.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))


def recommend_hybrid(
    seen: set[int],
    neighbors: dict[int, list[tuple[int, float]]],
    pop_score: dict[int, float],
    popularity_rank: list[int],
    k: int,
    alpha: float,
    candidate_cap: int = 5000,
) -> list[int]:
    """
    score_final = score_itemitem + alpha * score_pop

    candidate_cap: limite le nb de candidats considérés côté popularité pour rester rapide.
    """
    scores = defaultdict(float)

    # 1) Score item-item (re-buy autorisé)
    for pid in seen:
        for nid, s in neighbors.get(pid, []):
            # Re-buy: si déjà vu, on garde mais un peu moins fort
            if nid in seen:
                scores[nid] += 0.5 * s
            else:
                scores[nid] += s

    # 2) Ajout de popularité (sur un sous-ensemble des plus populaires pour éviter trop de candidats)
    cap = min(candidate_cap, len(popularity_rank))
    for pid in popularity_rank[:cap]:
        pid = int(pid)
        scores[pid] += alpha * pop_score.get(pid, 0.0)

    if not scores:
        # fallback pur
        return [int(x) for x in popularity_rank[:k]]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rec = []
    used = set()
    for pid, _ in ranked:
        if pid in used:
            continue
        rec.append(int(pid))
        used.add(pid)
        if len(rec) >= k:
            break
    return rec


def main():
    K = 10
    MAX_USERS = 5000  # mets None pour tout évaluer
    ALPHAS = [0.05, 0.10, 0.20]  # on teste 3 valeurs

    print("📥 Chargement train/test (parquet)...")
    train_df = load_parquet("interactions_train.parquet", columns=["user_id", "product_id"])
    test_df = load_parquet("interactions_test.parquet", columns=["user_id", "product_id"])

    print("📊 Popularité (ranking + scores normalisés)...")
    popularity_rank = train_df["product_id"].value_counts().index.astype("int64").tolist()
    pop_score = build_popularity_scores(train_df)

    print("🧩 Groupement test par user...")
    test_by_user = test_df.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))
    if MAX_USERS is not None and len(test_by_user) > MAX_USERS:
        test_by_user = test_by_user.sample(n=MAX_USERS, random_state=42)

    print("👀 Historique user (train) pour les users évalués...")
    seen_by_user = build_user_history(train_df, test_by_user.index)

    print("🔗 Chargement des voisins item–item...")
    neighbors = load_neighbors()

    # Exemple
    example_user = int(test_by_user.index[0])
    example_seen = seen_by_user.get(example_user, set())

    print("\n👤 Exemple de reco (user_id={}):".format(example_user))
    print("Nb produits vus (train):", len(example_seen))

    for a in ALPHAS:
        example_rec = recommend_hybrid(example_seen, neighbors, pop_score, popularity_rank, k=K, alpha=a)
        print(f"  alpha={a:.2f} -> {example_rec}")

    # Évaluation
    for a in ALPHAS:
        print(f"\n🧪 Évaluation HYBRID (alpha={a:.2f}) K={K} sur {len(test_by_user)} users...")
        precisions, recalls = [], []
        for user_id, relevant_set in test_by_user.items():
            seen = seen_by_user.get(user_id, set())
            rec = recommend_hybrid(seen, neighbors, pop_score, popularity_rank, k=K, alpha=a)
            precisions.append(precision_at_k(rec, relevant_set, K))
            recalls.append(recall_at_k(rec, relevant_set, K))

        precision = float(sum(precisions) / max(len(precisions), 1))
        recall = float(sum(recalls) / max(len(recalls), 1))

        print("✅ Résultats HYBRID")
        print(f"Precision@{K}: {precision:.4f}")
        print(f"Recall@{K}:    {recall:.4f}")


if __name__ == "__main__":
    main()
