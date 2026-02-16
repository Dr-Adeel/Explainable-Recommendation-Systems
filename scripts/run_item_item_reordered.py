from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

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


def build_popularity_rank(train_df: pd.DataFrame) -> list[int]:
    return train_df["product_id"].value_counts().index.astype("int64").tolist()


def build_user_seen(train_df: pd.DataFrame, user_ids: pd.Index) -> pd.Series:
    subset = train_df[train_df["user_id"].isin(user_ids)]
    return subset.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))


def build_user_reorder_strength(
    train_df: pd.DataFrame,
    user_ids: pd.Index,
    min_purchases: int = 2,
) -> dict[tuple[int, int], float]:
    """
    Retourne un dict (user_id, product_id) -> reorder_rate (0..1)
    On garde uniquement les produits que le user a acheté au moins `min_purchases` fois.
    """
    subset = train_df[train_df["user_id"].isin(user_ids)]

    # nb d'achats user-produit
    cnt = subset.groupby(["user_id", "product_id"]).size().rename("cnt")

    # taux de reorder = moyenne de reordered (0/1)
    rr = subset.groupby(["user_id", "product_id"])["reordered"].mean().rename("reorder_rate")

    stats = pd.concat([cnt, rr], axis=1).reset_index()
    stats = stats[stats["cnt"] >= min_purchases]

    return {(int(u), int(p)): float(r) for u, p, _, r in stats.itertuples(index=False)}


def recommend_item_item_reordered(
    user_id: int,
    seen: set[int],
    neighbors: dict[int, list[tuple[int, float]]],
    pop_rank: list[int],
    reorder_strength: dict[tuple[int, int], float],
    k: int = 10,
    beta: float = 1.5,
    pop_fill_cap: int = 2000,
) -> list[int]:
    """
    Item–Item + re-buy boost:
      score += s pour candidat nouveau
      score += beta * s * (0.5 + reorder_rate) pour candidat déjà vu
    Puis on complète avec popularité si besoin.

    beta : intensité du boost re-buy
    """
    scores = defaultdict(float)

    for pid in seen:
        for nid, s in neighbors.get(pid, []):
            if nid in seen:
                rr = reorder_strength.get((user_id, nid), 0.0)  # 0..1
                scores[nid] += beta * s * (0.5 + rr)  # >= 0.5*beta*s même si rr=0
            else:
                scores[nid] += s

    # Fallback / stabilisation: ajouter un petit score aux best-sellers
    # (sinon, si pas assez de voisins, score vide)
    for pid in pop_rank[:pop_fill_cap]:
        pid = int(pid)
        scores[pid] += 0.02  # très léger, juste pour remplir

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

    # sécurité: si jamais
    if len(rec) < k:
        for pid in pop_rank:
            pid = int(pid)
            if pid in used:
                continue
            rec.append(pid)
            used.add(pid)
            if len(rec) >= k:
                break

    return rec[:k]


def main():
    K = 10
    MAX_USERS = 5000  # mets None pour tout évaluer
    MIN_PURCHASES = 2  # pour considérer un produit comme "vraiment" re-buy
    BETA = 1.5         # boost re-buy (on pourra tester 1.0 / 2.0)

    print("📥 Chargement train/test...")
    train_df = load_parquet("interactions_train.parquet", columns=["user_id", "product_id", "reordered"])
    test_df = load_parquet("interactions_test.parquet", columns=["user_id", "product_id"])

    print("📊 Popularité globale...")
    pop_rank = build_popularity_rank(train_df)

    print("🧩 Test par user...")
    test_by_user = test_df.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))
    if MAX_USERS is not None and len(test_by_user) > MAX_USERS:
        test_by_user = test_by_user.sample(n=MAX_USERS, random_state=42)

    print("👀 Seen (train) par user...")
    seen_by_user = build_user_seen(train_df, test_by_user.index)

    print("🔁 Calcul reorder_strength (user, product) -> reorder_rate ...")
    reorder_strength = build_user_reorder_strength(train_df, test_by_user.index, min_purchases=MIN_PURCHASES)
    print(f"✅ Nb couples (user,product) avec min_purchases={MIN_PURCHASES}: {len(reorder_strength)}")

    print("🔗 Chargement voisins item–item...")
    neighbors = load_neighbors()

    # Exemple
    example_user = int(test_by_user.index[0])
    example_seen = seen_by_user.get(example_user, set())
    example_rec = recommend_item_item_reordered(
        example_user, example_seen, neighbors, pop_rank, reorder_strength, k=K, beta=BETA
    )
    print(f"\n👤 Exemple user_id={example_user}")
    print(f"Nb produits vus (train): {len(example_seen)}")
    print(f"Reco top-{K} (product_id): {example_rec}")

    print(f"\n🧪 Évaluation Item–Item + reordered boost (K={K}, beta={BETA}) sur {len(test_by_user)} users...")
    precisions, recalls = [], []

    for user_id, relevant_set in test_by_user.items():
        seen = seen_by_user.get(user_id, set())
        rec = recommend_item_item_reordered(user_id, seen, neighbors, pop_rank, reorder_strength, k=K, beta=BETA)
        precisions.append(precision_at_k(rec, relevant_set, K))
        recalls.append(recall_at_k(rec, relevant_set, K))

    precision = float(sum(precisions) / max(len(precisions), 1))
    recall = float(sum(recalls) / max(len(recalls), 1))

    print("✅ Résultats (reordered boost)")
    print(f"Users évalués: {len(test_by_user)}")
    print(f"Precision@{K}: {precision:.4f}")
    print(f"Recall@{K}:    {recall:.4f}")


if __name__ == "__main__":
    main()
