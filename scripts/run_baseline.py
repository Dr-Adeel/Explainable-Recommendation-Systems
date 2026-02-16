from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Permet d'importer src/ quand on lance depuis scripts/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.eval.metrics import precision_at_k, recall_at_k

DATA_DIR = Path("data/processed")


def load_parquet(name: str, columns=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_parquet(path, columns=columns)


def build_popularity_ranking(train_df: pd.DataFrame) -> list[int]:
    # Popularité = nombre d'achats (occurrences) par product_id
    counts = train_df["product_id"].value_counts()
    return counts.index.astype("int64").tolist()


def build_user_seen_products(train_df: pd.DataFrame, max_users: int | None = 5000) -> pd.Series:
    # user_id -> set(product_id) déjà achetés dans le train
    grouped = train_df.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))
    if max_users is not None and len(grouped) > max_users:
        grouped = grouped.sample(n=max_users, random_state=42)
    return grouped


def evaluate_popularity_global(popularity_rank: list[int], test_by_user: pd.Series, k: int) -> dict:
    precisions, recalls = [], []
    for relevant_set in test_by_user:
        precisions.append(precision_at_k(popularity_rank, relevant_set, k))
        recalls.append(recall_at_k(popularity_rank, relevant_set, k))
    return {
        "precision": float(sum(precisions) / max(len(precisions), 1)),
        "recall": float(sum(recalls) / max(len(recalls), 1)),
    }


def evaluate_popularity_personalized(
    popularity_rank: list[int],
    test_by_user: pd.Series,
    seen_by_user: pd.Series,
    k: int,
) -> dict:
    precisions, recalls = [], []

    # Pour accélérer un peu : set pour membership rapide
    for user_id, relevant_set in test_by_user.items():
        seen = seen_by_user.get(user_id, set())

        # Construire une reco Top-K en sautant les produits déjà vus
        rec = []
        for pid in popularity_rank:
            if pid in seen:
                continue
            rec.append(pid)
            if len(rec) >= k:
                break

        precisions.append(precision_at_k(rec, relevant_set, k))
        recalls.append(recall_at_k(rec, relevant_set, k))

    return {
        "precision": float(sum(precisions) / max(len(precisions), 1)),
        "recall": float(sum(recalls) / max(len(recalls), 1)),
    }


def main():
    K = 10
    MAX_USERS = 5000  # garde 5000 pour que ça aille vite. On mettra None plus tard si tu veux.

    print("📥 Chargement des datasets (parquet)...")
    train_df = load_parquet("interactions_train.parquet", columns=["user_id", "product_id"])
    test_df = load_parquet("interactions_test.parquet", columns=["user_id", "product_id"])

    print("📊 Ranking popularité global...")
    popularity_rank = build_popularity_ranking(train_df)

    print("🧩 Groupement test par user...")
    test_by_user = test_df.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))
    if MAX_USERS is not None and len(test_by_user) > MAX_USERS:
        test_by_user = test_by_user.sample(n=MAX_USERS, random_state=42)

    print("👀 Historique (seen) par user depuis le train...")
    # Important : on prend les mêmes users que ceux qu’on évalue dans le test
    train_subset = train_df[train_df["user_id"].isin(test_by_user.index)]
    seen_by_user = train_subset.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))

    print(f"🧪 Évaluation (K={K}) sur {len(test_by_user)} users...")

    # Baseline étape 3 (global)
    res_global = evaluate_popularity_global(popularity_rank, test_by_user, K)

    # Étape 4 (personnalisée: exclure déjà vus)
    res_personal = evaluate_popularity_personalized(popularity_rank, test_by_user, seen_by_user, K)

    print("\n✅ Résultats")
    print("— Popularité globale (étape 3)")
    print(f"Precision@{K}: {res_global['precision']:.4f}")
    print(f"Recall@{K}:    {res_global['recall']:.4f}")

    print("\n— Popularité personnalisée (étape 4: exclure déjà vus)")
    print(f"Precision@{K}: {res_personal['precision']:.4f}")
    print(f"Recall@{K}:    {res_personal['recall']:.4f}")

    print("\n🔝 Top-10 popularité (ids):")
    print(popularity_rank[:10])


if __name__ == "__main__":
    main()
