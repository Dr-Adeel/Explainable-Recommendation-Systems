from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/image_hf/processed")
TRAIN_PATH = DATA_DIR / "interactions_train.parquet"
TEST_PATH = DATA_DIR / "interactions_test.parquet"
NEIGH_PATH = DATA_DIR / "fashion_item_neighbors.parquet"

K = 10
MAX_USERS = 2000  # speed

# Evaluate "intent" (cart + purchase). You can switch to {"purchase"} later.
RELEVANT_EVENTS = {"cart", "purchase"}


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = recommended[:k]
    return len(set(top) & relevant) / float(k)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    return len(set(top) & relevant) / float(len(relevant))


def load_neighbors_with_scores() -> dict[int, list[tuple[int, float]]]:
    df = pd.read_parquet(NEIGH_PATH, columns=["product_id", "neighbor_id", "score"])
    df["product_id"] = df["product_id"].astype(int)
    df["neighbor_id"] = df["neighbor_id"].astype(int)
    df["score"] = df["score"].astype(float)

    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pid, nid, s in df.itertuples(index=False):
        neighbors[int(pid)].append((int(nid), float(s)))
    return neighbors


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH, columns=["user_id", "product_id", "event"])
    test_df = pd.read_parquet(TEST_PATH, columns=["user_id", "product_id", "event"])

    train_df["user_id"] = train_df["user_id"].astype(int)
    train_df["product_id"] = train_df["product_id"].astype(int)
    test_df["user_id"] = test_df["user_id"].astype(int)
    test_df["product_id"] = test_df["product_id"].astype(int)

    # Popularity fallback from train
    pop_rank = train_df["product_id"].value_counts().index.astype("int64").tolist()

    # Seen history (train)
    seen_by_user = train_df.groupby("user_id")["product_id"].apply(lambda s: set(s.astype("int64")))

    # Relevant items (test)
    test_rel = test_df[test_df["event"].isin(RELEVANT_EVENTS)].groupby("user_id")["product_id"].apply(
        lambda s: set(s.astype("int64"))
    )

    # Keep users that actually have relevant items
    test_rel = test_rel[test_rel.apply(len) > 0]
    if len(test_rel) == 0:
        raise RuntimeError("No relevant events in test. Increase simulation or adjust probabilities.")

    if len(test_rel) > MAX_USERS:
        test_rel = test_rel.sample(n=MAX_USERS, random_state=42)

    neighbors = load_neighbors_with_scores()

    precisions, recalls = [], []

    for user_id, relevant in test_rel.items():
        seen = seen_by_user.get(user_id, set())
        if not seen:
            continue

        scores = defaultdict(float)

        # Sum neighbor scores from all seen items
        for pid in seen:
            for nid, s in neighbors.get(int(pid), []):
                if nid in seen:
                    continue
                scores[nid] += float(s)

        # Fallback popularity (light) to fill
        for pid in pop_rank[:2000]:
            pid = int(pid)
            if pid in seen:
                continue
            scores[pid] += 0.01

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rec = [pid for pid, _ in ranked[:K]]

        precisions.append(precision_at_k(rec, relevant, K))
        recalls.append(recall_at_k(rec, relevant, K))

    print(f"Users evaluated: {len(precisions)}")
    print(f"Relevant events: {sorted(RELEVANT_EVENTS)}")
    print(f"Precision@{K}: {float(np.mean(precisions)):.4f}")
    print(f"Recall@{K}:    {float(np.mean(recalls)):.4f}")


if __name__ == "__main__":
    main()
