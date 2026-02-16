from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def build_popularity_rank(train_df: pd.DataFrame) -> List[int]:
    return train_df["product_id"].value_counts().index.astype("int64").tolist()


def build_user_seen(train_df: pd.DataFrame, user_id: int) -> Set[int]:
    sub = train_df[train_df["user_id"] == user_id]
    if sub.empty:
        return set()
    return set(sub["product_id"].astype("int64").tolist())


def recommend_user_item_item(
    user_id: int,
    train_df: pd.DataFrame,
    item_neighbors: Dict[int, List[Tuple[int, float]]],
    pop_rank: List[int],
    k: int = 10,
    pop_fill_cap: int = 2000,
) -> List[int]:
    seen = build_user_seen(train_df, user_id)
    if not seen:
        return pop_rank[:k]

    scores = defaultdict(float)

    for pid in seen:
        for nid, s in item_neighbors.get(int(pid), []):
            if nid in seen:
                continue
            scores[int(nid)] += float(s)

    for pid in pop_rank[:pop_fill_cap]:
        pid = int(pid)
        if pid in seen:
            continue
        scores[pid] += 0.01

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rec = [pid for pid, _ in ranked[:k]]

    if len(rec) < k:
        used = set(rec) | seen
        for pid in pop_rank:
            pid = int(pid)
            if pid in used:
                continue
            rec.append(pid)
            used.add(pid)
            if len(rec) >= k:
                break

    return rec[:k]
