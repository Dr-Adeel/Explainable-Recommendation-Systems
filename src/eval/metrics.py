from __future__ import annotations
from typing import Iterable, List, Set


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / len(topk)


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / len(relevant)
