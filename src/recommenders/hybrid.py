from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass(frozen=True)
class HybridItem:
    product_id: int
    score: float
    parts: Dict[str, float]
    reason: str


def _minmax_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {pid: 1.0 for pid in scores}
    return {pid: (s - mn) / (mx - mn) for pid, s in scores.items()}


def _build_reason(img_c: float, user_c: float, pop_c: float) -> str:
    parts = sorted([("image", img_c), ("user", user_c), ("popularity", pop_c)], key=lambda x: x[1], reverse=True)
    top, second = parts[0], parts[1]

    if top[1] <= 1e-9:
        return "Recommended via fallback popularity (cold-start)."

    if top[0] == "image":
        if second[0] == "user" and second[1] > 0.15 * top[1]:
            return "Recommended due to visual similarity and consistency with the user's history."
        return "Recommended mainly due to visual similarity."
    if top[0] == "user":
        if second[0] == "image" and second[1] > 0.15 * top[1]:
            return "Recommended due to the user's history and visual similarity."
        return "Recommended mainly due to the user's purchase history."
    return "Recommended mainly due to overall popularity (fallback)."


def fuse_hybrid(
    image_scores: Dict[int, float],
    user_scores: Dict[int, float],
    pop_scores: Optional[Dict[int, float]] = None,
    k: int = 10,
    alpha: float = 0.6,
    gamma: float = 0.05,
) -> List[HybridItem]:
    """
    Combine image and user recommendation signals with simple, explainable weighting.

    score_final = alpha * norm(image) + (1-alpha) * norm(user) + gamma * norm(popularity)
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    gamma = float(max(0.0, min(1.0, gamma)))

    img_n = _minmax_norm(image_scores)
    user_n = _minmax_norm(user_scores)
    pop_n = _minmax_norm(pop_scores or {})

    candidates = set(img_n) | set(user_n) | set(pop_n)
    if not candidates:
        return []

    final_scores: Dict[int, float] = {}
    items: List[HybridItem] = []

    for pid in candidates:
        img = img_n.get(pid, 0.0)
        usr = user_n.get(pid, 0.0)
        pop = pop_n.get(pid, 0.0)

        img_c = alpha * img
        user_c = (1.0 - alpha) * usr
        pop_c = gamma * pop

        score = img_c + user_c + pop_c
        final_scores[pid] = score

        reason = _build_reason(img_c, user_c, pop_c)
        items.append(
            HybridItem(
                product_id=int(pid),
                score=float(score),
                parts={
                    "image": float(img_c),
                    "user": float(user_c),
                    "popularity": float(pop_c),
                },
                reason=reason,
            )
        )

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:k]
