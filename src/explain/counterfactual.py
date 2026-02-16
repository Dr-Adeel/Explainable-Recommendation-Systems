"""
Counterfactual Reasoning for Hybrid Recommendations
====================================================
Answers: "If we removed signal X, how would the ranking change?"

For each candidate, we re-score with one signal zeroed-out and compare
the new rank against the original rank.  This produces explanations like:
  "Sans la similarité visuelle, ce produit passerait du rang 2 au rang 8."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis for one candidate."""
    item_idx: int
    original_rank: int
    original_score: float
    scenarios: Dict[str, "ScenarioResult"]
    summary_fr: str
    summary_en: str


@dataclass
class ScenarioResult:
    """What happens when we remove one signal."""
    signal_removed: str
    new_rank: int
    rank_delta: int          # positive = dropped in ranking
    new_score: float
    score_delta: float


def compute_counterfactuals(
    img_scores: np.ndarray,
    als_scores: np.ndarray,
    pop_scores: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    candidate_ids: np.ndarray,
    query_item: int,
    top_k: int = 10,
) -> List[CounterfactualResult]:
    """Compute counterfactual explanations for the top-k hybrid recommendations.

    Parameters
    ----------
    img_scores, als_scores, pop_scores : 1-d arrays (normalised 0-1), same length as candidate_ids
    alpha, beta, gamma : hybrid weights
    candidate_ids : array of item_idx for each candidate
    query_item : the query item_idx (excluded from results)
    top_k : number of results to explain

    Returns
    -------
    List of CounterfactualResult, one per top-k item.
    """
    # ── original ranking ──
    original_combined = alpha * img_scores + beta * als_scores + gamma * pop_scores
    sorted_idx = np.argsort(-original_combined)

    # build ordered list excluding self
    ordered = [int(candidate_ids[i]) for i in sorted_idx if int(candidate_ids[i]) != query_item]

    # Build rank lookup {item_idx: rank (1-based)}
    original_rank_map = {iid: rank + 1 for rank, iid in enumerate(ordered)}

    # ── scenario definitions ──
    scenarios_def = {
        "sans_image":      (0.0,   beta,  gamma, "similarité visuelle",   "visual similarity"),
        "sans_als":        (alpha, 0.0,   gamma, "filtrage collaboratif", "collaborative filtering"),
        "sans_popularite": (alpha, beta,  0.0,   "popularité",            "popularity"),
    }

    # pre-compute alternative rankings
    alt_rankings: Dict[str, List[int]] = {}
    alt_scores: Dict[str, np.ndarray] = {}
    for key, (a, b, g, _, _) in scenarios_def.items():
        combined = a * img_scores + b * als_scores + g * pop_scores
        alt_sorted = np.argsort(-combined)
        alt_ordered = [int(candidate_ids[i]) for i in alt_sorted if int(candidate_ids[i]) != query_item]
        alt_rankings[key] = alt_ordered
        alt_scores[key] = combined

    # Where each item appears in alternative ranking
    alt_rank_maps: Dict[str, Dict[int, int]] = {}
    for key, alt_ordered in alt_rankings.items():
        alt_rank_maps[key] = {iid: rank + 1 for rank, iid in enumerate(alt_ordered)}

    # ── build results for top-k ──
    results: List[CounterfactualResult] = []
    for item_idx in ordered[:top_k]:
        orig_rank = original_rank_map[item_idx]
        # find score index in candidate_ids
        cand_pos = int(np.where(candidate_ids == item_idx)[0][0])
        orig_score = float(original_combined[cand_pos])

        scenarios: Dict[str, ScenarioResult] = {}
        biggest_drop_signal = ""
        biggest_drop_delta = 0

        for key, (a, b, g, label_fr, label_en) in scenarios_def.items():
            new_rank = alt_rank_maps[key].get(item_idx, len(ordered))
            rank_delta = new_rank - orig_rank
            new_score = float(alt_scores[key][cand_pos])
            score_delta = new_score - orig_score

            scenarios[key] = ScenarioResult(
                signal_removed=key,
                new_rank=new_rank,
                rank_delta=rank_delta,
                new_score=round(new_score, 4),
                score_delta=round(score_delta, 4),
            )

            if rank_delta > biggest_drop_delta:
                biggest_drop_delta = rank_delta
                biggest_drop_signal = key

        # Compose human-readable summaries
        if biggest_drop_delta > 0:
            label_fr = scenarios_def[biggest_drop_signal][3]
            label_en = scenarios_def[biggest_drop_signal][4]
            summary_fr = (
                f"Sans la {label_fr}, ce produit passerait du rang {orig_rank} "
                f"au rang {orig_rank + biggest_drop_delta}. "
                f"La {label_fr} est donc le signal le plus déterminant."
            )
            summary_en = (
                f"Without {label_en}, this product would drop from rank {orig_rank} "
                f"to rank {orig_rank + biggest_drop_delta}. "
                f"{label_en.capitalize()} is the most influential signal."
            )
        else:
            summary_fr = "Le classement de ce produit est stable quel que soit le signal retiré."
            summary_en = "This product's ranking is stable regardless of which signal is removed."

        results.append(CounterfactualResult(
            item_idx=item_idx,
            original_rank=orig_rank,
            original_score=round(orig_score, 4),
            scenarios=scenarios,
            summary_fr=summary_fr,
            summary_en=summary_en,
        ))

    return results
