"""
Global Explanations — Model Behavior, Confidence & Data Patterns
================================================================
Provides a bird's-eye view of the recommendation model:
  1. Global feature importance (from Random Forest surrogate)
  2. Model confidence distribution (inter-tree variance)
  3. Dataset patterns (category distribution, popularity, density)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────── Data classes ───────────────────────────

@dataclass
class FeatureImportance:
    """Global importances from the surrogate RF."""
    feature_names: List[str]
    importances: List[float]          # values from model.feature_importances_
    description_fr: str = ""
    description_en: str = ""


@dataclass
class ConfidenceStats:
    """Summary of model confidence across a sample of predictions."""
    mean_confidence: float
    median_confidence: float
    std_confidence: float
    histogram_bins: List[float]
    histogram_counts: List[int]
    n_samples: int
    description_fr: str = ""
    description_en: str = ""


@dataclass
class DataPatterns:
    """High-level dataset statistics."""
    n_items: int
    n_users: int
    n_interactions: int
    density_pct: float
    top_categories: List[Dict[str, Any]]     # [{name, count, pct}, ...]
    top_popular_items: List[Dict[str, Any]]  # [{item_idx, title, interactions}, ...]
    score_distribution: Dict[str, float]     # {mean, std, min, max, median}


@dataclass
class GlobalExplanation:
    """Aggregated global explanation."""
    feature_importance: Optional[FeatureImportance] = None
    confidence: Optional[ConfidenceStats] = None
    data_patterns: Optional[DataPatterns] = None


# ─────────────────────── Functions ──────────────────────────────

FEATURE_NAMES = ["multimodal_cosine", "als_dot", "popularity"]


def compute_feature_importance(surrogate_model) -> Optional[FeatureImportance]:
    """Extract global feature importances from a trained surrogate RF.

    Parameters
    ----------
    surrogate_model : dict or sklearn model
        Either a dict with key 'model' or a raw sklearn model with
        ``feature_importances_`` attribute.
    """
    model = surrogate_model
    if isinstance(surrogate_model, dict) and "model" in surrogate_model:
        model = surrogate_model["model"]

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    imp_list = importances.tolist()
    # Find dominant feature
    top_idx = int(np.argmax(imp_list))
    top_name = FEATURE_NAMES[top_idx] if top_idx < len(FEATURE_NAMES) else f"feature_{top_idx}"
    top_pct = round(imp_list[top_idx] * 100, 2)

    desc_fr = (
        f"Le modèle repose à {top_pct}% sur la feature « {top_name} ». "
        + ", ".join(
            f"{FEATURE_NAMES[i]}: {round(v * 100, 2)}%"
            for i, v in enumerate(imp_list)
        )
        + "."
    )
    desc_en = (
        f"The model relies {top_pct}% on '{top_name}'. "
        + ", ".join(
            f"{FEATURE_NAMES[i]}: {round(v * 100, 2)}%"
            for i, v in enumerate(imp_list)
        )
        + "."
    )

    return FeatureImportance(
        feature_names=FEATURE_NAMES,
        importances=imp_list,
        description_fr=desc_fr,
        description_en=desc_en,
    )


def compute_confidence(
    surrogate_model,
    X_sample: np.ndarray,
    n_bins: int = 10,
) -> Optional[ConfidenceStats]:
    """Compute model confidence using inter-tree prediction variance.

    For a Random Forest, each tree gives a prediction. Low variance → high
    confidence, high variance → low confidence.  We define:
        confidence = 1 − normalised_std

    Parameters
    ----------
    surrogate_model : dict or sklearn model
    X_sample : (N, 3) array of feature vectors to evaluate
    n_bins : number of histogram bins
    """
    model = surrogate_model
    if isinstance(surrogate_model, dict) and "model" in surrogate_model:
        model = surrogate_model["model"]

    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) == 0:
        return None

    # Collect predictions from each tree
    preds = np.array([tree.predict(X_sample) for tree in estimators])  # (n_trees, n_samples)
    stds = preds.std(axis=0)  # per-sample std

    # Normalise: divide by prediction range so values are comparable
    pred_range = preds.max() - preds.min()
    if pred_range > 1e-12:
        norm_std = stds / pred_range
    else:
        norm_std = stds

    confidence = np.clip(1.0 - norm_std, 0, 1)

    counts, bin_edges = np.histogram(confidence, bins=n_bins, range=(0, 1))

    mean_c = float(np.mean(confidence))
    desc_fr = (
        f"Confiance moyenne du modèle : {mean_c:.1%} "
        f"(calculée sur {len(confidence)} paires via la variance inter-arbres du Random Forest)."
    )
    desc_en = (
        f"Average model confidence: {mean_c:.1%} "
        f"(computed over {len(confidence)} pairs via Random Forest inter-tree variance)."
    )

    return ConfidenceStats(
        mean_confidence=round(mean_c, 4),
        median_confidence=round(float(np.median(confidence)), 4),
        std_confidence=round(float(np.std(confidence)), 4),
        histogram_bins=[round(float(b), 3) for b in bin_edges.tolist()],
        histogram_counts=counts.tolist(),
        n_samples=int(len(confidence)),
        description_fr=desc_fr,
        description_en=desc_en,
    )


def compute_data_patterns(
    items_df: pd.DataFrame,
    pop_scores: Optional[np.ndarray] = None,
    n_users: int = 0,
    n_interactions: int = 0,
    top_n: int = 10,
) -> DataPatterns:
    """Compute high-level dataset statistics.

    Parameters
    ----------
    items_df : DataFrame with at least item_idx, title, main_category
    pop_scores : normalised popularity array (index = item_idx)
    n_users, n_interactions : numbers from the interaction matrix
    top_n : how many top categories / items to return
    """
    n_items = len(items_df)
    density = (n_interactions / (n_users * n_items) * 100) if (n_users > 0 and n_items > 0) else 0.0

    # Category distribution
    cat_col = "main_category" if "main_category" in items_df.columns else None
    top_cats: List[Dict[str, Any]] = []
    if cat_col:
        vc = items_df[cat_col].value_counts().head(top_n)
        total = len(items_df)
        top_cats = [
            {"name": str(cat), "count": int(cnt), "pct": round(cnt / total * 100, 1)}
            for cat, cnt in vc.items()
        ]

    # Top popular items
    top_items: List[Dict[str, Any]] = []
    if pop_scores is not None and len(pop_scores) > 0:
        top_idxs = np.argsort(-pop_scores)[:top_n]
        for idx in top_idxs:
            row = items_df.loc[items_df["item_idx"] == int(idx)]
            title = str(row.iloc[0].get("title", "")) if not row.empty else ""
            top_items.append({
                "item_idx": int(idx),
                "title": title[:80],
                "popularity_score": round(float(pop_scores[idx]), 4),
            })

    # Score distribution (popularity)
    score_dist: Dict[str, float] = {}
    if pop_scores is not None and len(pop_scores) > 0:
        score_dist = {
            "mean": round(float(np.mean(pop_scores)), 4),
            "std": round(float(np.std(pop_scores)), 4),
            "min": round(float(np.min(pop_scores)), 4),
            "max": round(float(np.max(pop_scores)), 4),
            "median": round(float(np.median(pop_scores)), 4),
        }

    return DataPatterns(
        n_items=n_items,
        n_users=n_users,
        n_interactions=n_interactions,
        density_pct=round(density, 4),
        top_categories=top_cats,
        top_popular_items=top_items,
        score_distribution=score_dist,
    )


def build_sample_features(
    multimodal_X: Optional[np.ndarray],
    multimodal_ids: Optional[list],
    als_item_factors: Optional[np.ndarray],
    pop_scores: Optional[np.ndarray],
    n_samples: int = 2000,
) -> np.ndarray:
    """Generate a sample of (multimodal_cosine, als_dot, popularity) feature vectors
    for confidence estimation.  Same logic as shap_surrogate.build_training_data but
    lighter weight.
    """
    if multimodal_X is None or len(multimodal_X) == 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = np.random.default_rng(42)
    n_items = multimodal_X.shape[0]
    pairs = rng.integers(0, n_items, size=(n_samples, 2))

    feats = []
    for a, b in pairs:
        va, vb = multimodal_X[a], multimodal_X[b]
        cos = float(np.dot(va, vb) / ((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12)))

        als_sim = 0.0
        if als_item_factors is not None and multimodal_ids is not None:
            try:
                ai = als_item_factors[multimodal_ids[a]]
                bi = als_item_factors[multimodal_ids[b]]
                als_sim = float(np.dot(ai.astype(np.float32), bi.astype(np.float32)))
            except Exception:
                als_sim = 0.0

        pop = 0.0
        if pop_scores is not None and multimodal_ids is not None:
            try:
                pop = float(pop_scores[multimodal_ids[b]])
            except Exception:
                pop = 0.0

        feats.append([cos, als_sim, pop])

    return np.asarray(feats, dtype=np.float32)
