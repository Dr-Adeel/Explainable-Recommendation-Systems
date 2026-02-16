"""
Domain Adapter — Abstract interface for domain-agnostic recommendations.
=========================================================================
Each domain (e-commerce, healthcare, education, …) implements this
interface so the recommendation engine can work identically regardless
of the underlying data semantics.

The engine always manipulates generic concepts:
  item_id, user_id, title, category, embedding, score
Only the adapter knows how to map those to domain-specific columns,
paths, and explanation texts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DomainAdapter(ABC):
    """Abstract base class that every domain must implement."""

    # ── identity ──────────────────────────────────────────────

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short identifier, e.g. 'ecommerce', 'healthcare'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI."""
        ...

    # ── data loading ──────────────────────────────────────────

    @abstractmethod
    def load_items(self) -> pd.DataFrame:
        """Load the item catalog with columns renamed to generic names:
        item_id, item_raw_id, title, category, image_path.
        """
        ...

    @abstractmethod
    def load_interactions(self, split: str = "train") -> pd.DataFrame:
        """Load interactions for a given split (train/valid/test)
        with columns: user_id, item_id, value.
        """
        ...

    # ── column mapping helpers ────────────────────────────────

    @abstractmethod
    def get_column_map(self) -> Dict[str, str]:
        """Return mapping from generic column name → domain column name.
        Example: {'item_id': 'treatment_id', 'title': 'treatment_name'}
        """
        ...

    # ── display ───────────────────────────────────────────────

    @abstractmethod
    def get_item_display(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format an item dict for UI display.
        Must return: {title, category, image_url, description}.
        """
        ...

    @abstractmethod
    def entity_labels(self) -> Dict[str, str]:
        """Return human-readable labels for generic entities.
        Keys: item_singular, item_plural, user_singular, user_plural,
              interaction, category_label.
        """
        ...

    # ── explanations ──────────────────────────────────────────

    @abstractmethod
    def explain_reason(
        self,
        shares: Dict[str, float],
        lang: str = "fr",
    ) -> str:
        """Generate a domain-specific explanation sentence.

        Parameters
        ----------
        shares : dict with keys 'image', 'als', 'pop' (percentages 0-100)
        lang   : 'fr' or 'en'
        """
        ...

    # ── paths ─────────────────────────────────────────────────

    @abstractmethod
    def get_paths(self) -> Dict[str, Path]:
        """Return a dict of all data paths the engine needs:
        items_path, embeddings_path, catalog_path, multimodal_embeddings_path,
        faiss_index_path, surrogate_path, feedback_path, images_dir, ...
        """
        ...

    @abstractmethod
    def get_als_config(self) -> Dict[str, Any]:
        """Return ALS-specific configuration:
        model_dirs, csr_filename, factors, iterations, ...
        """
        ...

    @abstractmethod
    def get_engine_defaults(self) -> Dict[str, Any]:
        """Return default hybrid weights and model names."""
        ...
