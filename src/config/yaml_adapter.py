"""
YAML-based Domain Adapter — Concrete implementation that reads a domain
YAML configuration file and implements the DomainAdapter interface.
=======================================================================
This is the main adapter used in production.  Instead of writing a new
Python class for each domain, you just write a YAML file in
``src/config/domains/<domain>.yaml`` and this adapter does the rest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.config.domain_adapter import DomainAdapter


class YAMLDomainAdapter(DomainAdapter):
    """Generic adapter driven entirely by a YAML config file."""

    def __init__(self, config_path: str | Path):
        with open(config_path, encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    # ── identity ──

    @property
    def domain_name(self) -> str:
        return self._cfg["domain"]

    @property
    def display_name(self) -> str:
        return self._cfg["display_name"]

    # ── column mapping ──

    def get_column_map(self) -> Dict[str, str]:
        return dict(self._cfg["columns"])

    def _col(self, generic: str) -> str:
        """Shortcut: generic column name → domain column name."""
        return self._cfg["columns"].get(generic, generic)

    # ── data loading ──

    def load_items(self) -> pd.DataFrame:
        path = Path(self._cfg["data"]["items_path"])
        if not path.exists():
            raise FileNotFoundError(f"Items file not found: {path}")
        df = pd.read_parquet(path)
        # Rename domain-specific columns to generic names
        col_map = self.get_column_map()
        reverse = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse)
        return df

    def load_interactions(self, split: str = "train") -> pd.DataFrame:
        items_path = Path(self._cfg["data"]["items_path"])
        base_dir = items_path.parent
        inter_path = base_dir / f"interactions_{split}.parquet"
        if not inter_path.exists():
            raise FileNotFoundError(f"Interactions file not found: {inter_path}")
        df = pd.read_parquet(inter_path)
        col_map = self.get_column_map()
        reverse = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse)
        return df

    # ── display ──

    def get_item_display(self, item: Dict[str, Any]) -> Dict[str, str]:
        col = self._cfg["columns"]
        return {
            "title": str(item.get(col["title"], item.get("title", ""))),
            "category": str(item.get(col["category"], item.get("category", ""))),
            "image_url": str(item.get(col["image_path"], item.get("image_path", ""))),
            "description": "",
        }

    def entity_labels(self) -> Dict[str, str]:
        return dict(self._cfg["entities"])

    # ── explanations ──

    def explain_reason(
        self,
        shares: Dict[str, float],
        lang: str = "fr",
    ) -> str:
        templates = self._cfg.get("explanations", {}).get(lang, {})
        if not templates:
            templates = self._cfg.get("explanations", {}).get("fr", {})

        img_pct = shares.get("image", 0)
        als_pct = shares.get("als", 0)
        pop_pct = shares.get("pop", 0)

        parts = sorted(
            [("image", img_pct), ("als", als_pct), ("pop", pop_pct)],
            key=lambda x: x[1],
            reverse=True,
        )
        top_name, top_val = parts[0]
        second_name, second_val = parts[1]

        if top_val < 1:
            return templates.get("fallback", "Recommended based on combined signals.")

        if top_name == "image":
            if second_name == "als" and second_val > 15:
                return templates.get("image_reinforced_als", "").format(
                    pct=f"{img_pct:.0f}", als_pct=f"{als_pct:.0f}", img_pct=f"{img_pct:.0f}"
                )
            return templates.get("image_dominant", "").format(
                pct=f"{img_pct:.0f}", als_pct=f"{als_pct:.0f}", img_pct=f"{img_pct:.0f}"
            )

        if top_name == "als":
            if second_name == "image" and second_val > 15:
                return templates.get("als_reinforced_image", "").format(
                    pct=f"{als_pct:.0f}", img_pct=f"{img_pct:.0f}", als_pct=f"{als_pct:.0f}"
                )
            return templates.get("als_dominant", "").format(
                pct=f"{als_pct:.0f}", img_pct=f"{img_pct:.0f}", als_pct=f"{als_pct:.0f}"
            )

        # popularity dominant
        return templates.get("popularity_dominant", "").format(
            pct=f"{pop_pct:.0f}", img_pct=f"{img_pct:.0f}", als_pct=f"{als_pct:.0f}"
        )

    # ── paths ──

    def get_paths(self) -> Dict[str, Path]:
        d = self._cfg["data"]
        paths = {}
        for key in [
            "items_path", "embeddings_path", "catalog_path",
            "multimodal_embeddings_path", "faiss_index_path",
            "text_embeddings_path", "surrogate_path", "feedback_path",
            "images_dir",
        ]:
            if key in d:
                paths[key] = Path(d[key])
        return paths

    def get_als_config(self) -> Dict[str, Any]:
        als_data = self._cfg["data"].get("als", {})
        als_engine = self._cfg["engines"].get("als", {})
        return {
            "model_dirs": [Path(p) for p in als_data.get("model_dirs", [])],
            "csr_filename": als_data.get("csr_filename", "train_csr.npz"),
            **als_engine,
        }

    def get_engine_defaults(self) -> Dict[str, Any]:
        return dict(self._cfg["engines"])
