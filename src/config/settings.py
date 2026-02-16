"""
Settings — Domain-agnostic configuration loader.
==================================================
Reads a domain YAML config and exposes a global ``DomainAdapter`` instance
that all modules (API, frontend, scripts) can import.

Usage
-----
    from src.config.settings import get_adapter, get_config

    adapter = get_adapter()           # DomainAdapter instance
    config  = get_config()            # raw dict from YAML

To switch domain, set the environment variable before starting:
    $env:RECO_DOMAIN = "healthcare"   # PowerShell
    export RECO_DOMAIN=healthcare     # Linux/macOS

Default domain: ``ecommerce``
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.domain_adapter import DomainAdapter
from src.config.yaml_adapter import YAMLDomainAdapter


# ── constants ──
_DOMAINS_DIR = Path(__file__).parent / "domains"
_DEFAULT_DOMAIN = "ecommerce"

# ── module-level singletons ──
_adapter: Optional[DomainAdapter] = None
_config: Optional[Dict[str, Any]] = None


def _resolve_domain() -> str:
    """Determine which domain to load (env var or default)."""
    return os.environ.get("RECO_DOMAIN", _DEFAULT_DOMAIN).strip().lower()


def _load_config(domain: str) -> Dict[str, Any]:
    """Load raw YAML config for the given domain."""
    path = _DOMAINS_DIR / f"{domain}.yaml"
    if not path.exists():
        available = [f.stem for f in _DOMAINS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Domain config '{domain}.yaml' not found in {_DOMAINS_DIR}. "
            f"Available: {available}"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config(domain: Optional[str] = None) -> Dict[str, Any]:
    """Return the raw configuration dict for the active domain."""
    global _config
    domain = domain or _resolve_domain()
    if _config is None or _config.get("domain") != domain:
        _config = _load_config(domain)
    return _config


def get_adapter(domain: Optional[str] = None) -> DomainAdapter:
    """Return the ``DomainAdapter`` singleton for the active domain."""
    global _adapter
    domain = domain or _resolve_domain()
    if _adapter is None or _adapter.domain_name != domain:
        path = _DOMAINS_DIR / f"{domain}.yaml"
        _adapter = YAMLDomainAdapter(path)
    return _adapter


def list_available_domains() -> list[str]:
    """Return the list of available domain identifiers."""
    return sorted(f.stem for f in _DOMAINS_DIR.glob("*.yaml"))
