"""Causal-action world-model components.

This package is intentionally separate from the legacy training and data paths.
"""

from .config import load_and_resolve_config, resolve_config
from .model import CausalDiT

__all__ = ["CausalDiT", "load_and_resolve_config", "resolve_config"]
