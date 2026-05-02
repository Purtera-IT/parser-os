"""Stable public data models for Parser OS segmentation.

The current implementation re-exports the proven dataclasses so existing
callers keep exact equality/serialization behavior while the pipeline is split
around them.  Future passes should depend on this module, not on the core
engine file.
"""
from __future__ import annotations

from .._core_detect_standalone import Rect, VisibleBox, VisibleBoxResult

__all__ = ["Rect", "VisibleBox", "VisibleBoxResult"]
