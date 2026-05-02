"""Configuration boundary for detection, synthesis, overlay, and QA.

`Cfg` intentionally aliases the core detector config in this migration step:
that preserves every existing tuning knob and default.  New semantic passes add
small dataclasses here and are wired through the pass registry instead of adding
another branch to the detector monolith.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._core_detect_standalone import Cfg


@dataclass(frozen=True)
class PassRuntimeConfig:
    """Common switchboard for one modular pass."""

    enabled: bool = False
    emit_layer: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level extensibility config.

    `detector` is the battle-tested v1 `Cfg`.  `semantic_passes` is where v2
    modules such as symbols, row grouping, revision clouds, legends, and
    cross-reference bubbles plug in.  Keeping those knobs outside `Cfg` prevents
    new concerns from bloating the raster contour detector.
    """

    detector: Cfg = field(default_factory=Cfg)
    semantic_passes: dict[str, PassRuntimeConfig] = field(default_factory=dict)


__all__ = ["Cfg", "PipelineConfig", "PassRuntimeConfig"]
