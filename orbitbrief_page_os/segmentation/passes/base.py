"""Pass interfaces used by the explicit segmentation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..core.config import Cfg, PassRuntimeConfig
from ..core.models import VisibleBoxResult


@dataclass
class PageContext:
    """Immutable-ish inputs shared by detection/synthesis passes."""

    pdf_path: Path
    page_index: int
    cfg: Cfg


@dataclass
class PipelineState:
    """Mutable state that each pass can enrich."""

    result: VisibleBoxResult | None = None
    rgb: np.ndarray | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PassInfo:
    name: str
    stage: str
    layer_flag: str | None
    order: int
    description: str
    config: PassRuntimeConfig = field(default_factory=PassRuntimeConfig)


class OverlayPass(Protocol):
    info: PassInfo

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        """Return the updated state. Passes should be deterministic and local."""
        ...
