"""Explicit Parser OS overlay pipeline.

This module is the new orchestration seam.  The first migration stage keeps the
v1 detector intact as `VisibleBoxPass` so output stays byte-for-byte stable
while adding a pass registry, a stable config/model boundary, and a place for
future semantic passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .._core_detect_standalone import render_overlay as core_render_overlay
from ..overlay_layers import OverlayLayer
from ..passes import (
    CellularTitlePass,
    CoverPageTitleBandsPass,
    GridCellCompletionPass,
    MultiColContactPass,
    ProseLayoutBandsPass,
    TocLayoutBandsPass,
    VisibleBoxPass,
    OverlayPass,
    PageContext,
    PipelineState,
    RasterLineRepairPass,
    SemanticCleanupPass,
    TitleBlockDetectionPass,
)
from .config import Cfg, PipelineConfig
from .models import VisibleBoxResult


@dataclass
class OverlayPipeline:
    """Small, deterministic stage runner."""

    passes: list[OverlayPass] = field(default_factory=lambda: [
        VisibleBoxPass(),       # 100
        SemanticCleanupPass(),         # 200
        TitleBlockDetectionPass(),     # 205 — after cleanup so its boxes are never filtered
        GridCellCompletionPass(),      # 230
        CellularTitlePass(),           # 235
        ProseLayoutBandsPass(),        # 236 — fitz-driven prose page overlays
        CoverPageTitleBandsPass(),     # 236 — portrait RFP cover title/footer bands
        TocLayoutBandsPass(),          # 237 — table-of-contents specific overlays
        MultiColContactPass(),         # 238 — borderless multi-col contact blocks
        RasterLineRepairPass(),        # 240
    ])

    def run(self, pdf_path: str | Path, page_index: int = 0, cfg: Cfg | None = None) -> PipelineState:
        ctx = PageContext(pdf_path=Path(pdf_path), page_index=page_index, cfg=cfg or Cfg())
        state = PipelineState()
        for p in sorted(self.passes, key=lambda item: item.info.order):
            state = p.run(ctx, state)
        if state.result is None or state.rgb is None:
            raise RuntimeError("overlay pipeline completed without a detection result")
        return state

    def pass_table(self) -> list[dict[str, object]]:
        return [
            {
                "order": p.info.order,
                "name": p.info.name,
                "stage": p.info.stage,
                "layer_flag": p.info.layer_flag,
                "description": p.info.description,
            }
            for p in sorted(self.passes, key=lambda item: item.info.order)
        ]


def build_pipeline(config: PipelineConfig | None = None, extra_passes: Iterable[OverlayPass] | None = None) -> OverlayPipeline:
    """Build the current production pipeline.

    `extra_passes` is intended for experimental semantics under tests.  The
    default stack still begins with the compatibility pass, then applies small
    post-processing passes that make colors match the v2 legend contract.
    """
    passes: list[OverlayPass] = [
        VisibleBoxPass(),
        SemanticCleanupPass(),
        TitleBlockDetectionPass(),
        GridCellCompletionPass(),
        CellularTitlePass(),
        ProseLayoutBandsPass(),
        CoverPageTitleBandsPass(),
        TocLayoutBandsPass(),
        MultiColContactPass(),
        RasterLineRepairPass(),
    ]
    if extra_passes:
        passes.extend(extra_passes)
    return OverlayPipeline(passes=passes)


def detect(pdf_path: str | Path, page_index: int = 0, cfg: Cfg | None = None) -> tuple[VisibleBoxResult, np.ndarray]:
    """Compatibility function matching the old public API."""
    state = build_pipeline().run(pdf_path, page_index=page_index, cfg=cfg)
    assert state.result is not None and state.rgb is not None
    return state.result, state.rgb


def render_overlay(
    rgb: np.ndarray,
    result: VisibleBoxResult,
    out_path: str | Path,
    *,
    draw_labels: bool = True,
    layers: OverlayLayer | None = None,
) -> Path:
    """Compatibility renderer matching the old public API."""
    return core_render_overlay(rgb, result, out_path, draw_labels=draw_labels, layers=layers)


__all__ = ["OverlayPipeline", "build_pipeline", "detect", "render_overlay"]
