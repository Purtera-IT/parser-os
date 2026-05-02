"""Compatibility pass: v1 raster + PyMuPDF detector as one isolated stage.

This is deliberately the only pass that calls the core implementation.  It
locks current behavior while the internal pieces are migrated one by one into
smaller modules.  Regression tests should stay green after each extraction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .._core_detect_standalone import detect as core_detect
from .base import PageContext, PassInfo, PipelineState


@dataclass
class VisibleBoxPass:
    info: PassInfo = PassInfo(
        name="visible_boxes",
        stage="detect+synthesize",
        layer_flag="ALL",
        order=100,
        description=(
            "Rasterize with pypdfium2, OpenCV morphology, hierarchy/coloring, "
            "plus optional PyMuPDF text-section and mini-table synthesis."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        result, rgb = core_detect(ctx.pdf_path, page_index=ctx.page_index, cfg=ctx.cfg)
        state.result = result
        state.rgb = rgb
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        return state
