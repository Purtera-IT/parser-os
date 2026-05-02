"""No-op semantic pass slots for legend v2.

Adding a real semantic concern should mean replacing exactly one class here (or
adding a sibling module), assigning its `OverlayLayer` bit, extending the legend
registry, and adding golden tests.  These no-ops make the intended seams visible
without changing today's overlays.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import PageContext, PassInfo, PipelineState


@dataclass
class NoOpSemanticPass:
    info: PassInfo

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        state.artifacts.setdefault("inactive_passes", []).append(self.info.name)
        return state


FUTURE_PASS_INFOS: list[PassInfo] = [
    PassInfo(
        name="symbol_equipment_tags",
        stage="semantic",
        layer_flag="SYMBOL_TAGS",
        order=300,
        description="Detect valves, motors, FA devices, equipment IDs, and tag glyphs as first-class regions.",
    ),
    PassInfo(
        name="row_column_grouping",
        stage="semantic",
        layer_flag="ROW_COL_GROUPS",
        order=320,
        description="Emit visual row/column ownership spines, bands, or brackets tying body rows to headers.",
    ),
    PassInfo(
        name="revision_cloud_notes",
        stage="semantic",
        layer_flag="REVISION_CALLOUTS",
        order=340,
        description="Separate revision clouds, delta triangles, note callouts, and issue grids from body schedules.",
    ),
    PassInfo(
        name="legend_block_classifier",
        stage="semantic",
        layer_flag="LEGEND_BLOCKS",
        order=360,
        description="Classify graphic/symbol legend blocks separately from schedule tables and drawing grids.",
    ),
    PassInfo(
        name="cross_reference_bubbles",
        stage="semantic",
        layer_flag="CROSS_REFS",
        order=380,
        description="Detect section/detail/sheet reference bubbles and connect them to contained text spans.",
    ),
    PassInfo(
        name="multi_scale_bands",
        stage="semantic",
        layer_flag="MULTI_SCALE_BANDS",
        order=400,
        description="Make title strip, data body, margin mini-table, and page chrome scale classes explicit.",
    ),
]


def build_future_noops() -> list[NoOpSemanticPass]:
    return [NoOpSemanticPass(info) for info in FUTURE_PASS_INFOS]
