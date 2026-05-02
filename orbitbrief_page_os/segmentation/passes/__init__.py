"""Overlay pipeline passes."""
from .base import OverlayPass, PageContext, PassInfo, PipelineState
from .visible_boxes import VisibleBoxPass
from .title_block_detection import TitleBlockDetectionPass, detect_title_block
from .semantic_cleanup import SemanticCleanupPass, cleanup_boxes
from .grid_cell_completion import GridCellCompletionPass, complete_grid_cells
from .cellular_title import CellularTitlePass, detect_cellular_titles
from .raster_line_repair import RasterLineRepairPass, extract_sidebar_line_repairs
from .placeholders import NoOpSemanticPass, build_future_noops
from .multicol_contact import MultiColContactPass

__all__ = [
    "OverlayPass",
    "PageContext",
    "PassInfo",
    "PipelineState",
    "VisibleBoxPass",
    "TitleBlockDetectionPass",
    "detect_title_block",
    "SemanticCleanupPass",
    "cleanup_boxes",
    "GridCellCompletionPass",
    "complete_grid_cells",
    "CellularTitlePass",
    "detect_cellular_titles",
    "RasterLineRepairPass",
    "extract_sidebar_line_repairs",
    "NoOpSemanticPass",
    "build_future_noops",
    "MultiColContactPass",
]
