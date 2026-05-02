"""Core config/models plus lazy pipeline exports.

The pipeline imports pass modules, and pass modules import core config/models.  A
lazy export keeps `from ...core import detect` working without reintroducing a
package import cycle.
"""
from .config import Cfg, PassRuntimeConfig, PipelineConfig
from .models import Rect, VisibleBox, VisibleBoxResult

_PIPELINE_EXPORTS = {"OverlayPipeline", "build_pipeline", "detect", "render_overlay"}


def __getattr__(name: str):
    if name in _PIPELINE_EXPORTS:
        from . import pipeline as _pipeline
        return getattr(_pipeline, name)
    raise AttributeError(name)


__all__ = [
    "Cfg",
    "PassRuntimeConfig",
    "PipelineConfig",
    "Rect",
    "VisibleBox",
    "VisibleBoxResult",
    "OverlayPipeline",
    "build_pipeline",
    "detect",
    "render_overlay",
]
