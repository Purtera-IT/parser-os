"""Local universal symbol detector loader — class-agnostic YOLO that finds symbol
boxes on any schematic. Meaning is assigned per-document by LegendIndex, so it's
universal. Loads the RunPod-trained weights from SOWSMITH_SYMBOL_DETECTOR; returns
[] (caller falls back to region_proposals/objectness) when unavailable."""
from __future__ import annotations
import io
import os
from dataclasses import dataclass

_MODEL = None
_TRIED = False


@dataclass
class DetectedBox:
    bbox_px: tuple[int, int, int, int]
    score: float


def _load():
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    _TRIED = True
    path = os.environ.get("SOWSMITH_SYMBOL_DETECTOR")
    if not path or not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        _MODEL = YOLO(path)
    except Exception:
        _MODEL = None
    return _MODEL


def available() -> bool:
    return _load() is not None


def detect(image, *, conf: float = 0.25, imgsz: int = 1280) -> list[DetectedBox]:
    """Detect symbol boxes on a PIL image. [] if the detector isn't installed."""
    m = _load()
    if m is None:
        return []
    try:
        res = m.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
        out = []
        for b in res.boxes:
            x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
            out.append(DetectedBox((x0, y0, x1, y1), float(b.conf[0])))
        return out
    except Exception:
        return []
