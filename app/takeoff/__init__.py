"""Low-voltage construction takeoff pipeline.

This package adds a deterministic takeoff layer alongside the existing
``orbitbrief_pdf`` structured parser. It extracts low-voltage device
instances (Wireless Nodes, POS, TVs, card readers, door alarms, etc.),
classifies sheets, parses zone notes, applies floor multipliers, and
emits a stable ``TakeoffDocument`` with quote lines suitable for
downstream pricing.

The public surface is intentionally small:

* :func:`app.takeoff.pipeline.build_low_voltage_takeoff` — the one-shot
  function that consumes a PDF path and returns a TakeoffDocument.
* :mod:`app.takeoff.schemas` — Pydantic models for the document.
* :mod:`app.takeoff.exports` — JSON / markdown / atom emitters.

Nothing in this package opens the network, calls an LLM, or runs OCR.
PyMuPDF native text is the only source of truth.
"""
from __future__ import annotations

from app.takeoff.schemas import (
    BBox,
    DeviceInstance,
    LegendRule,
    QuoteLine,
    SheetRecord,
    SymbolCandidate,
    TakeoffDocument,
)

__all__ = [
    "BBox",
    "DeviceInstance",
    "LegendRule",
    "QuoteLine",
    "SheetRecord",
    "SymbolCandidate",
    "TakeoffDocument",
]
