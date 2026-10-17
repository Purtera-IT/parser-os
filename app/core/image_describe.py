"""What a job image shows that prices the work, as plain fact lines.

A Word document's floor plan or an attached photo was a marker and nothing
more ("[Image awaiting OCR / vision / OLE extraction] media/image1.png"): on
010038 the AP, TV and printer counts live only in two floor-plan figures, and
on 010043 the exterior brick mounts above each door only in three photos. The
grader read them; the brief could not.

One call to the hosted vision model per image (the PDF image pass's own
client and switch), a logo or signature answered as nothing. The lines it
returns are parser lines like any other: the heads decide what they are.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

PROMPT = """This image comes from a field-services job file ({filename}): a floor plan, a site photo, a
diagram or a screenshot. List what it shows that a project manager needs to price the work:
- count each kind of marked device or item, using the image's own legend or labels;
- rooms or areas named, and where the items sit;
- mounting surfaces and site conditions (brick, concrete, drop ceiling, height, outdoor);
- equipment already installed, distances and cable paths written on it.
Report ONLY what the image itself shows. The document text below is there only so you can name marks by its
legend (e.g. "3 Aruba APs" for red hexagons numbered 1-3 when the legend says red hexagons are Aruba APs);
never repeat a fact from that text. Count a mark by its number labels (marks numbered 1, 2, 3 are 3) and
only count what you can see clearly. Room numbers, door numbers and gridlines are not devices; count a
device only when a legend, label or the picture itself says what it is; a room's name is not a device, and
a plan with no marked devices has no device counts. Ignore furniture and decoration. Never guess. If it is a logo,
signature, banner or decoration, answer with no facts.
DOCUMENT TEXT (legend and names only): {context}
Return JSON only: {{"meaningful": true|false, "facts": ["one short sentence per fact"]}}
"""


def enabled() -> bool:
    raw = os.environ.get("SOWSMITH_DOC_IMAGE_VISION", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    try:
        from app.core import pdf_image_vision

        return pdf_image_vision.enabled() and pdf_image_vision._hosted_vision_enabled()
    except Exception:
        return False


MIN_BYTES = int(os.environ.get("SOWSMITH_DOC_IMAGE_MIN_BYTES", "12000"))
MAX_PER_FILE = int(os.environ.get("SOWSMITH_DOC_IMAGE_MAX", "8"))
MAX_FACTS = 10


def describe(image_bytes: bytes, filename: str, context: str = "") -> list[str]:
    """Fact lines for one image, or [] (not meaningful, too small, or any failure)."""
    if not image_bytes or len(image_bytes) < MIN_BYTES:
        return []
    try:
        from app.core import pdf_image_vision

        raw = pdf_image_vision._vlm(image_bytes, PROMPT.format(filename=filename, context=(context or "")[:2500]), model=None, max_tokens=700)
    except Exception as exc:  # pragma: no cover - a failed call costs the image, never the parse
        log.info("image_describe failed: %s", type(exc).__name__)
        return []
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        got: dict[str, Any] = json.loads(m.group(0)) if m else {}
    except Exception:
        return []
    if not got.get("meaningful"):
        return []
    facts = [re.sub(r"\s+", " ", str(f)).strip() for f in got.get("facts") or [] if str(f).strip()]
    legend = bool(_LEGEND.search(context or ""))
    keep = []
    for f in facts:
        if not 6 <= len(f) <= 300:
            continue
        # A picture cannot show that something is absent.
        if re.match(r"(?i)\s*(no|none|there (is|are) no|nothing)\b", f):
            continue
        # Counted marks need the document's legend to say what they are; without
        # one the model counted 12 "black squares" on 010043's plan that are not there.
        if not legend and re.search(r"(?i)\b(marked|symbol|square|circle|triangle|diamond|hexagon|icon)s?\b", f):
            continue
        keep.append(f)
    return keep[:MAX_FACTS]


#: A legend line in the document's text: "Red Hexagons are Aruba APs".
_LEGEND = re.compile(r"(?i)\b(red|blue|green|yellow|orange|black|white|purple|pink|grey|gray)\s+\w+\s+(is|are)\b")


__all__ = ["enabled", "describe", "MAX_PER_FILE"]
