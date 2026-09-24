"""What colour is a label actually printed in, and what does the legend say
that colour means.

A vendor drawing carries its bill of materials as a COLOUR KEY: "orange means
Huzzard supplied, purple means Installer supplied", and then eighteen labels in
one colour or the other. That key is the most commercially loaded thing on the
sheet -- it assigns every part to a company -- and it is the one thing a vision
model is bad at. Asked to read deal 010288's drawing, gpt-4.1-mini transcribed
every word correctly and put "Cat5e/6 cable (up to 6')" in the vendor's colour
when it is printed in the installer's. That single flip moves a cable from our
bill to the customer's.

Colour is not a judgment call, it is a measurement. So the model is never asked
for it. OCR gives each label a polygon, this samples the ink inside it, and the
legend entries -- printed in their own colours, on the same sheet -- supply the
reference. Nothing here hard-codes a colour name: a drawing keyed in green and
red calibrates itself exactly the same way.
"""
from __future__ import annotations

import colorsys
import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

#: Ink, not paper and not anti-aliasing. Text occupies a minority of the pixels
#: inside its own bounding box, so the rest of the box has to be excluded or
#: every label measures as white.
MIN_SATURATION = 0.35
MIN_VALUE = 0.25
#: Below this the sample is a few stray pixels and says nothing.
MIN_INK_PIXELS = 8
#: Hues are bucketed at 10 degrees, so this allows a bucket and a half of
#: measurement noise and no more. It is deliberately tight: on 010288 the two
#: legend colours measure 30 and 240, and the vendor's own brand blue -- the
#: logo and the title block -- measures 200 to 210. Thirty degrees is the whole
#: margin between "the installer buys this" and "this is the sheet's title", so
#: a generous tolerance would put a title block on somebody's bill of
#: materials. Anything outside lands unassigned, which is a real answer: the
#: sheet does not say who supplies it.
HUE_TOLERANCE_DEG = 15

_LEGEND_HINT = re.compile(
    r"\b(supplied|provided|furnished|by others|existing|new|proposed|typ\.?)\b", re.I)


def _hue_histogram(image: Any, polygon: list[float]) -> Counter:
    xs, ys = polygon[0::2], polygon[1::2]
    box = (max(0, int(min(xs))), max(0, int(min(ys))),
           int(max(xs)) + 1, int(max(ys)) + 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return Counter()
    votes: Counter = Counter()
    for pixel in image.crop(box).getdata():
        r, g, b = pixel[0] / 255, pixel[1] / 255, pixel[2] / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s < MIN_SATURATION or v < MIN_VALUE:
            continue
        votes[round(h * 36) * 10 % 360] += 1
    return votes


def ink_hue(image: Any, polygon: list[float]) -> int | None:
    """Dominant ink hue in degrees, or None when the label is black or grey.

    Black and grey are not legend colours -- on this sheet they are the door,
    the hardware bodies and the small print -- so a label with no saturated ink
    is correctly unassignable rather than wrongly assigned.
    """
    votes = _hue_histogram(image, polygon)
    if not votes:
        return None
    hue, n = votes.most_common(1)[0]
    return hue if n >= MIN_INK_PIXELS else None


def _circular_delta(a: int, b: int) -> int:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def legend_reference_hues(image: Any, lines: list[dict[str, Any]]) -> dict[int, str]:
    """{hue: what it means}, read off the legend in the drawing's own ink.

    A legend entry is printed in the colour it defines -- that is what makes it
    a legend -- so the sheet calibrates itself. Two entries that measure the
    same hue are both dropped: an ambiguous key is worse than no key, because
    it would assign parts to a company on a coin flip.
    """
    found: dict[int, str] = {}
    clashes: set[int] = set()
    for line in lines:
        content = str(line.get("content") or "").strip()
        if not _LEGEND_HINT.search(content):
            continue
        hue = ink_hue(image, line.get("polygon") or [])
        if hue is None:
            continue
        meaning = content.lstrip("-–—• ").strip()
        if hue in found and found[hue] != meaning:
            clashes.add(hue)
            continue
        found[hue] = meaning
    for hue in clashes:
        logger.info("linked_picture_ink: two legend entries share hue %s; dropping it", hue)
        found.pop(hue, None)
    return found


def classify(image: Any, polygon: list[float],
             references: dict[int, str]) -> tuple[int, str] | None:
    """(hue, meaning) for a label, or None when it matches no legend entry."""
    hue = ink_hue(image, polygon)
    if hue is None or not references:
        return None
    best = min(references, key=lambda ref: _circular_delta(hue, ref))
    if _circular_delta(hue, best) > HUE_TOLERANCE_DEG:
        return None
    return hue, references[best]


__all__ = ["ink_hue", "legend_reference_hues", "classify", "HUE_TOLERANCE_DEG"]
