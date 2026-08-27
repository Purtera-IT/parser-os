"""Tiny inline thumbnails for disputed image crops — pixels the PM can SEE.

Why this exists: a disputed-skip PM culprit card carries ``crop_ref`` (a blob
path) and nothing that renders. No frontend route serves blob images by path,
so the PM has to go looking — which means they don't. Embedding a downscaled
JPEG as a ``data:`` URI carries the actual pixels through the same channel the
verdict already travels (envelope -> core card -> UI) with zero new
infrastructure.

Contract (matches the rest of the image-gate loop):
  * GUESS-FREE. Any failure — unreadable bytes, a mode Pillow can't convert, a
    decompression bomb, an encoder blowing up — returns ``None``. A missing
    thumbnail is a fine outcome; a wrong or half-written one is not.
  * BOUNDED. A thumbnail that would bloat the envelope is worse than no
    thumbnail. One retry at smaller dimensions/quality, then give up
    (``too_large``). The caller stamps the receipt.
  * RECEIPTED. :func:`make_thumb_data_uri_receipted` returns *why* it produced
    nothing so the caller can stamp a liveness receipt (doctrine 2 in
    ``docs/IMAGE_GATE_LOOP.md``: a silent zero and a real zero must never look
    the same). :func:`make_thumb_data_uri` is the thin value-only wrapper.

Sizing: max width ~200px preserving aspect ratio, JPEG quality ~70, which lands
a typical crop around 6-10KB of base64. The hard cap is
``SOWSMITH_PDF_IMAGE_THUMB_KB`` (default 24KB of data-URI text); the per-compile
count cap lives with the caller (``pdf_image_vision``).
"""
from __future__ import annotations

import base64
import io
import os

# First attempt: readable on a card without being heavy.
_MAX_WIDTH = 200
_QUALITY = 70
# Retry: smaller and lossier, one shot only. Still over cap -> no thumbnail.
_RETRY_MAX_WIDTH = 120
_RETRY_QUALITY = 45
# Default data-URI budget per thumbnail, in KB of the final string.
_DEFAULT_MAX_KB = 24

_PREFIX = "data:image/jpeg;base64,"


def max_kb() -> int:
    """Per-thumbnail size cap in KB of the final data URI. Bad values fall back
    to the default rather than disabling the cap (guess-free)."""
    try:
        val = int(os.environ.get("SOWSMITH_PDF_IMAGE_THUMB_KB", str(_DEFAULT_MAX_KB)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_KB
    return val if val > 0 else _DEFAULT_MAX_KB


def _flatten_to_rgb(img):
    """RGB pixels for any input mode. Transparency (RGBA/LA/P-with-alpha) is
    composited onto WHITE — a card renders on a light surface, and JPEG has no
    alpha channel, so the alternative is black smears where the crop was clear.
    """
    from PIL import Image

    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode in ("P", "PA") and "transparency" in img.info
    )
    if has_alpha:
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        # P (palette), L / 1 (grey / bilevel), CMYK, I;16, YCbCr ...
        return img.convert("RGB")
    return img


def _encode(img, *, max_width: int, quality: int) -> str:
    """Downscale (aspect preserved, never upscale) and JPEG-encode as a data URI."""
    from PIL import Image

    if img.width > max_width:
        height = max(1, round(img.height * max_width / img.width))
        img = img.resize((max_width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return _PREFIX + base64.b64encode(buf.getvalue()).decode("ascii")


def make_thumb_data_uri_receipted(
    crop_bytes: bytes, *, cap_kb: int | None = None,
) -> tuple[str | None, str | None]:
    """``(data_uri, error)`` for one crop — exactly one side is ever set.

      * ``(uri, None)``   — a thumbnail within the size cap.
      * ``(None, name)``  — the exception class that killed it, or
        ``'too_large'`` when even the retry blew the cap.
      * ``(None, None)``  — nothing to work with (empty bytes); not a failure.
    """
    if not crop_bytes:
        return None, None
    cap = (cap_kb if cap_kb is not None else max_kb()) * 1024
    try:
        from PIL import Image

        with Image.open(io.BytesIO(crop_bytes)) as opened:
            opened.load()
            rgb = _flatten_to_rgb(opened)
        uri = _encode(rgb, max_width=_MAX_WIDTH, quality=_QUALITY)
        if len(uri) <= cap:
            return uri, None
        uri = _encode(rgb, max_width=_RETRY_MAX_WIDTH, quality=_RETRY_QUALITY)
        if len(uri) <= cap:
            return uri, None
        # Guess-free: no thumbnail beats a bloated envelope.
        return None, "too_large"
    except Exception as exc:
        return None, type(exc).__name__


def make_thumb_data_uri(crop_bytes: bytes, *, cap_kb: int | None = None) -> str | None:
    """A ``data:image/jpeg;base64,...`` thumbnail for ``crop_bytes``, or ``None``
    when one cannot be produced within the size cap. Never raises."""
    return make_thumb_data_uri_receipted(crop_bytes, cap_kb=cap_kb)[0]
