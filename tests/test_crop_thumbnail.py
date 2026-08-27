"""Inline crop thumbnails: bounded, mode-agnostic, guess-free.

A PM culprit card needs pixels it can render. These cover the contract:
a real data URI that decodes, aspect preserved under a 200px max width, every
Pillow mode flattened to RGB, and every failure path returning None WITH a
receipt (never a bloated envelope, never a silent zero).
"""
import base64
import io

from app.core import crop_thumbnail as ct


def _png(size=(640, 480), mode="RGB", color=(200, 30, 30)):
    from PIL import Image
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode(uri):
    from PIL import Image
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


# ── happy path ──────────────────────────────────────────────────────


def test_thumb_is_a_decodable_jpeg_data_uri():
    uri = ct.make_thumb_data_uri(_png())
    assert uri is not None
    img = _decode(uri)
    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_max_width_respected_and_aspect_preserved():
    uri = ct.make_thumb_data_uri(_png(size=(1600, 400)))
    img = _decode(uri)
    assert img.width == 200  # max width, not the 1600 original
    assert img.height == 50  # 4:1 preserved
    assert img.width <= ct._MAX_WIDTH


def test_small_image_is_never_upscaled():
    uri = ct.make_thumb_data_uri(_png(size=(64, 48)))
    img = _decode(uri)
    assert (img.width, img.height) == (64, 48)


def test_thumb_within_size_cap():
    uri = ct.make_thumb_data_uri(_png(size=(1200, 900)))
    assert len(uri) <= ct.max_kb() * 1024


def test_receipted_success_carries_no_error():
    uri, err = ct.make_thumb_data_uri_receipted(_png())
    assert uri and err is None


# ── input modes: everything becomes RGB ─────────────────────────────


def test_rgba_transparency_flattens_onto_white():
    from PIL import Image
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))  # fully transparent
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = _decode(ct.make_thumb_data_uri(buf.getvalue()))
    assert out.mode == "RGB"
    r, g, b = out.convert("RGB").getpixel((10, 10))
    assert r > 240 and g > 240 and b > 240  # white, not a black smear


def test_palette_mode_handled():
    from PIL import Image
    img = Image.new("P", (300, 200))
    img.putpalette([0, 128, 255] * 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = _decode(ct.make_thumb_data_uri(buf.getvalue()))
    assert out.mode == "RGB"


def test_palette_with_transparency_handled():
    from PIL import Image
    img = Image.new("P", (300, 200))
    img.putpalette([0, 128, 255] * 256)
    img.info["transparency"] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG", transparency=0)
    out = _decode(ct.make_thumb_data_uri(buf.getvalue()))
    assert out.mode == "RGB"


def test_cmyk_mode_handled():
    from PIL import Image
    img = Image.new("CMYK", (400, 300), (10, 20, 30, 5))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    out = _decode(ct.make_thumb_data_uri(buf.getvalue()))
    assert out.mode == "RGB"


def test_greyscale_mode_handled():
    out = _decode(ct.make_thumb_data_uri(_png(size=(300, 300), mode="L", color=128)))
    assert out.mode == "RGB"


def test_bilevel_mode_handled():
    out = _decode(ct.make_thumb_data_uri(_png(size=(300, 300), mode="1", color=1)))
    assert out.mode == "RGB"


# ── bounded: retry once, then no thumbnail at all ───────────────────


def test_retry_shrinks_when_first_pass_blows_the_cap():
    """A cap the 200px pass cannot meet falls back to the smaller retry rather
    than giving up outright."""
    from PIL import Image
    import random
    rnd = random.Random(7)
    img = Image.new("RGB", (800, 800))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                 for _ in range(800 * 800)])  # noise: incompressible
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    full, _ = ct.make_thumb_data_uri_receipted(buf.getvalue())
    uri, err = ct.make_thumb_data_uri_receipted(buf.getvalue(), cap_kb=6)
    assert err is None and uri is not None
    assert len(uri) <= 6 * 1024
    assert len(uri) < len(full)  # actually degraded, not the same bytes
    assert _decode(uri).width == ct._RETRY_MAX_WIDTH


def test_uncompressible_input_degrades_to_none_with_receipt():
    """Guess-free: a thumbnail that would bloat the envelope is not shipped."""
    from PIL import Image
    import random
    rnd = random.Random(11)
    img = Image.new("RGB", (900, 900))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                 for _ in range(900 * 900)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri, err = ct.make_thumb_data_uri_receipted(buf.getvalue(), cap_kb=1)
    assert uri is None
    assert err == "too_large"


# ── guess-free failure paths ────────────────────────────────────────


def test_corrupt_bytes_return_none_with_exception_receipt():
    uri, err = ct.make_thumb_data_uri_receipted(b"\x89PNG\r\n" + b"0" * 5000)
    assert uri is None
    assert err and err != "too_large"  # the exception class, never silence
    assert ct.make_thumb_data_uri(b"\x89PNG\r\n" + b"0" * 5000) is None


def test_empty_bytes_are_not_a_failure():
    assert ct.make_thumb_data_uri_receipted(b"") == (None, None)


def test_encoder_failure_is_receipted_not_raised(monkeypatch):
    def _boom(*a, **k):
        raise MemoryError("no room")

    monkeypatch.setattr(ct, "_encode", _boom)
    uri, err = ct.make_thumb_data_uri_receipted(_png())
    assert uri is None
    assert err == "MemoryError"


# ── config ──────────────────────────────────────────────────────────


def test_size_cap_configurable_via_env(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_THUMB_KB", "9")
    assert ct.max_kb() == 9


def test_bad_or_nonpositive_cap_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_THUMB_KB", "not-a-number")
    assert ct.max_kb() == ct._DEFAULT_MAX_KB
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_THUMB_KB", "0")
    assert ct.max_kb() == ct._DEFAULT_MAX_KB


def test_default_cap_is_24kb(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_THUMB_KB", raising=False)
    assert ct.max_kb() == 24
