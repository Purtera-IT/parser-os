"""Image readers on the go-live branch: a marker says what happened to its image, no filename gate,
a pool for the image loop, OCR for text-bearing crops, no markers for comment shapes, and each
vision / OCR call in the compile trace. Readers extract; the heads decide. Hermetic and made up."""
from __future__ import annotations

import io
import threading
import time
import types
import zipfile

import pytest

from app.core import pdf_image_vision as piv
from app.core import telemetry

from tests.test_pdf_image_vision import _marker


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr(piv, "_vision_reachable", lambda: True)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")


def _markers(tmp_path, n, name="install_guide.pdf"):
    out = []
    for i in range(n):
        m = _marker(tmp_path, region=f"page{i}/image1", saved_name=f"m{i}.png", size=5000 + i)
        m.raw_text = (f"[Image extracted - saved to {m.value['saved_path']} (5,000 bytes), awaiting OCR / vision]"
                      f" page{i}/image1 in {name}.")
        m.source_refs = [types.SimpleNamespace(filename=name)]
        out.append(m)
    return out


def _kinds(monkeypatch, by_region):
    def _gate(*, crop, caption, saved_path, attribution=None):
        kind = by_region[attribution["region_ref"]]
        return kind not in piv._SKIP_KINDS, kind, "vlm_gate", None
    monkeypatch.setattr(piv, "_classify_image", _gate)


def test_a_read_or_skipped_marker_says_what_happened(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_MAX", "2")
    ms = _markers(tmp_path, 3)
    _kinds(monkeypatch, {"page0/image1": "photo", "page1/image1": "logo", "page2/image1": "photo"})
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: '{"description": "a battery charger on a wall", '
                        '"facts": [{"kind": "equipment", "text": "charger mounted on wall"}]}')
    monkeypatch.setattr(piv, "_context_guard", lambda *a, **k: True)
    out = piv.process_image_markers(ms)
    assert out
    assert ms[0].raw_text.startswith("[Image read: 2 facts] page0/image1")
    assert ms[1].raw_text.startswith("[Image skipped: logo] page1/image1")
    assert ms[2].raw_text.startswith("[Image skipped: over the image cap]")
    assert ms[0].value["image_status"] == "read: 2 facts"


def test_too_small_says_so(tmp_path):
    m = _marker(tmp_path, size=10)
    m.raw_text = "[Image extracted - saved to x (10 bytes), awaiting OCR / vision] page2/image7 in a.pdf."
    piv.process_image_markers([m])
    assert m.raw_text.startswith("[Image skipped: too small]")


def test_a_failed_vision_call_leaves_the_marker_awaiting(monkeypatch, tmp_path):
    ms = _markers(tmp_path, 1)
    _kinds(monkeypatch, {"page0/image1": "photo"})
    monkeypatch.setattr(piv, "_vlm_inner", lambda *a, **k: "")
    piv.process_image_markers(ms)
    assert "awaiting OCR" in ms[0].raw_text


def test_a_pdf_without_the_extension_is_read(monkeypatch, tmp_path):
    ms = _markers(tmp_path, 1, name="TEK_Store Installs_1.2")
    seen = []
    _kinds(monkeypatch, {"page0/image1": "logo"})
    monkeypatch.setattr(piv, "_stamp_skip_verdict",
                        lambda marker, kind, **k: seen.append(kind) or marker.value.update(gate_verdict={"kind": kind}))
    piv.process_image_markers(ms)
    assert seen == ["logo"]


def test_the_image_loop_runs_on_a_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(piv, "_hosted_vision_enabled", lambda: True)
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_MAX", "120")
    ms = _markers(tmp_path, 24)
    live, peak, lock = [0], [0], threading.Lock()

    def _one(**kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.1)  # one image's gate + describe, scaled down
        with lock:
            live[0] -= 1
        return []

    monkeypatch.setattr(piv, "_process_one", _one)
    t = time.monotonic()
    piv.process_image_markers(ms)
    took = time.monotonic() - t
    assert peak[0] == 6
    assert took < 0.1 * 24 / 3  # sequential would be 2.4 s; the pool takes ~0.5 s


def test_text_bearing_crops_are_read_by_ocr_not_vision(monkeypatch, tmp_path):
    ms = _markers(tmp_path, 1)
    _kinds(monkeypatch, {"page0/image1": "screenshot"})
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "Switch port 12\nVLAN 30 voice\n\nVLAN 30 voice")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: pytest.fail("a text-bearing crop must not call vision"))
    out = piv.process_image_markers(ms)
    assert [a.raw_text for a in out] == ["Switch port 12", "VLAN 30 voice"]
    assert {a.value.get("fact_kind") for a in out} == {"image_ocr_line"}
    assert ms[0].raw_text.startswith("[Image read: 2 facts]")


def test_a_screenshot_with_no_ocr_still_goes_to_vision(monkeypatch, tmp_path):
    ms = _markers(tmp_path, 1)
    _kinds(monkeypatch, {"page0/image1": "screenshot"})
    called = []
    monkeypatch.setattr(piv, "_transcribe", lambda **k: called.append(1) or [])
    piv.process_image_markers(ms)
    assert called == [1]


def test_vision_and_ocr_calls_are_in_the_compile_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(piv, "_hosted_vision_enabled", lambda: False)
    monkeypatch.setattr(piv, "_use_ollama_for_pdf_images", lambda: True)
    monkeypatch.setattr(piv, "_ollama_vision_direct", lambda *a, **k: "")
    telemetry.CALLS_BY_STAGE.clear()
    monkeypatch.setattr(telemetry, "CURRENT_STAGE", "pdf_image_vision")
    piv._vlm(b"x", "p", model=None, max_tokens=10)
    piv._vlm(b"x", "p", model=None, max_tokens=10)
    row = telemetry.CALLS_BY_STAGE["pdf_image_vision"]
    assert row["vision_calls"] == 2 and row["vision_empty"] == 2 and "vision_s" in row


def test_xlsx_comment_shapes_and_rels_get_no_marker(tmp_path):
    from app.parsers.binary_markers import emit_zip_binary_markers
    from app.core.schemas import ArtifactType

    p = tmp_path / "book.xlsx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("xl/drawings/vmlDrawing1.vml", "<xml/>")
        z.writestr("xl/drawings/_rels/drawing1.xml.rels", "<xml/>")
        z.writestr("xl/drawings/drawing1.xml", "<xml/>")
        z.writestr("xl/media/image1.png", b"png")
    got = emit_zip_binary_markers(path=p, project_id="d", artifact_id="a", filename="book.xlsx",
                                  artifact_type=ArtifactType.xlsx, parser_version="t")
    refs = sorted(a.value["region_ref"] for a in got)
    assert refs == ["drawings/drawing1.xml", "media/image1.png"]


def test_docx_marker_says_what_describe_did(monkeypatch):
    from app.core import image_describe
    from app.parsers.docx_parser import DocxParser

    monkeypatch.setattr(image_describe, "enabled", lambda: True)
    monkeypatch.setattr(image_describe, "describe_outcome",
                        lambda data, fn, ctx: (["Three access points on the plan"], "read: 1 fact"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p>Plan</w:p></w:document>")
        z.writestr("word/media/image1.png", b"png" * 10)
    path = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "s.docx"
    path.write_bytes(buf.getvalue())
    atoms = DocxParser()._emit_embedded_media_markers(project_id="d", artifact_id="a", filename="s.docx", path=path)
    assert atoms[0].raw_text.startswith("[Image read: 1 fact] media/image1.png")
    assert atoms[1].raw_text == "Three access points on the plan"


def test_describe_outcome_names_the_skip(monkeypatch):
    from app.core import image_describe

    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: '{"meaningful": false, "kind": "logo"}')
    assert image_describe.describe_outcome(b"x" * 20000, "s.docx") == ([], "skipped: logo")
    assert image_describe.describe_outcome(b"x", "s.docx") == ([], "skipped: too small")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: "")
    assert image_describe.describe_outcome(b"x" * 20000, "s.docx") == ([], None)
