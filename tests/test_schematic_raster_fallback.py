"""PR8 — raster fallback + optional classifier seam tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
np = pytest.importorskip("numpy")

from orbitbrief_page_os.segmentation.schematic.classifier import (
    ClassifierConfig,
    from_config,
    hash_file,
)
from orbitbrief_page_os.segmentation.schematic.ocr import (
    is_available as ocr_is_available,
    ocr_words,
    status_warning,
)
from orbitbrief_page_os.segmentation.schematic.raster import (
    deskew_grayscale,
    is_text_poor_page,
    render_page_to_ndarray,
)


def _make_text_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(15):
        page.insert_text((72, 80 + 20 * i), f"line {i}: lorem ipsum dolor sit amet" * 2, fontsize=10)
    doc.save(str(path))
    doc.close()


def _make_image_only_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Page has only a single rectangle drawn — no text layer at all.
    page.draw_rect(fitz.Rect(100, 100, 500, 700), color=(0, 0, 0), width=2)
    doc.save(str(path))
    doc.close()


def test_is_text_poor_page_detects_image_only(tmp_path: Path) -> None:
    img_pdf = tmp_path / "image_only.pdf"
    _make_image_only_pdf(img_pdf)
    doc = fitz.open(str(img_pdf))
    try:
        assert is_text_poor_page(doc.load_page(0)) is True
    finally:
        doc.close()


def test_is_text_poor_page_negative_on_dense_text(tmp_path: Path) -> None:
    text_pdf = tmp_path / "text.pdf"
    _make_text_pdf(text_pdf)
    doc = fitz.open(str(text_pdf))
    try:
        assert is_text_poor_page(doc.load_page(0)) is False
    finally:
        doc.close()


def test_render_page_to_ndarray_is_deterministic(tmp_path: Path) -> None:
    pdf = tmp_path / "text.pdf"
    _make_text_pdf(pdf)
    doc_a = fitz.open(str(pdf))
    doc_b = fitz.open(str(pdf))
    try:
        a = render_page_to_ndarray(doc_a.load_page(0))
        b = render_page_to_ndarray(doc_b.load_page(0))
    finally:
        doc_a.close()
        doc_b.close()
    assert a is not None and b is not None
    assert a.dtype == np.uint8
    assert (a == b).all()


def test_deskew_grayscale_handles_pre_aligned_image(tmp_path: Path) -> None:
    pdf = tmp_path / "image_only.pdf"
    _make_image_only_pdf(pdf)
    doc = fitz.open(str(pdf))
    try:
        arr = render_page_to_ndarray(doc.load_page(0))
    finally:
        doc.close()
    deskewed = deskew_grayscale(arr)
    assert deskewed is not None
    assert deskewed.shape == arr.shape


def test_ocr_words_returns_empty_when_tesseract_missing() -> None:
    # We don't require Tesseract for CI — when it's missing the
    # adapter must return [] rather than raising.
    arr = np.zeros((100, 100), dtype=np.uint8)
    words = ocr_words(arr)
    if ocr_is_available():
        # Tesseract installed: an empty image should yield no words.
        assert isinstance(words, list)
    else:
        assert words == []


def test_status_warning_has_correct_shape() -> None:
    w = status_warning(page_index=4, sheet_number="T0.01")
    loc = w.locator_dict()
    assert loc["warning_type"] == "ocr_unavailable"
    assert loc["page"] == 4


def test_classifier_missing_file_returns_none(tmp_path: Path) -> None:
    cfg = ClassifierConfig(
        model_path=tmp_path / "does_not_exist.onnx",
        expected_sha256="0" * 64,
        input_shape=(1, 3, 64, 64),
    )
    clf, reason = from_config(cfg)
    assert clf is None
    assert "not found" in reason


def test_classifier_hash_mismatch_returns_none(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"definitely-not-onnx")
    cfg = ClassifierConfig(
        model_path=model,
        expected_sha256="0" * 64,
        input_shape=(1, 3, 64, 64),
    )
    clf, reason = from_config(cfg)
    assert clf is None
    assert "hash mismatch" in reason


def test_hash_file_matches_known_sha256(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    payload = b"hello, schematic world"
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert hash_file(f) == expected
