"""Tests for the optional OCR fallback pass.

OCR engines (EasyOCR, Tesseract) are NOT installed by default — they
pull >500MB of model files / native deps. The OCR module must handle
their absence gracefully: returning ``[]`` + a warning rather than
crashing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.takeoff.legend_extractor import load_default_legend_rules
from app.takeoff.ocr_signals import (
    OCREngineHandle,
    ocr_candidates_for_page,
)
from app.takeoff.schemas import SheetRecord


def test_load_engine_returns_none_when_neither_easyocr_nor_pytesseract() -> None:
    """If neither engine imports cleanly, the loader returns a handle
    named 'none'. This must NOT raise."""
    from app.takeoff.ocr_signals import _load_engine

    handle = _load_engine()
    # In CI we expect 'none'; in dev one of the engines may be installed.
    assert handle.name in {"none", "easyocr", "pytesseract"}


def test_ocr_candidates_for_page_no_engine_returns_warning() -> None:
    sheet = SheetRecord(page_index=0, page_type="floor_plan", in_scope=True)
    rules = load_default_legend_rules()
    handle = OCREngineHandle(name="none", detector=None)
    cands, reason = ocr_candidates_for_page(
        page=None, sheet=sheet, legend_rules=rules, engine=handle,
    )
    assert cands == []
    assert reason and "ocr_engine_unavailable" in reason


_MARRIOTT_PDF = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)


def _easyocr_installed() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _MARRIOTT_PDF.exists()
    or not os.environ.get("RUN_SLOW_TESTS")
    or _easyocr_installed(),
    reason="Marriott PDF + RUN_SLOW_TESTS=1 required, and easyocr must NOT be installed",
)
def test_pipeline_opt_in_env_flag_does_not_crash_without_engine() -> None:
    """Setting the OCR env flag with no engine installed should produce
    a warning but never crash the parse."""
    from app.takeoff.pipeline import build_low_voltage_takeoff

    prev = os.environ.get("PARSER_OS_ENABLE_OCR_SIGNALS")
    os.environ["PARSER_OS_ENABLE_OCR_SIGNALS"] = "1"
    try:
        takeoff = build_low_voltage_takeoff(_MARRIOTT_PDF)
    finally:
        if prev is None:
            os.environ.pop("PARSER_OS_ENABLE_OCR_SIGNALS", None)
        else:
            os.environ["PARSER_OS_ENABLE_OCR_SIGNALS"] = prev
    # Warning should be present.
    warn_blob = "\n".join(takeoff.warnings)
    assert "ocr_signals_skipped_no_engine" in warn_blob
    # WN invariant must still hold.
    wn = takeoff.summary.get("wireless_node_outlet") or {}
    assert wn.get("extended_count") == 335
