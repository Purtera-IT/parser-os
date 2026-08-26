"""PDF image skip-veto: abstain-first contract (mocked model, no network)."""
from app.core import pdf_image_veto as piveto


def _reset():
    piveto._holder.clear()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_VETO", raising=False)
    _reset()
    assert piveto.enabled() is False
    assert piveto.veto("caption: rack elevation", "ocr: APC UPS") is None


def test_abstains_when_model_absent(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_DIR", "/nope/missing")
    _reset()
    assert piveto.veto("caption: install steps", "ocr: set vlan 10") is None
    assert piveto.is_ready() is False


def test_abstains_on_degenerate_feature(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    # Model would load if present — empty feature must still abstain first.
    monkeypatch.setattr(piveto, "_load", lambda: ("fake", "tok", "torch", {0: "meaningful"}))
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: 0.99)
    assert piveto.veto("", "") is None


def test_abstains_below_confidence_bar(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_CONF", "0.88")
    monkeypatch.setattr(piveto, "_load", lambda: ("fake", "tok", "torch", {0: "meaningful"}))
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: 0.50)
    assert piveto.veto("caption: Figure B-2", "ocr: patch panel") is None


def test_fires_when_confident_meaningful(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_CONF", "0.88")
    monkeypatch.setattr(piveto, "_load", lambda: ("fake", "tok", "torch", {0: "meaningful"}))
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: 0.95)
    prob = piveto.veto("caption: Figure B-2. Patch Panel", "ocr: 18 Total Data Outlets")
    assert prob == 0.95
