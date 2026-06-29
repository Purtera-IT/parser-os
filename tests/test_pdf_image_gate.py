"""PDF image CPU gate: abstain-first contract (no torch/models needed)."""
from app.core import pdf_image_gate as pig


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_GATE_CPU", raising=False)
    assert pig.enabled() is False
    assert pig.classify("caption: photo of rack", "ocr: APC UPS") is None


def test_abstains_when_model_absent(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_GATE_CPU", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_GATE_DIR", "/nope/missing")
    pig._holder.clear()
    assert pig.classify("caption: install steps", "ocr: set vlan 10") is None
    assert pig.is_ready() is False


def test_gate_feature_text():
    t = pig.gate_feature_text("Upload photo of charger", "VLAN 10 port gi0/1")
    assert "caption:" in t
    assert "ocr:" in t
