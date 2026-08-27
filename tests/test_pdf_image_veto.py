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


# ── soft band: [soft_bar, hard_bar) — harvest zone, never a hard veto ─


def _model_at(monkeypatch, prob):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_CONF", "0.88")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_SOFT_CONF", "0.70")
    monkeypatch.setattr(piveto, "_load", lambda: ("fake", "tok", "torch", {0: "meaningful"}))
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: prob)


CAP, OCR = "caption: Figure B-2. Patch Panel", "ocr: 18 Total Data Outlets"


def test_band_boundaries_exact(monkeypatch):
    """0.69 -> nothing; 0.70 -> soft; 0.879 -> soft; 0.88 -> hard (soft None)."""
    _model_at(monkeypatch, 0.69)
    assert piveto.veto(CAP, OCR) is None
    assert piveto.soft_veto(CAP, OCR) is None

    _model_at(monkeypatch, 0.70)
    assert piveto.veto(CAP, OCR) is None
    assert piveto.soft_veto(CAP, OCR) == 0.70   # inclusive lower bound

    _model_at(monkeypatch, 0.879)
    assert piveto.veto(CAP, OCR) is None
    assert piveto.soft_veto(CAP, OCR) == 0.879  # just under the hard bar

    _model_at(monkeypatch, 0.88)
    assert piveto.veto(CAP, OCR) == 0.88        # hard fires
    assert piveto.soft_veto(CAP, OCR) is None   # exclusive upper bound


def test_soft_disabled_when_veto_off(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_VETO", raising=False)
    monkeypatch.setattr(piveto, "_load", lambda: ("fake", "tok", "torch", {0: "meaningful"}))
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: 0.75)
    assert piveto.soft_veto(CAP, OCR) is None


def test_soft_abstains_on_degenerate_feature(monkeypatch):
    _model_at(monkeypatch, 0.75)
    assert piveto.soft_veto("", "") is None


def test_soft_abstains_when_model_absent(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_DIR", "/nope/missing")
    _reset()
    assert piveto.soft_veto(CAP, OCR) is None
    _reset()


def test_soft_abstains_on_any_failure(monkeypatch):
    """Guess-free: an exploding scorer means NO soft veto, never a guess."""
    _model_at(monkeypatch, 0.75)

    def _boom(*a, **k):
        raise RuntimeError("scorer died")

    monkeypatch.setattr(piveto, "_meaningful_prob", _boom)
    assert piveto.soft_veto(CAP, OCR) is None
    assert piveto.veto(CAP, OCR) is None


def test_soft_bar_env_override_and_bad_value(monkeypatch):
    _model_at(monkeypatch, 0.65)
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_SOFT_CONF", "0.60")
    assert piveto.soft_veto(CAP, OCR) == 0.65   # custom band [0.60, 0.88)
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_SOFT_CONF", "banana")
    assert piveto.soft_veto(CAP, OCR) is None   # falls back to default 0.70
