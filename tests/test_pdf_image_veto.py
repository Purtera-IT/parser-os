"""PDF image skip-veto: abstain-first contract + liveness receipt
(mocked model, no network)."""
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


# ── liveness receipt: check() reports WHY it abstained ──────────────
#
# veto()/soft_veto() collapse every abstain to None, which is exactly the
# silent zero the doctrine forbids: "the head judged this a real skip" and
# "there is no head" are the same None. check() is the one call that keeps
# them apart — and the one forward pass both bands read.


def test_check_reports_loaded_and_the_prob_it_abstained_on(monkeypatch):
    """The receipt that was missing: model ran, said 0.04, declined to veto."""
    _model_at(monkeypatch, 0.04)
    assert piveto.check(CAP, OCR) == {
        "model": "loaded", "meaningful_prob": 0.04, "band": None,
    }
    assert piveto.veto(CAP, OCR) is None       # band decision unchanged
    assert piveto.soft_veto(CAP, OCR) is None


def test_check_reports_absent_model(monkeypatch):
    """Same abstain, different receipt: the head cannot fire at all."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_DIR", "/nope/missing")
    _reset()
    assert piveto.check(CAP, OCR) == {
        "model": "absent", "meaningful_prob": None, "band": None,
        "reason": "model_absent",
    }
    assert piveto.model_state() == "absent"
    assert piveto.is_ready() is False
    _reset()


def test_check_reports_bands(monkeypatch):
    """Both bands come out of ONE consultation, receipt attached."""
    _model_at(monkeypatch, 0.93)
    assert piveto.check(CAP, OCR) == {
        "model": "loaded", "meaningful_prob": 0.93, "band": "hard",
    }
    _model_at(monkeypatch, 0.75)
    assert piveto.check(CAP, OCR) == {
        "model": "loaded", "meaningful_prob": 0.75, "band": "soft",
    }


def test_check_when_disabled_claims_no_state(monkeypatch):
    """OFF means we never looked — guess-free, so 'unknown', not 'absent'."""
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_VETO", raising=False)

    def _boom():
        raise AssertionError("must not load a model when the veto is off")

    monkeypatch.setattr(piveto, "_load", _boom)
    assert piveto.check(CAP, OCR) == {
        "model": "unknown", "meaningful_prob": None, "band": None,
        "reason": "disabled",
    }


def test_check_degenerate_feature_still_reports_the_head(monkeypatch):
    """Nothing to judge, but the envelope still learns whether a head exists."""
    _model_at(monkeypatch, 0.99)
    rec = piveto.check("", "")
    assert rec["meaningful_prob"] is None and rec["band"] is None
    assert rec["reason"] == "no_context"
    assert rec["model"] == "loaded"


def test_check_reports_mis_shipped_checkpoint(monkeypatch):
    """Loaded but with no 'meaningful' label: it can never fire, and that is
    visible instead of looking like a confident agreement."""
    _model_at(monkeypatch, 0.99)
    monkeypatch.setattr(piveto, "_meaningful_prob", lambda *a, **k: None)
    assert piveto.check(CAP, OCR) == {
        "model": "loaded", "meaningful_prob": None, "band": None,
        "reason": "no_meaningful_label",
    }


def test_check_reports_inference_failure(monkeypatch):
    """Guess-free: a scorer that dies records the failure, never a verdict."""
    _model_at(monkeypatch, 0.75)

    def _boom(*a, **k):
        raise RuntimeError("scorer died")

    monkeypatch.setattr(piveto, "_meaningful_prob", _boom)
    assert piveto.check(CAP, OCR) == {
        "model": "loaded", "meaningful_prob": None, "band": None,
        "reason": "infer_failed",
    }


def test_check_reports_unknown_when_the_load_blows_up(monkeypatch, tmp_path):
    """A checkpoint dir that exists but will not load is NOT 'absent' — we
    cannot tell what is in there, so the receipt says unknown."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO_DIR", str(tmp_path))  # empty dir
    _reset()
    assert piveto.check(CAP, OCR) == {
        "model": "unknown", "meaningful_prob": None, "band": None,
        "reason": "load_failed",
    }
    assert piveto.model_state() == "unknown"
    _reset()


def test_model_state_reuses_the_lazy_singleton(monkeypatch):
    """Cheap by construction: asking for the receipt never costs a second
    load, however many images ask."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    loads = {"n": 0}

    def _dir():
        loads["n"] += 1
        return "/nope/missing"

    monkeypatch.setattr(piveto, "_veto_dir", _dir)
    _reset()
    assert piveto.model_state() == "absent"
    assert piveto.check(CAP, OCR)["model"] == "absent"
    assert piveto.check(CAP, OCR)["model"] == "absent"
    assert piveto.is_ready() is False
    assert loads["n"] == 1
    _reset()
