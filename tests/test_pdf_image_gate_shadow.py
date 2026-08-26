"""CPU gate shadow logs pairs without changing routing."""
from app.core import pdf_image_gate, pdf_image_vision
from app.core.training_log import TrainingLog


def test_shadow_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_GATE_SHADOW", raising=False)
    assert pdf_image_gate.shadow_enabled() is False


def test_shadow_logs_pair_without_cpu_routing(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_GATE_SHADOW", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_GATE_CPU", "0")
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", str(tmp_path / "t.db"))
    monkeypatch.setattr(pdf_image_gate, "probe", lambda *a, **k: (True, "diagram", 0.61))
    monkeypatch.setattr(pdf_image_gate, "classify", lambda *a, **k: None)
    monkeypatch.setattr(
        pdf_image_vision, "_vlm",
        lambda *a, **k: '{"meaningful": false, "image_kind": "skip"}',
    )
    monkeypatch.setattr(pdf_image_vision, "_ocr_crop", lambda *a, **k: "18 outlets")
    monkeypatch.setattr(pdf_image_vision, "_store_classify_image", lambda *a, **k: None)
    monkeypatch.setattr(pdf_image_vision, "_log_gate_silver", lambda *a, **k: None)

    meaningful, kind, via, conf = pdf_image_vision._classify_image(
        crop=b"x", caption="Figure A", saved_path="x.png",
        attribution={"deal_id": "d1", "pdf": "a.pdf", "page": 1, "region_ref": "page1/image0"},
    )
    assert via == "vlm_gate"
    assert meaningful is False
    assert kind == "skip"
    log = TrainingLog(str(tmp_path / "t.db"))
    rows = list(log.rows(relation="pdf_image_gate_shadow"))
    assert len(rows) == 1
    assert rows[0].provenance["cpu_kind"] == "diagram"
    assert rows[0].provenance["teacher_kind"] == "skip"
    assert rows[0].provenance["agree"] is False
