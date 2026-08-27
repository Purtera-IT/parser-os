"""PDF embedded-image understanding: abstain-first contract + routing + guards.

OFF or no endpoint -> emits nothing (byte-identical to today). When on, the
classify gate routes to describe/transcribe, and the verbatim/context guards
drop fabricated content. The VLM + OCR are monkeypatched so these run offline.
"""
import types

from app.core import pdf_image_vision as piv


def _mock_reachable(monkeypatch):
    monkeypatch.setattr(piv, "_vision_reachable", lambda: True)


def _marker(tmp_path, *, saved_name="page2_image7.png", region="page2/image7",
            caption="Upload photo showing Battery Charger Mounting", size=5000):
    p = tmp_path / saved_name
    p.write_bytes(b"\x89PNG\r\n" + b"0" * size)  # >min_bytes, content irrelevant (VLM mocked)
    atom = types.SimpleNamespace(
        id="atm_marker_1",
        project_id="proj1",
        artifact_id="art1",
        parser_version="vtest",
        value={"kind": "image_marker", "region_ref": region,
               "saved_path": str(p), "expected_content": caption},
        source_refs=[types.SimpleNamespace(filename="install_guide.pdf")],
    )
    return atom


# ── abstain-first ───────────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_VISION", raising=False)
    assert piv.enabled() is False
    assert piv.process_image_markers([object()]) == []


def test_abstains_without_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr(piv, "_vision_reachable", lambda: False)
    assert piv.process_image_markers([_marker(tmp_path)]) == []


def test_tiny_crop_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    called = {"n": 0}
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or "{}")
    m = _marker(tmp_path, size=10)  # below SOWSMITH_PDF_IMAGE_MIN_BYTES
    assert piv.process_image_markers([m]) == []
    assert called["n"] == 0  # never even called the gate


# ── helpers ─────────────────────────────────────────────────────────


def test_parse_json_obj_handles_noise():
    assert piv._parse_json_obj('garbage {"a": 1} trailing') == {"a": 1}
    assert piv._parse_json_obj("not json") == {}


def test_iter_image_markers_filters(tmp_path):
    good = _marker(tmp_path)
    not_image = types.SimpleNamespace(
        value={"kind": "chart_marker", "region_ref": "page1/chart1", "saved_path": "x"},
        source_refs=[types.SimpleNamespace(filename="a.pdf")],
    )
    no_save = types.SimpleNamespace(
        value={"kind": "image_marker", "region_ref": "page1/image1"},
        source_refs=[types.SimpleNamespace(filename="a.pdf")],
    )
    rows = list(piv._iter_image_markers([good, not_image, no_save]))
    assert len(rows) == 1
    assert rows[0][1] == "install_guide.pdf"
    assert rows[0][2] == 2  # page index


def test_context_guard():
    assert piv._context_guard("anything", "x", 0.0) is True            # disabled
    assert piv._context_guard("battery charger mounting wall", "", 0.3) is True   # no ground text
    assert piv._context_guard(
        "battery charger mounted on electrical room wall",
        "the battery charger is mounted in the electrical room", 0.3) is True
    assert piv._context_guard(
        "elephants dancing on saturn rings tonight",
        "the battery charger is mounted in the electrical room", 0.3) is False


def test_verbatim_ok():
    ocr = "set vlan 10 on switch port gigabitethernet 0 1"
    assert piv._verbatim_ok("set vlan 10 on switch port", ocr) is True
    assert piv._verbatim_ok("reboot the firewall appliance now", ocr) is False
    assert piv._verbatim_ok("anything", "") is False  # no OCR -> reject


# ── routing: describe ───────────────────────────────────────────────


def _route_vlm(gate_kind, *, describe=None, transcribe=None):
    def _impl(image_bytes, prompt, *, model=None, max_tokens=0):
        if "triaging" in prompt:
            return ('{"image_kind": "%s", "has_text": true, "meaningful": true}' % gate_kind)
        if "describing an image" in prompt:
            return describe or "{}"
        if "transcribing" in prompt:
            return transcribe or "{}"
        return "{}"
    return _impl


def test_describe_emits_grounded_atoms(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))  # no doc on disk
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "photo",
        describe='{"description": "Battery charger mounted on the north wall",'
                 ' "facts": [{"kind": "equipment", "text": "wall-mounted battery charger"}]}',
    ))
    out = piv.process_image_markers([_marker(tmp_path)])
    kinds = {a.value["fact_kind"] for a in out}
    assert "image_description" in kinds
    assert any(k.startswith("image_fact:") for k in kinds)
    assert all(a.value["via"] == "pdf_image_vision" for a in out)
    assert all("pdf_image_vision" in a.review_flags for a in out)


def test_skip_kind_abstains(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_vlm", lambda image_bytes, prompt, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    assert piv.process_image_markers([_marker(tmp_path)]) == []


# ── routing: transcribe + verbatim guard ────────────────────────────


def test_transcribe_drops_fabricated_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    # OCR sees only the real command.
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "Step 1 set vlan 10 on port gi0/1")
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "instructions",
        transcribe='{"summary": "VLAN setup", "steps": ['
                   '{"n": 1, "action": "set vlan on port", "command": "set vlan 10 on port gi0/1"},'
                   '{"n": 2, "action": "reboot the firewall", "command": "reload firewall now"}]}',
    ))
    out = piv.process_image_markers([_marker(tmp_path)])
    steps = [a for a in out if a.value["fact_kind"] == "image_instruction_step"]
    assert len(steps) == 1                      # fabricated step 2 dropped by verbatim guard
    assert "vlan 10" in steps[0].raw_text
    assert any(a.value["fact_kind"] == "image_instructions_summary" for a in out)


def test_ocr_crop_vlm_fallback(monkeypatch, tmp_path):
    """When the OCR chain yields nothing, the transcribe path falls back to the
    VLM OCR call (allow_vlm); the gate context must NOT (cost guard)."""
    monkeypatch.setattr("app.parsers._ocr_chain.ocr_image_file", lambda p: {"text": ""})
    calls = {"n": 0}

    def _vlm(image_bytes, prompt, **k):
        calls["n"] += 1
        return "set vlan 10 on port gi0/1"

    monkeypatch.setattr(piv, "_vlm", _vlm)
    p = tmp_path / "c.png"
    p.write_bytes(b"\x89PNG" + b"0" * 5000)
    # gate context: no VLM OCR
    assert piv._ocr_crop(str(p), b"x") == ""
    assert calls["n"] == 0
    # transcribe context: VLM OCR fires
    assert "vlan 10" in piv._ocr_crop(str(p), b"x", allow_vlm=True)
    assert calls["n"] == 1


def test_transcribe_uses_vlm_ocr_when_chain_empty(monkeypatch, tmp_path):
    """End-to-end: empty OCR chain -> VLM OCR anchor -> verbatim step survives."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr("app.parsers._ocr_chain.ocr_image_file", lambda p: {"text": ""})

    def _vlm(image_bytes, prompt, *, model=None, max_tokens=0):
        if "triaging" in prompt:
            return '{"image_kind": "instructions", "has_text": true, "meaningful": true}'
        if "OCR engine" in prompt:
            return "Step 1 set vlan 10 on port gi0/1"
        if "transcribing" in prompt:
            return ('{"summary": "VLAN setup", "steps": [{"n": 1, '
                    '"action": "set vlan on port", "command": "set vlan 10 on port gi0/1"}]}')
        return "{}"

    monkeypatch.setattr(piv, "_vlm", _vlm)
    out = piv.process_image_markers([_marker(tmp_path)])
    steps = [a for a in out if a.value["fact_kind"] == "image_instruction_step"]
    assert len(steps) == 1
    assert "vlan 10" in steps[0].raw_text


def test_dedup_identical_crops(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    calls = {"n": 0}

    def _vlm(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"image_kind": "photo", "has_text": false, "meaningful": true}'
        return '{"description": "rack photo", "facts": []}'

    monkeypatch.setattr(piv, "_vlm", _vlm)
    m1 = _marker(tmp_path, region="page2/image7", saved_name="a.png")
    m2 = _marker(tmp_path, region="page3/image2", saved_name="b.png")
    # Same bytes -> same hash -> second marker skipped
    out = piv.process_image_markers([m1, m2])
    assert calls["n"] == 2  # gate + describe once
    assert len(out) == 1


def test_caption_mismatch_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    page = "Rack elevation diagram for the MDF room."
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: (page, "", "", 1))
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "photo",
        describe='{"description": "Rack elevation in the MDF room", "facts": []}',
    ))
    out = piv.process_image_markers([_marker(tmp_path)])
    assert out
    assert "image_answer_mismatch" in out[0].review_flags


def test_resolve_crop_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img_dir = tmp_path / "_extracted_images" / "deal"
    img_dir.mkdir(parents=True)
    p = img_dir / "page0_image1.png"
    p.write_bytes(b"\x89PNG" + b"x" * 5000)
    rel = "_extracted_images/deal/page0_image1.png"
    assert piv._resolve_crop_path(rel) is not None
    assert len(piv._load_crop(rel)) > 5000


def test_force_ollama_default_on(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_FORCE_OLLAMA", raising=False)
    assert piv._use_ollama_for_pdf_images() is True


def test_table_image_emits_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "2 x Cat6 cable 500ft")

    def _vlm2(image_bytes, prompt, **k):
        if "triaging" in prompt:
            return '{"image_kind": "table_image", "meaningful": true, "has_text": true}'
        return '{"line_items": [{"qty": "2", "description": "Cat6 cable", "total": "500ft"}]}'
    monkeypatch.setattr(piv, "_vlm", _vlm2)
    out = piv.process_image_markers([_marker(tmp_path)])
    assert any(a.value["fact_kind"].startswith("table_row:") for a in out)


def test_store_correction_overrides_classify_without_vlm(monkeypatch, tmp_path):
    """PM image-head correction should override the classify gate instantly."""
    import numpy as np
    from app.core.decide import set_store
    from app.core.feedback_store import Correction, FeedbackStore, SCOPE_GLOBAL

    def _embed(texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = 1.0 if "rack elevation" in t.lower() else 0.1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 1e-9, n, 1.0)

    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    store.add(Correction(
        id="corr_img_test",
        relation="pdf_image_kind",
        verdict="diagram",
        scope=SCOPE_GLOBAL,
        exemplars=["MDF rack elevation below"],
        status="active",
    ))
    set_store(store)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")
    try:
        from app.core import pdf_image_gate
        monkeypatch.setattr(pdf_image_gate, "classify", lambda *a, **k: None)
    except Exception:
        pass
    vlm_called = {"n": 0}
    monkeypatch.setattr(
        piv, "_vlm",
        lambda *a, **k: vlm_called.__setitem__("n", vlm_called["n"] + 1) or "{}",
    )
    meaningful, kind, via, conf = piv._classify_image(
        crop=b"x", caption="MDF rack elevation below", saved_path=str(tmp_path / "x.png"),
    )
    set_store(None)
    assert via == "store_gate"
    assert kind == "diagram"
    assert meaningful is True
    assert vlm_called["n"] == 0


# ── gate silver logging: dedup, attribution, honest teachers ────────


def _fresh_log():
    from app.core.training_log import TrainingLog, set_training_log
    log = TrainingLog(":memory:")
    set_training_log(log)
    return log


def _clear_log():
    from app.core.training_log import set_training_log
    set_training_log(None)


def test_gate_silver_content_hash_id_no_dup_on_recompile(monkeypatch, tmp_path):
    """Recompiling the same deal must upsert the same row, not append a copy."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    log = _fresh_log()
    try:
        assert piv.process_image_markers([_marker(tmp_path)]) == []
        assert piv.process_image_markers([_marker(tmp_path)]) == []  # recompile
        rows = log.rows(relation="pdf_image_kind")
        assert len(rows) == 1
        assert rows[0].id.startswith("trn_vlm_")
        assert rows[0].label == "skip"
    finally:
        _clear_log()


def test_gate_silver_attribution_and_split(monkeypatch, tmp_path):
    """deal_id column + pdf/page/region provenance ride on every silver row."""
    from app.core.training_log import assign_split
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    log = _fresh_log()
    try:
        piv.process_image_markers([_marker(tmp_path)])
        (row,) = log.rows(relation="pdf_image_kind")
        assert row.deal_id == "proj1"
        assert row.project_id == "proj1"
        assert row.split == assign_split("proj1")
        assert row.teacher == "llm"
        assert row.provenance["via"] == "vlm_gate"
        assert row.provenance["pdf"] == "install_guide.pdf"
        assert row.provenance["page"] == 2
        assert row.provenance["region_ref"] == "page2/image7"
        assert row.provenance["image_sha16"]
    finally:
        _clear_log()


def test_cpu_gate_never_logged_and_skip_receipted(monkeypatch, tmp_path):
    """The distilled student's own verdicts are not training silver; the skip
    is still receipted on the marker (kind + via)."""
    from app.core import pdf_image_gate
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")
    monkeypatch.setattr(pdf_image_gate, "classify", lambda *a, **k: (False, "logo"))
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: "{}")  # must never be reached
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        assert log.count(relation="pdf_image_kind") == 0
        assert m.value["gate_verdict"] == {"kind": "skip", "via": "cpu_gate"}
    finally:
        _clear_log()


def test_degenerate_feature_never_logged(monkeypatch, tmp_path):
    """No caption + no OCR -> 'no context' -> the verdict applies at runtime
    but teaches nothing, so no silver row is written."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    log = _fresh_log()
    try:
        m = _marker(tmp_path, caption="")
        assert piv.process_image_markers([m]) == []
        assert log.count(relation="pdf_image_kind") == 0
        # The skip is still a receipted, traceable decision.
        assert m.value["gate_verdict"] == {"kind": "logo", "via": "vlm_gate"}
    finally:
        _clear_log()


def test_store_gate_logs_teacher_store(monkeypatch):
    """A store-decided verdict is logged as teacher='store', never 'llm'."""
    import types as _types
    monkeypatch.setattr(
        "app.core.decide.decide",
        lambda *a, **k: _types.SimpleNamespace(
            source="store", verdict="diagram", confidence=0.91),
    )
    log = _fresh_log()
    try:
        hit = piv._store_classify_image(
            "MDF rack elevation below", "",
            {"deal_id": "dealX", "project_id": "dealX",
             "pdf": "a.pdf", "page": 1, "region_ref": "page1/image1",
             "image_sha16": "abc123"},
        )
        assert hit == (True, "diagram", "store_gate", 0.91)
        (row,) = log.rows(relation="pdf_image_kind")
        assert row.teacher == "store"
        assert row.confidence == 0.91
        assert row.deal_id == "dealX"
        assert row.id.startswith("trn_vlm_")
    finally:
        _clear_log()


def test_skip_stamp_added_fields_only(monkeypatch, tmp_path):
    """A meaningful image's marker is NOT stamped; emitted atoms are unchanged
    apart from already-recorded fields."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "photo",
        describe='{"description": "Battery charger mounted on the north wall", "facts": []}',
    ))
    m = _marker(tmp_path)
    out = piv.process_image_markers([m])
    assert out
    assert "gate_verdict" not in m.value


# ── skip veto: recorded second opinion, routing untouched ───────────


def test_vlm_skip_veto_extends_verdict_and_logs(monkeypatch, tmp_path):
    """vlm_gate skip + confident veto -> gate_verdict.veto + trn_veto_ row."""
    from app.core import pdf_image_veto
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 Total Data Outlets")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "decorative", "has_text": false, "meaningful": false}')
    monkeypatch.setattr(pdf_image_veto, "veto", lambda *a, **k: 0.93)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert gv["kind"] == "decorative"
        assert gv["via"] == "vlm_gate"
        assert gv["veto"] == {"meaningful_prob": 0.93, "model": "pdf_image_veto"}
        (row,) = log.rows(relation="pdf_image_veto")
        assert row.id.startswith("trn_veto_")
        assert row.teacher == "veto"
        assert row.label == "meaningful"
        assert row.deal_id == "proj1"
        assert row.provenance["via"] == "vlm_gate"
        assert row.provenance["pdf"] == "install_guide.pdf"
        assert row.provenance["region_ref"] == "page2/image7"
        assert row.provenance["model"] == "pdf_image_veto"
        # Silver kind channel still logged separately for the skip itself.
        assert log.count(relation="pdf_image_kind") == 1
    finally:
        _clear_log()


def test_cpu_gate_skip_never_veto_checked(monkeypatch, tmp_path):
    """cpu_gate skips must not call the veto (sibling student, not independent)."""
    from app.core import pdf_image_gate, pdf_image_veto
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "ocr text")
    monkeypatch.setattr(pdf_image_gate, "classify", lambda *a, **k: (False, "logo"))
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return 0.99

    monkeypatch.setattr(pdf_image_veto, "veto", _boom)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        assert called["n"] == 0
        assert m.value["gate_verdict"]["kind"] == "skip"
        assert m.value["gate_verdict"]["via"] == "cpu_gate"
        assert "veto" not in m.value["gate_verdict"]
        assert log.count(relation="pdf_image_veto") == 0
    finally:
        _clear_log()


def test_veto_off_leaves_routing_identical(monkeypatch, tmp_path):
    """With veto disabled, skip path is unchanged (no veto key, no veto rows)."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_VETO", raising=False)
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "ocr")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        assert m.value["gate_verdict"]["kind"] == "logo"
        assert m.value["gate_verdict"]["via"] == "vlm_gate"
        assert "veto" not in m.value["gate_verdict"]
        assert log.count(relation="pdf_image_veto") == 0
    finally:
        _clear_log()


def test_veto_abstain_does_not_extend_verdict(monkeypatch, tmp_path):
    """Veto enabled but abstaining -> skip receipt only, no veto stamp/row."""
    from app.core import pdf_image_veto
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "ocr")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    monkeypatch.setattr(pdf_image_veto, "veto", lambda *a, **k: None)
    monkeypatch.setattr(pdf_image_veto, "soft_veto", lambda *a, **k: None)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        assert m.value["gate_verdict"]["kind"] == "logo"
        assert m.value["gate_verdict"]["via"] == "vlm_gate"
        assert "veto" not in m.value["gate_verdict"]
        assert "veto_soft" not in m.value["gate_verdict"]
        assert log.count(relation="pdf_image_veto") == 0
    finally:
        _clear_log()


def test_hard_veto_row_carries_band_hard(monkeypatch, tmp_path):
    """Hard vetoes get provenance band='hard' (soft rows must be filterable)."""
    from app.core import pdf_image_veto
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 Total Data Outlets")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "decorative", "has_text": false, "meaningful": false}')
    monkeypatch.setattr(pdf_image_veto, "veto", lambda *a, **k: 0.93)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        (row,) = log.rows(relation="pdf_image_veto")
        assert row.provenance["band"] == "hard"
        assert "crop_ref" not in row.provenance  # blob gate off -> no upload
    finally:
        _clear_log()


# ── soft veto band: harvest signal only, never a PM card ────────────


def _soft_setup(monkeypatch, *, soft=0.75, hard=None):
    from app.core import pdf_image_veto
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VETO", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 Total Data Outlets")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "decorative", "has_text": false, "meaningful": false}')
    monkeypatch.setattr(pdf_image_veto, "veto", lambda *a, **k: hard)
    monkeypatch.setattr(pdf_image_veto, "soft_veto", lambda *a, **k: soft)


def test_soft_veto_stamps_veto_soft_never_veto(monkeypatch, tmp_path):
    """Soft band -> gate_verdict.veto_soft + band='soft' row; the 'veto' key
    (core's ONLY culprit-card trigger) is never set."""
    _soft_setup(monkeypatch, soft=0.75)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert gv["veto_soft"] == {"meaningful_prob": 0.75}
        assert "veto" not in gv  # soft must NEVER become a PM card
        (row,) = log.rows(relation="pdf_image_veto")
        assert row.id.startswith("trn_veto_")
        assert row.teacher == "veto"
        assert row.confidence == 0.75
        assert row.provenance["band"] == "soft"
        assert row.provenance["via"] == "vlm_gate"
    finally:
        _clear_log()


def test_soft_not_checked_when_hard_fires(monkeypatch, tmp_path):
    """Hard band wins; soft_veto is only consulted when the hard veto abstains."""
    from app.core import pdf_image_veto
    _soft_setup(monkeypatch, hard=0.93)

    def _boom(*a, **k):
        raise AssertionError("soft_veto must not be called when hard fired")

    monkeypatch.setattr(pdf_image_veto, "soft_veto", _boom)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert gv["veto"] == {"meaningful_prob": 0.93, "model": "pdf_image_veto"}
        assert "veto_soft" not in gv
    finally:
        _clear_log()


def test_soft_veto_failure_means_no_soft_veto(monkeypatch, tmp_path):
    """Guess-free: soft_veto raising -> plain skip receipt, nothing else."""
    from app.core import pdf_image_veto
    _soft_setup(monkeypatch, soft=None)

    def _boom(*a, **k):
        raise RuntimeError("soft head died")

    monkeypatch.setattr(pdf_image_veto, "soft_veto", _boom)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert "veto" not in gv and "veto_soft" not in gv
        assert log.count(relation="pdf_image_veto") == 0
    finally:
        _clear_log()


# ── disputed-crop persistence: gated best-effort + liveness receipt ─


class _FakeContainer:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def upload_blob(self, *, name, data, overwrite):
        if self.fail is not None:
            raise self.fail
        self.calls.append((name, bytes(data), overwrite))


def _blob_on(monkeypatch, fake):
    monkeypatch.setenv("SOWSMITH_FEEDBACK_BLOB", "1")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    monkeypatch.setattr(piv, "_crop_container_client", lambda: fake)


def _sha16(m):
    import hashlib
    return hashlib.sha256(open(m.value["saved_path"], "rb").read()).hexdigest()[:16]


def test_hard_veto_uploads_crop_and_stamps_ref(monkeypatch, tmp_path):
    from app.core import pdf_image_veto
    _soft_setup(monkeypatch, hard=0.93, soft=None)
    monkeypatch.setattr(pdf_image_veto, "soft_veto", lambda *a, **k: None)
    fake = _FakeContainer()
    _blob_on(monkeypatch, fake)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        expected = f"deals/proj1/orbitbrief/disputed_crops/{_sha16(m)}.png"
        gv = m.value["gate_verdict"]
        assert gv["crop_ref"] == expected
        assert "crop_ref_error" not in gv
        [(name, data, overwrite)] = fake.calls
        assert name == expected
        assert data == open(m.value["saved_path"], "rb").read()  # the pixels
        assert overwrite is True
        (row,) = log.rows(relation="pdf_image_veto")
        assert row.provenance["crop_ref"] == expected
    finally:
        _clear_log()


def test_soft_veto_uploads_crop_too(monkeypatch, tmp_path):
    """Soft disputes persist pixels as well — the review queue needs to see them."""
    _soft_setup(monkeypatch, soft=0.75)
    fake = _FakeContainer()
    _blob_on(monkeypatch, fake)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        expected = f"deals/proj1/orbitbrief/disputed_crops/{_sha16(m)}.png"
        assert m.value["gate_verdict"]["crop_ref"] == expected
        assert len(fake.calls) == 1
    finally:
        _clear_log()


def test_crop_upload_failure_emits_liveness_receipt(monkeypatch, tmp_path):
    """Doctrine: a dead uploader must be OBSERVABLE — failure stamps the
    exception class as crop_ref_error, never silence."""
    _soft_setup(monkeypatch, hard=0.93, soft=None)
    fake = _FakeContainer(fail=ConnectionError("blob down"))
    _blob_on(monkeypatch, fake)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert gv["crop_ref_error"] == "ConnectionError"
        assert "crop_ref" not in gv
        assert gv["veto"]["meaningful_prob"] == 0.93  # veto still recorded
        (row,) = log.rows(relation="pdf_image_veto")
        assert "crop_ref" not in row.provenance
    finally:
        _clear_log()


def test_crop_upload_gate_off_is_deliberate_silence(monkeypatch, tmp_path):
    """SOWSMITH_FEEDBACK_BLOB unset = configured OFF: no upload attempt, no
    receipt (a config choice is not a dead uploader)."""
    _soft_setup(monkeypatch, hard=0.93, soft=None)
    monkeypatch.delenv("SOWSMITH_FEEDBACK_BLOB", raising=False)

    def _boom():
        raise AssertionError("client must not be constructed when gated off")

    monkeypatch.setattr(piv, "_crop_container_client", _boom)
    log = _fresh_log()
    try:
        m = _marker(tmp_path)
        assert piv.process_image_markers([m]) == []
        gv = m.value["gate_verdict"]
        assert "crop_ref" not in gv and "crop_ref_error" not in gv
        assert gv["veto"]["meaningful_prob"] == 0.93
    finally:
        _clear_log()


def test_undisputed_skip_never_uploads(monkeypatch, tmp_path):
    """Only DISPUTED images (a fired veto) persist crops — never every image."""
    _soft_setup(monkeypatch, hard=None, soft=None)
    fake = _FakeContainer()
    _blob_on(monkeypatch, fake)
    m = _marker(tmp_path)
    assert piv.process_image_markers([m]) == []
    assert fake.calls == []
    gv = m.value["gate_verdict"]
    assert "crop_ref" not in gv and "crop_ref_error" not in gv


def test_skip_stamp_includes_ocr_preview(monkeypatch, tmp_path):
    """Skip receipts carry a short OCR preview for the PM culprit surface."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    _mock_reachable(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 Total Data Outlets Comm Cabinet")
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k:
                        '{"image_kind": "decorative", "has_text": false, "meaningful": false}')
    m = _marker(tmp_path)
    assert piv.process_image_markers([m]) == []
    gv = m.value["gate_verdict"]
    assert gv["kind"] == "decorative"
    assert gv["via"] == "vlm_gate"
    assert "18 Total Data Outlets" in gv["ocr_preview"]
