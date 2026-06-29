"""PDF embedded-image understanding: abstain-first contract + routing + guards.

OFF or no endpoint -> emits nothing (byte-identical to today). When on, the
classify gate routes to describe/transcribe, and the verbatim/context guards
drop fabricated content. The VLM + OCR are monkeypatched so these run offline.
"""
import types

from app.core import pdf_image_vision as piv


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
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: False)
    assert piv.process_image_markers([_marker(tmp_path)]) == []


def test_tiny_crop_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
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
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
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
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
    monkeypatch.setattr(piv, "_vlm", lambda image_bytes, prompt, **k:
                        '{"image_kind": "logo", "has_text": false, "meaningful": false}')
    assert piv.process_image_markers([_marker(tmp_path)]) == []


# ── routing: transcribe + verbatim guard ────────────────────────────


def test_transcribe_drops_fabricated_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    # OCR sees only the real command.
    monkeypatch.setattr(piv, "_ocr_crop", lambda p: "Step 1 set vlan 10 on port gi0/1")
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


def test_dedup_identical_crops(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
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
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
    page = "Rack elevation diagram for the MDF room."
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: (page, "", "", 1))
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "photo",
        describe='{"description": "Rack elevation in the MDF room", "facts": []}',
    ))
    out = piv.process_image_markers([_marker(tmp_path)])
    assert out
    assert "image_answer_mismatch" in out[0].review_flags


def test_table_image_emits_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr("app.core.vision_extraction.vision_endpoint_reachable", lambda: True)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr(piv, "_ocr_crop", lambda p: "2 x Cat6 cable 500ft")
    monkeypatch.setattr(piv, "_vlm", _route_vlm(
        "table_image",
        describe='{"line_items": [{"qty": "2", "description": "Cat6 cable", "total": "500ft"}]}',
    ))
    # Second VLM call is BOM extract — return JSON the vision parser understands
    def _vlm2(image_bytes, prompt, **k):
        if "triaging" in prompt:
            return '{"image_kind": "table_image", "meaningful": true, "has_text": true}'
        return '{"line_items": [{"qty": "2", "description": "Cat6 cable", "total": "500ft"}]}'
    monkeypatch.setattr(piv, "_vlm", _vlm2)
    out = piv.process_image_markers([_marker(tmp_path)])
    assert any(a.value["fact_kind"].startswith("table_row:") for a in out)
