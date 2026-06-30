"""Structured-JSON parser — routing, flattening, robustness, source-replay.

JSON is the most common uploaded artifact shape in real deals (portal intake
payloads, manifests, API exports, URL lists). These tests prove the parser:
  * wins routing on structured JSON but DEFERS transcript-shaped JSON;
  * flattens nested objects / arrays of scalars / arrays of objects / JSONL;
  * never crashes on malformed, empty, huge, deeply-nested, or weird input;
  * stamps replayable JSON-Pointer locators and valid EvidenceAtoms.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.ids import stable_id
from app.core.schemas import ArtifactType, AtomType, EvidenceAtom
from app.parsers.json_parser import JsonParser, _dotted, _pointer
from app.parsers.registry import choose_parser


def _emit(tmp_path: Path, name: str, payload, *, raw: str | None = None):
    p = tmp_path / name
    p.write_text(raw if raw is not None else json.dumps(payload), encoding="utf-8")
    out = JsonParser().parse_artifact_full(
        project_id="T", artifact_id=stable_id("art", str(p)), path=p,
    )
    return p, out


def _texts(out) -> list[str]:
    return [a.raw_text for a in out.atoms]


# ── routing ──────────────────────────────────────────────────────────

def test_structured_json_routes_to_json_parser(tmp_path: Path):
    p = tmp_path / "INTAKE_REQUEST.json"
    p.write_text(json.dumps({"site": {"name": "X"}, "request": {"type": "install"}}))
    parser, match, _ = choose_parser(p)
    assert parser is not None
    assert parser.parser_name == "json"
    assert match.confidence >= 0.85
    assert match.artifact_type == ArtifactType.json


def test_jsonl_routes_to_json_parser(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n')
    parser, match, _ = choose_parser(p)
    assert parser is not None and parser.parser_name == "json"
    assert match.confidence >= 0.85


def test_transcript_shaped_json_defers_to_transcript(tmp_path: Path):
    p = tmp_path / "meeting.json"
    p.write_text(json.dumps({"utterances": [{"speaker": "A", "text": "hi"}]}))
    parser, match, _ = choose_parser(p)
    assert parser is not None
    assert parser.parser_name == "transcript"


def test_segments_shaped_json_defers_to_transcript(tmp_path: Path):
    p = tmp_path / "call.json"
    p.write_text(json.dumps({"segments": [{"speaker": "A", "text": "hi", "start": 0}]}))
    parser, _, _ = choose_parser(p)
    assert parser is not None and parser.parser_name == "transcript"


def test_routing_is_deterministic(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"case_id": "OX-1", "domain_pack": "security_camera"}))
    a = choose_parser(p)
    b = choose_parser(p)
    assert a[1].parser_name == b[1].parser_name
    assert a[1].confidence == b[1].confidence


# ── flattening: shapes ───────────────────────────────────────────────

def test_nested_object_flattens_with_dotted_paths(tmp_path: Path):
    payload = {
        "site": {"name": "BarCons FCU", "address": {"city": "Columbus", "state": "IN"}},
        "request": {"type": "install"},
    }
    _, out = _emit(tmp_path, "intake.json", payload)
    texts = _texts(out)
    assert "site.name: BarCons FCU" in texts
    assert "site.address.city: Columbus" in texts
    assert "site.address.state: IN" in texts
    assert "request.type: install" in texts


def test_flat_object_keyvalues(tmp_path: Path):
    payload = {"case_id": "OX-9", "domain_pack": "security_camera", "intake_id": "OX-0009"}
    _, out = _emit(tmp_path, "case_manifest.json", payload)
    texts = _texts(out)
    assert "case_id: OX-9" in texts
    assert "domain_pack: security_camera" in texts


def test_array_of_scalars(tmp_path: Path):
    _, out = _emit(tmp_path, "docs.json", ["http://a/x.pdf", "http://b/y.docx"])
    texts = _texts(out)
    assert "[0]: http://a/x.pdf" in texts
    assert "[1]: http://b/y.docx" in texts


def test_array_of_objects_indexes_each_record(tmp_path: Path):
    payload = [
        {"part": "C9300-48P", "qty": 2},
        {"part": "SMT3000RM2U", "qty": 1},
    ]
    _, out = _emit(tmp_path, "bom.json", payload)
    texts = _texts(out)
    assert "[0].part: C9300-48P" in texts
    assert "[0].qty: 2" in texts
    assert "[1].part: SMT3000RM2U" in texts


def test_top_level_scalar(tmp_path: Path):
    _, out = _emit(tmp_path, "scalar.json", "just a string")
    assert len(out.atoms) == 1
    assert out.atoms[0].raw_text == "just a string"


def test_jsonl_records_indexed_by_line(tmp_path: Path):
    _, out = _emit(
        tmp_path, "events.jsonl", None,
        raw='{"id": 1, "name": "alpha"}\n{"id": 2, "name": "beta"}\n',
    )
    texts = _texts(out)
    assert "line1.id: 1" in texts
    assert "line1.name: alpha" in texts
    assert "line2.name: beta" in texts


# ── value typing ─────────────────────────────────────────────────────

def test_booleans_numbers_rendered_and_typed(tmp_path: Path):
    payload = {"escort_required": False, "qty": 3, "rate": 12.5, "active": True}
    _, out = _emit(tmp_path, "vals.json", payload)
    by_path = {a.value.get("key_path"): a for a in out.atoms}
    assert by_path["escort_required"].raw_text == "escort_required: false"
    assert by_path["escort_required"].value["value_type"] == "boolean"
    assert by_path["qty"].value["value_type"] == "number"
    assert by_path["rate"].raw_text == "rate: 12.5"
    assert by_path["active"].raw_text == "active: true"


def test_nulls_and_empty_strings_skipped(tmp_path: Path):
    payload = {"a": None, "b": "", "c": "   ", "d": "keep"}
    _, out = _emit(tmp_path, "nulls.json", payload)
    texts = _texts(out)
    assert texts == ["d: keep"]


# ── source-replay locators ───────────────────────────────────────────

def test_atoms_carry_json_pointer_locator(tmp_path: Path):
    _, out = _emit(tmp_path, "x.json", {"site": {"address": {"city": "Columbus"}}})
    a = next(a for a in out.atoms if "city" in a.raw_text)
    src = a.source_refs[0]
    assert src.artifact_type == ArtifactType.json
    assert src.locator["json_pointer"] == "/site/address/city"
    assert src.locator["key_path"] == "site.address.city"
    assert src.extraction_method == "json_flatten"


def test_json_pointer_round_trips_to_value(tmp_path: Path):
    payload = {"a": {"b": [{"c": "deep-value"}]}}
    _, out = _emit(tmp_path, "deep.json", payload)
    a = next(a for a in out.atoms if a.raw_text.endswith("deep-value"))
    ptr = a.source_refs[0].locator["json_pointer"]
    # Resolve the pointer manually against the original document.
    node = payload
    for tok in ptr.split("/")[1:]:
        node = node[int(tok)] if tok.isdigit() else node[tok]
    assert node == "deep-value"


def test_pointer_escapes_special_chars():
    assert _pointer(["a/b", "c~d"]) == "/a~1b/c~0d"
    assert _dotted(["items", "[0]", "name"]) == "items[0].name"


# ── robustness ───────────────────────────────────────────────────────

def test_malformed_json_emits_raw_fallback_not_crash(tmp_path: Path):
    _, out = _emit(tmp_path, "broken.json", None, raw='{"a": 1, oops not json')
    assert len(out.atoms) == 1
    assert out.atoms[0].raw_text.startswith("(unparsed JSON)")
    assert any("not valid" in w for w in out.warnings)


def test_empty_object_emits_marker(tmp_path: Path):
    _, out = _emit(tmp_path, "empty.json", {})
    assert len(out.atoms) == 1
    assert "no scalar values" in out.atoms[0].raw_text


def test_empty_file_no_crash(tmp_path: Path):
    _, out = _emit(tmp_path, "blank.json", None, raw="")
    assert out.atoms == []
    assert any("empty" in w for w in out.warnings)


def test_all_null_object_emits_marker(tmp_path: Path):
    _, out = _emit(tmp_path, "nulls.json", {"a": None, "b": None})
    assert len(out.atoms) == 1
    assert "no scalar values" in out.atoms[0].raw_text


def test_deeply_nested_does_not_crash(tmp_path: Path):
    node: dict = {}
    cur = node
    for i in range(500):  # far beyond max_depth
        cur["child"] = {}
        cur = cur["child"]
    cur["leaf"] = "bottom"
    _, out = _emit(tmp_path, "deep.json", node)
    # Should not raise; deep leaf is beyond the depth cap so simply absent.
    assert isinstance(out.atoms, list)


def test_atom_cap_truncates_with_warning(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_JSON_MAX_ATOMS", "10")
    payload = {f"k{i}": f"v{i}" for i in range(100)}
    _, out = _emit(tmp_path, "big.json", payload)
    assert len(out.atoms) == 10
    assert any("truncated" in w for w in out.warnings)


def test_unicode_and_bom(tmp_path: Path):
    p = tmp_path / "u.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"name": "café \u2014 naïve"}).encode("utf-8"))
    out = JsonParser().parse_artifact_full(
        project_id="T", artifact_id="a", path=p,
    )
    assert any("café" in a.raw_text for a in out.atoms)


def test_huge_string_value_is_capped(tmp_path: Path):
    _, out = _emit(tmp_path, "huge.json", {"blob": "x" * 50000})
    a = out.atoms[0]
    assert len(a.raw_text) < 2100


def test_heterogeneous_array(tmp_path: Path):
    _, out = _emit(tmp_path, "mixed.json", [1, "two", True, None, {"k": "v"}, [9]])
    texts = _texts(out)
    assert "[0]: 1" in texts
    assert "[1]: two" in texts
    assert "[2]: true" in texts
    assert "[4].k: v" in texts
    assert "[5][0]: 9" in texts


# ── atom validity ────────────────────────────────────────────────────

def test_all_atoms_are_valid_evidence_atoms(tmp_path: Path):
    payload = {
        "site": {"name": "X", "lift_required": "yes"},
        "items": [{"part": "A"}, {"part": "B"}],
        "notes": ["call onsite contact"],
    }
    _, out = _emit(tmp_path, "full.json", payload)
    assert out.atoms
    for a in out.atoms:
        assert isinstance(a, EvidenceAtom)
        assert a.raw_text.strip()
        assert a.normalized_text.strip()
        assert a.source_refs and a.source_refs[0].artifact_type == ArtifactType.json
        assert isinstance(a.atom_type, AtomType)
        assert 0.0 <= a.confidence <= 1.0


def test_constraint_language_classifies_as_constraint(tmp_path: Path):
    # Natural-language values flow through the same classifier the universal
    # parsers use, so a JSON note that reads as a constraint buckets correctly.
    _, out = _emit(tmp_path, "c.json", {"note": "escort required for all techs"})
    a = out.atoms[0]
    assert a.atom_type == AtomType.constraint


def test_exclusion_language_classifies_as_exclusion(tmp_path: Path):
    _, out = _emit(tmp_path, "e.json", {"note": "cabling is out of scope for this site"})
    assert out.atoms[0].atom_type == AtomType.exclusion


def test_ids_are_deterministic(tmp_path: Path):
    payload = {"a": {"b": "c"}}
    _, o1 = _emit(tmp_path, "id1.json", payload)
    p2 = tmp_path / "id1.json"  # same artifact -> same ids
    o2 = JsonParser().parse_artifact_full(
        project_id="T", artifact_id=stable_id("art", str(p2)), path=p2,
    )
    assert [a.id for a in o1.atoms] == [a.id for a in o2.atoms]
