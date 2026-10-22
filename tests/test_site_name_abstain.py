"""PUR-22 / PUR-49 / PUR-50: site names are copied from documents or abstain.

Synthetic atoms and fixtures only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import site_facility_head as sfh
from app.core.site_facility_head import (
    SITE_NAME_UNKNOWN_FLAG,
    annotate_site_facility_labels,
    decide_site_facility_label,
    name_is_verbatim,
)
from app.core.site_naming import annotate_site_name_status, site_name_label
from tools.base_health import normalize
from tools.site_name_attribution import attribute, iter_envelopes
from tools.site_name_training_examples import extract_examples, normalized_key

FIX = Path(__file__).parent / "fixtures" / "site_names"


class _Atom:
    def __init__(self, value, raw_text=""):
        self.atom_type = type("T", (), {"value": "physical_site"})()
        self.value = value
        self.raw_text = raw_text
        self.entity_keys = []
        self.review_flags = []


ADDRESS_SITES = [
    ({"name": "625 W. Adams St, Springfield, IL 60661", "street_address": "625 W. Adams St",
      "city": "SPRINGFIELD", "state": "IL"}, "625 W. Adams St |, Springfield, IL 60661"),
    ({"name": "100 S COMMONS STE 145", "city": "Pittsburgh"}, "100 S COMMONS STE 145 PITTSBURGH, PA15212"),
    ({"street_address": "1 Elm Dr", "city": "Ogdenville"}, "1 Elm Dr Ogdenville"),
    ({}, "Some text with no name"),
    ({"city": "Capital City", "aliases": ["capital city warehouse"]}, "77 Fake Blvd, Capital City"),
]


@pytest.fixture(autouse=True)
def _no_champion(monkeypatch):
    import app.learning.head_registry as hr
    monkeypatch.setattr(hr, "get_head_registry", lambda: None)
    monkeypatch.setattr(sfh, "log_rows", lambda rows: None)


@pytest.mark.parametrize("value,raw", ADDRESS_SITES)
def test_rule_never_emits_a_name_absent_from_source(value, raw):
    atom = _Atom(dict(value), raw)
    d = decide_site_facility_label(atom)
    source = " ".join([raw, *[str(v) for v in value.get("aliases", [])],
                       str(value.get("name") or ""), str(value.get("facility_name") or "")])
    if d.facility_name is not None:
        assert normalize(d.facility_name) in normalize(source)
        assert not normalize(d.facility_name).endswith(" office") or "office" in normalize(raw)
    assert d.facility_name not in {"Site 1"}
    assert not (d.facility_name or "").lower().endswith(f"{str(value.get('city','')).lower()} office") or not value.get("city")


def test_address_only_site_abstains_and_still_exists():
    atom = _Atom({"id": "S3", "name": "625 W. Adams St, Springfield, IL 60661", "city": "Springfield",
                  "street_address": "625 W. Adams St"}, "625 W. Adams St, Springfield, IL 60661")
    atoms, n = annotate_site_facility_labels([atom], project_id="syn")
    assert n == 1 and atoms == [atom]
    v = atom.value
    assert v["facility_name"] is None and v["name"] is None and v["display_name"] is None
    assert v["street_address"] == "625 W. Adams St"
    assert v["name_source"] == {"rule": "rule:abstain", "verbatim": None,
                                "abstain_reason": "no_document_names_site"}
    assert SITE_NAME_UNKNOWN_FLAG in atom.review_flags


def test_real_facility_name_is_kept_and_tagged_verbatim():
    atom = _Atom({"id": "S1", "facility_name": "Riverside Clinic"}, "Riverside Clinic, 4 Oak Ave")
    annotate_site_facility_labels([atom])
    assert atom.value["facility_name"] == "Riverside Clinic"
    assert atom.value["name_source"]["rule"] == "rule:facility_name_field"
    assert atom.value["name_source"]["verbatim"] is True


class _Verdict:
    def __init__(self, verdict):
        self.verdict, self.confidence, self.route_llm = verdict, 0.9, False


class _Head:
    def __init__(self, verdict):
        self._v = verdict

    def classify(self, vec, candidates):
        return _Verdict(self._v)


def _with_champion(monkeypatch, verdict):
    import app.core.embedding_retrieval as er
    import app.learning.head_registry as hr
    reg = type("R", (), {"champion": lambda self, rel: (_Head(verdict), {})})()
    monkeypatch.setattr(hr, "get_head_registry", lambda: reg)
    monkeypatch.setattr(er, "embed_texts", lambda texts: [[0.0]] * len(texts))


def test_champion_city_office_does_not_compose(monkeypatch):
    _with_champion(monkeypatch, "city_office")
    atom = _Atom({"name": "10 Main Rd", "city": "Shelbyville"}, "10 Main Rd, Shelbyville")
    d = decide_site_facility_label(atom)
    assert d.source == "neural_head" and d.facility_name is None
    assert d.abstain_reason == "head_verdict_name_not_in_source"


def test_champion_city_office_copies_when_document_says_it(monkeypatch):
    _with_champion(monkeypatch, "city_office")
    atom = _Atom({"name": "10 Main Rd", "city": "Shelbyville"}, "Shelbyville office, 10 Main Rd")
    d = decide_site_facility_label(atom)
    assert d.facility_name == "Shelbyville office"
    assert name_is_verbatim(d.facility_name, atom.raw_text) is True


def test_unnamed_site_renders_n_of_m():
    rows = [{"site": "site:a", "facility_name": "HQ"}, {"site": "site:b"}, {"site": "site:c", "facility_name": None}]
    assert annotate_site_name_status(rows) == 2
    assert rows[0]["name_label"] == "HQ"
    assert rows[1]["name_label"] == "site 2 of 3, name unknown"
    assert rows[2]["name_status"] == "unknown"
    assert site_name_label(3, 7, None) == "site 3 of 7, name unknown"


def test_attribution_report_on_synthetic_envelopes():
    r = attribute(iter_envelopes(FIX / "envelopes"), "2026-09-17")
    assert r["as_of"] == "2026-09-17" and r["envelopes"] == 2
    rows = {x["rule"]: x for x in r["rules"]}
    assert rows["legacy:city_office_composed"]["invented"] == 2
    assert rows["legacy:city_office_composed"]["deals_hit"] == 2
    assert rows["legacy:city_as_name"]["wrong_field"] == 1
    assert rows["rule:facility_name_field"]["verbatim"] == 1
    assert r["rules"][0]["rule"] == "legacy:city_office_composed"
    assert r["unnamed_site_atoms"] == 1
    assert r["totals"]["undecidable"] == 1


def test_training_examples_key_on_document_and_abstain():
    kit = json.loads((FIX / "kit_synthetic.json").read_text())
    env = json.loads((FIX / "envelope_for_kit.json").read_text())
    ex = extract_examples(kit, env)
    stl = [e for e in ex if e["kit_site_name"] == "St. Louis Office"]
    # spelling variants, one key; the sow2 mention is excluded by the provenance join
    assert {e["document_id"] for e in stl} == {"sow1"}
    assert {e["normalized_key"] for e in stl} == {"st louis office"}
    assert {e["source_text"] for e in stl} == {"SAINT LOUIS OFFICE", "Saint Louis office"}
    for e in stl:
        text = next(a["text"] for a in env["atoms"] if a["id"] == e["atom_id"])
        assert text[e["span"][0]:e["span"][1]] == e["source_text"]
    none = [e for e in ex if e["kit_site_name"] == "Nowhere Annex"]
    assert none and none[0]["document_id"] is None and none[0]["abstain_reason"] == "no_document_names_site"
    assert normalized_key("Saint Louis Office") == normalized_key("ST. LOUIS OFFICE")
