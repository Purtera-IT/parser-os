"""SOW writer rules, batch 11 (pass Y grades: 010326 D, 010279 B, 010082 B, 010209 B,
010373 A-, plus the stringified-cite drops 010115 and the 000123 count)."""
from __future__ import annotations

import json

from app.core import c3_sow as m


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


def _fact(t, cites=("a",), **kw):
    return {"text": t, "cites": list(cites), "kind": "fact", **kw}


def _std(t):
    return {"text": t, "cites": [], "kind": "method", "standard": True}


# --- 010115 pass Y: cites the writer sent as a stringified list -----------------

def test_a_stringified_cite_list_is_a_list_of_ids():
    x = m._item({"text": "Use common hand tools as needed.", "kind": "fact",
                 "cites": ["['atm_8e8d8cc3134002fb', 'atm_3c0f824b78b9c2c3']"]})
    assert x["cites"] == ["atm_8e8d8cc3134002fb", "atm_3c0f824b78b9c2c3"]
    assert m._item({"text": "t", "cites": "atm_8e8d8cc3134002fb"})["cites"] == ["atm_8e8d8cc3134002fb"]
    assert m._item({"text": "t", "cites": ["atm_8e8d8cc3134002fb"]})["cites"] == ["atm_8e8d8cc3134002fb"]
    assert m._item({"text": "t", "cites": None})["cites"] == []


def test_a_mistyped_cite_names_the_one_fact_it_nearly_spells():
    known = {"atm_3c0f824b78b9c2c3", "atm_8e8d8cc3134002fb", "atm_88c6acfe1909b383"}
    assert m._near_cite("atm_3c0f824d0a8fe246", known) == "atm_3c0f824b78b9c2c3"   # long prefix
    assert m._near_cite("atm_0a8c6cfe1909b383", known) == "atm_88c6acfe1909b383"   # long suffix
    assert m._near_cite("atm_ffff0000aaaa1111", known) is None
    # Two ids sharing the prefix: no guess.
    assert m._near_cite("atm_3c0f824d0a8fe246", known | {"atm_3c0f8240000000aa"}) is None


def test_the_check_keeps_a_line_whose_cites_were_stringified(monkeypatch):
    monkeypatch.setattr(m, "_complete", lambda prompt, *a, **k: json.dumps({"supported": list(range(50))}))
    c = _content()
    c["scope"] = [m._item({"text": "Use common hand tools as needed for dismantling and boxing.", "kind": "fact",
                           "cites": ["['atm_8e8d8cc3134002fb']"]}),
                  m._item({"text": "Complete dismantling and boxing of all everyday POS hardware.", "kind": "fact",
                           "cites": ["atm_3c0f824d0a8fe246"]})]
    lines = {"atm_8e8d8cc3134002fb": "Use common hand tools as needed for dismantling and boxing",
             "atm_3c0f824b78b9c2c3": "Complete dismantling and boxing of all everyday POS hardware"}
    m.check(c, lines)
    assert [x["cites"] for x in c["scope"]] == [["atm_8e8d8cc3134002fb"], ["atm_3c0f824b78b9c2c3"]]
    assert not c.get("_unknown_cites")


# --- 000123 pass Y: a line in dropped never renders ------------------------------

def test_a_dropped_line_never_renders():
    c = _content()
    c["overview"] = [_fact("Provider will install POS systems at one hundred sixty-one (161) BevMo locations.", ()),
                     _fact("Provider will install POS systems at 161 Customer locations.")]
    c["dropped_unsupported"] = [{"list": "overview", "text": "Provider will install POS systems at one hundred sixty-one (161) BevMo locations.",
                                 "cites": [], "reason": "method line with a number or name no fact gives"}]
    m._dropped_not_rendered(c)
    assert _texts(c, "overview") == ["Provider will install POS systems at 161 Customer locations."]
