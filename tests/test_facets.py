"""Facet dashboard sections: the safe-off contract (no torch/model needed here).

With the flag off OR the facet head absent, build_facet_sections must return a
disabled, empty result and never raise — that's the guarantee that makes it safe
to ship OFF/cold = byte-identical envelope (the key is omitted entirely).
"""
from app.core import facets


class _Atom:
    def __init__(self, aid):
        self.id = aid
        self.text = "some clause text"


def test_disabled_when_flag_off(monkeypatch):
    monkeypatch.delenv("SOWSMITH_FACET_SECTIONS", raising=False)
    out = facets.build_facet_sections([_Atom("atm_1"), _Atom("atm_2")])
    assert out == {"enabled": False, "sections": []}


def test_disabled_when_head_absent(monkeypatch, tmp_path):
    # flag ON but no model dir -> head absent -> still disabled, never raises
    monkeypatch.setenv("SOWSMITH_FACET_SECTIONS", "1")
    monkeypatch.setenv("SOWSMITH_CONTRASTIVE_FACET_DIR", str(tmp_path / "nope"))
    out = facets.build_facet_sections([_Atom("atm_1")])
    assert out["enabled"] is False


def test_assign_facets_abstains_without_head(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_CONTRASTIVE_FACET_DIR", str(tmp_path / "nope"))
    assert facets.assign_facets(["a", "b", "c"]) == [None, None, None]
    assert facets.assign_facets([]) == []


def test_empty_atoms(monkeypatch):
    monkeypatch.setenv("SOWSMITH_FACET_SECTIONS", "1")
    assert facets.build_facet_sections([]) == {"enabled": False, "sections": []}
