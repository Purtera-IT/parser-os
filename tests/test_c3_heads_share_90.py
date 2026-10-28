"""Heads decide more, measured safe (2026-10-28, held-out test A by confidence vs the rule on the same lines):
the type head's confirmations on lines typing never saw are recorded (c3_confirm), and the keep head decides
first on NEVER-type lines (its drops; its keeps only above SOWSMITH_C3_NEVER_KEEP_BAR). Synthetic lines."""
from types import SimpleNamespace as NS

import pytest

from app.core import c3_brief, c3_share, c3_type_base


def _atom(i, t, text="x", value=None, status="approved"):
    return NS(id=f"a{i}", atom_type=t, raw_text=text, review_status=status, value=dict(value or {}))


@pytest.fixture
def types(monkeypatch):
    t: dict = {}
    monkeypatch.setitem(c3_type_base._DEAL, "types", t)
    monkeypatch.setitem(c3_type_base._DEAL, "sha", {})
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY_SURE", "0.5")
    monkeypatch.delenv("SOWSMITH_C3_AUTHORITY_SURE_BY_TYPE", raising=False)
    monkeypatch.setattr(c3_type_base, "skipped", lambda a, sha=None: False)
    return t


# ---------------------------------------------------------------- record_confirms


def test_confirm_recorded_for_keep_and_own_type_at_the_bar(types):
    atoms = [_atom(1, "physical_site"), _atom(2, "deal_metadata"), _atom(3, "service_line"),
             _atom(4, "scope_item"), _atom(5, "task"),
             _atom(6, "stakeholder", value={"c3_label": {"type": "_keep", "p": 0.9, "act": "keep"}}), _atom(7, "task")]
    types.update({"a1": ["_keep", 0.97], "a2": ["deal_metadata", 0.81], "a3": ["_keep", 0.42],
                  "a4": ["task", 0.95], "a5": ["_keep", 0.5], "a6": ["_keep", 0.99]})
    assert c3_type_base.record_confirms(atoms) == 3
    got = {a.id: a.value.get("c3_confirm") for a in atoms}
    assert got["a1"] == {"type": "physical_site", "p": 0.97, "said": "_keep"}
    assert got["a2"] == {"type": "deal_metadata", "p": 0.81, "said": "deal_metadata"}
    assert got["a5"]["p"] == 0.5, "at the bar counts"
    assert got["a3"] is None, "below the authority bar is an abstention: the rule's"
    assert got["a4"] is None, "another type is a disagreement, never a confirmation"
    assert got["a6"] is None and got["a7"] is None, "a labeled line keeps its label; an unread line gets nothing"
    assert all(a.atom_type == t for a, t in zip(atoms, ["physical_site", "deal_metadata", "service_line", "scope_item",
                                                        "task", "stakeholder", "task"])), "a record, never a retype"
    assert all("c3_label" not in a.value for a in atoms if a.id != "a6"), "no c3_label: label readers unchanged"


def test_confirm_off_switch(types, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_RECORD_CONFIRM", "off")
    atoms = [_atom(1, "physical_site")]
    types.update({"a1": ["_keep", 0.99]})
    assert c3_type_base.record_confirms(atoms) == 0 and "c3_confirm" not in atoms[0].value


def test_confirm_skips_deal_kit_lines(types, monkeypatch):
    monkeypatch.setattr(c3_type_base, "skipped", lambda a, sha=None: True)
    atoms = [_atom(1, "rate_card")]
    types.update({"a1": ["_keep", 0.99]})
    assert c3_type_base.record_confirms(atoms) == 0


def test_share_counts_a_standing_confirmation_as_the_heads():
    atoms = [_atom(1, "physical_site", value={"c3_confirm": {"type": "physical_site", "p": 0.9, "said": "_keep"}}),
             _atom(2, "scope_item", value={"c3_confirm": {"type": "task", "p": 0.9, "said": "_keep"}}),
             _atom(3, "task")]
    c = c3_share.line_types(atoms, None)
    assert c["head"] == 1 and c["rule"] == 2, "a line a rule re-typed after the confirmation is the rule's"
    site, _ = c3_share.sites(atoms)
    assert site["head"] == 1


# ---------------------------------------------------------------- NEVER types: admission first


def _build(atoms, reads):
    return c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})


def test_admission_drop_decides_never_type_lines(types):
    atoms = [_atom(1, "raw_utterance", "uh, yeah, so anyway"), _atom(2, "entity", "Acme Corp"),
             _atom(3, "raw_utterance", "Install 4 cameras at the dock door."),
             _atom(4, "raw_utterance", "ok", status="rejected"), _atom(5, "raw_table_row", "a | b")]
    reads = {"a1": {"col:admission": ["drop", 0.93]}, "a2": {"col:admission": ["drop", 0.51]},
             "a3": {"col:admission": ["keep", 0.97]}, "a4": {"col:admission": ["drop", 0.99]},
             "a5": {"col:admission": ["drop", 0.45]}}
    b = _build(atoms, reads)
    keep = b["decided_by"]["keep"]
    assert keep["admission:drop_never_type"] == 2
    assert keep["rule:never"] == 3, "a keep below NEVER_KEEP_BAR, a rejected line and an unsure drop stay the rule's"
    kept = [x["id"] for v in b["sections"].values() for x in v] + [x["id"] for x in b["also_mentioned"]]
    assert kept == [], "the outcome is the rule's: every NEVER line is still out"


def test_never_keep_bar_lets_a_sure_keep_in(types, monkeypatch):
    monkeypatch.setattr(c3_brief, "NEVER_KEEP_BAR", 0.99)
    atoms = [_atom(1, "raw_utterance", "Install 4 cameras at the dock door."), _atom(2, "raw_utterance", "Mount a TV.")]
    reads = {"a1": {"col:admission": ["keep", 0.995]}, "a2": {"col:admission": ["keep", 0.98]}}
    b = _build(atoms, reads)
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a1"]
    assert b["decided_by"]["keep"]["admission:keep"] == 1 and b["decided_by"]["keep"]["rule:never"] == 1


def test_never_admission_off_restores_rule_first(types, monkeypatch):
    monkeypatch.setattr(c3_brief, "NEVER_ADMISSION", False)
    atoms = [_atom(1, "raw_utterance", "uh, yeah")]
    b = _build(atoms, {"a1": {"col:admission": ["drop", 0.99]}})
    assert b["decided_by"]["keep"] == {"rule:never": 1}


def test_never_admission_needs_the_admission_keep_head(types):
    atoms = [_atom(1, "raw_utterance", "uh, yeah")]
    b = _build(atoms, {"a1": {"read:co_action": ["reject", 0.99]}})
    assert b["decided_by"]["keep"] == {"rule:never": 1}


# ---------------------------------------------------------------- our own paper in a quoting read


@pytest.fixture
def quoting(types, monkeypatch):
    monkeypatch.setitem(c3_type_base._DEAL, "view", "quoting")
    monkeypatch.setitem(c3_type_base._DEAL, "view_ours", {"a1", "a3"})
    monkeypatch.delenv("SOWSMITH_C3_OURS_OUT", raising=False)
    return types


def _quoting_atoms():
    return [_atom(1, "scope_item", "Our quote: install 12 cameras."), _atom(2, "scope_item", "Please quote 12 cameras."),
            _atom(3, "deal_metadata", "Quote #1234 valid 30 days")]


def test_ours_left_out_of_a_quoting_brief_visibly(quoting):
    b = _build(_quoting_atoms(), {})            # no reads: every line would be kept by rule:type
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a2"]
    assert b["left_out"]["ours_in_quoting_view"] == ["a1", "a3"]
    assert b["left_out"]["by_reason"]["ours_in_quoting_view"] == 2
    assert {"a1", "a3"} <= set(b["left_out"]["ids"]), "the quote reads left_out.ids: it does not count them"
    assert b["decided_by"]["keep"]["view:ours_in_quoting_view"] == 2


def test_ours_left_out_even_when_a_head_would_keep(quoting):
    b = _build(_quoting_atoms(), {"a1": {"col:admission": ["keep", 0.99]}, "a2": {"col:admission": ["keep", 0.99]}})
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a2"]


@pytest.mark.parametrize("how", ["planning", "off"])
def test_ours_out_only_in_a_quoting_read_and_with_the_setting(quoting, monkeypatch, how):
    if how == "planning":
        monkeypatch.setitem(c3_type_base._DEAL, "view", "planning")
    else:
        monkeypatch.setenv("SOWSMITH_C3_OURS_OUT", "off")
    b = _build(_quoting_atoms(), {})
    assert sorted(x["id"] for v in b["sections"].values() for x in v) == ["a1", "a2", "a3"]
    assert b["left_out"]["ours_in_quoting_view"] == []


def test_quote_site_roster_skips_ours(quoting, monkeypatch):
    monkeypatch.setattr("app.core.c3_brief._deal_kit", lambda a: False)
    atoms = [_atom(1, "physical_site", "Ship to: our warehouse"), _atom(2, "physical_site", "Site: 10 Main St")]
    for a in atoms:
        a.entity_keys = [f"site:{a.id}"]
        a.review_flags = []
    assert [s["site"] for s in c3_type_base.quote_sites(atoms)] == ["site:a2"]
