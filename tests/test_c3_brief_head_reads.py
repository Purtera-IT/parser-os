"""The brief carries what the heads actually said about a place line.

Answering "what did the head say about this line" used to need a probe built by
hand against the service, and a probe that sends different row fields is a
different read -- which is how a head came to be reported as disagreeing with
itself when the only difference was the payload. These lines' readings ride in
the envelope so the question is a read, not a probe.
"""
import types

from app.core.c3_brief import _stamp_head_reads


def _atom(aid, atom_type="scope_item", keys=()):
    return types.SimpleNamespace(id=aid, atom_type=atom_type, entity_keys=list(keys))


def test_a_place_line_carries_every_head_it_was_read_with():
    atoms = [_atom("a1", "physical_site", ["site:la_jolla_ca_92037"])]
    reads = {"a1": {"col:admission": ["drop", 0.833], "jdg:site_role": ["job_site", 0.956]}}
    brief = {}
    _stamp_head_reads(brief, atoms, reads)
    line = brief["head_reads"]["lines"]["a1"]
    assert line["why_here"] == "typed physical_site"
    assert line["heads"]["col:admission"] == ["drop", 0.833]
    assert line["heads"]["jdg:site_role"] == ["job_site", 0.956]


def test_a_line_with_only_a_site_key_is_carried_too():
    atoms = [_atom("a1", "scope_item", ["site:la_jolla"])]
    reads = {"a1": {"col:admission": ["keep", 0.9]}}
    brief = {}
    _stamp_head_reads(brief, atoms, reads)
    assert brief["head_reads"]["lines"]["a1"]["why_here"] == "carries a site: key"


def test_an_address_key_is_carried():
    atoms = [_atom("a1", "scope_item", ["address:1_main_st"])]
    brief = {}
    _stamp_head_reads(brief, atoms, {"a1": {"jdg:site_role": ["vendor_or_billing_address", 0.95]}})
    assert brief["head_reads"]["lines"]["a1"]["why_here"] == "carries an address: key"


def test_a_line_that_names_no_place_is_not_carried_by_default():
    atoms = [_atom("a1", "small_talk", [])]
    brief = {}
    _stamp_head_reads(brief, atoms, {"a1": {"col:admission": ["drop", 0.99]}})
    assert "head_reads" not in brief


def test_every_line_with_the_dump_switch(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_READS_DUMP", "1")
    atoms = [_atom("a1", "small_talk", []), _atom("a2", "scope_item", [])]
    brief = {}
    _stamp_head_reads(brief, atoms, {"a1": {"col:admission": ["drop", 0.9]},
                                     "a2": {"col:admission": ["keep", 0.9]}})
    assert brief["head_reads"]["every_line"] is True
    assert set(brief["head_reads"]["lines"]) == {"a1", "a2"}


def test_a_line_the_compile_never_read_is_not_invented():
    atoms = [_atom("a1", "physical_site", ["site:x"]), _atom("a2", "physical_site", ["site:y"])]
    brief = {}
    _stamp_head_reads(brief, atoms, {"a1": {"col:admission": ["keep", 0.9]}})
    assert set(brief["head_reads"]["lines"]) == {"a1"}
    assert brief["head_reads"]["candidates"] == 1


def test_no_reads_at_all_writes_nothing():
    brief = {}
    _stamp_head_reads(brief, [_atom("a1", "physical_site", ["site:x"])], {})
    assert brief == {}
