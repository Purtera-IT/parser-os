"""A head may hold a site back; it may never leave the deal with none, and never silently.

010171 priced `site_count 0` with `by: "none"` on a deal whose one address — "4250 Executive
Square, Suite 900, La Jolla, CA 92037" — sat in the corpus twice, typed `physical_site`, admitted,
in a customer's email. Asked the whole deal in one batch, `jdg:site_role` reads that line
`job_site` at 0.9562. The roster saw the compile's own read of it, and nothing anywhere said a
place had been dropped: the count just was zero.

Two rules, both from the 010452 lesson that a quote pricing no place is never the right answer:
a veto that would empty the roster does not apply, and either way the held-back line is named.
"""
from __future__ import annotations

import types

import pytest

from app.core import c3_type_base as c3


def _atom(aid, text, keys, kind="physical_site", **kw):
    a = types.SimpleNamespace(id=aid, text=text, atom_type=kind, entity_keys=list(keys),
                              review_flags=[], value=None, artifact_id="art_1")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture(autouse=True)
def _clean_deal():
    c3._DEAL.clear()
    yield
    c3._DEAL.clear()


def _reads(**by_atom):
    c3._DEAL["reads"] = {aid: {"jdg:site_role": list(v)} for aid, v in by_atom.items()}


def test_a_veto_that_would_empty_the_roster_does_not_apply():
    a = _atom("atm_1", "4250 Executive Square, Suite 900, La Jolla, CA 92037", ["site:la_jolla_ca_92037"])
    _reads(atm_1=["vendor_or_billing_address", 0.97])
    got = c3.quote_sites([a])
    assert len(got) == 1, "the deal's only place must survive a veto"
    held = got[0]["role_head_held_back"]
    assert held[0]["answer"] == "vendor_or_billing_address"
    assert held[0]["p"] == 0.97
    assert "no site" in held[0]["why"]


def test_a_veto_still_applies_when_another_site_stands():
    good = _atom("atm_good", "Work site: 22 Mill Rd, Akron, OH 44301", ["site:akron"])
    vendor = _atom("atm_vend", "Remit to 1 Lockbox Plaza, Chicago, IL 60603", ["site:chicago"])
    _reads(atm_vend=["vendor_or_billing_address", 0.97])
    got = c3.quote_sites([good, vendor])
    assert [s["site"] for s in got] == ["site:akron"]


def test_a_held_back_line_is_always_named_even_when_the_veto_stood():
    good = _atom("atm_good", "Work site: 22 Mill Rd, Akron, OH 44301", ["site:akron"])
    vendor = _atom("atm_vend", "Remit to 1 Lockbox Plaza, Chicago, IL 60603", ["site:chicago"])
    _reads(atm_vend=["vendor_or_billing_address", 0.97])
    rec: list[dict] = []
    c3.quote_sites([good, vendor], record=rec)
    assert len(rec) == 1
    assert rec[0]["atom_id"] == "atm_vend"
    assert rec[0]["head"] == "jdg:site_role"
    assert "somebody else's address" in rec[0]["why"]


def test_job_site_and_absent_are_never_held_back():
    a = _atom("atm_1", "Work site: 22 Mill Rd, Akron, OH 44301", ["site:akron"])
    b = _atom("atm_2", "Work site: 8 Pine St, Reno, NV 89501", ["site:reno"])
    _reads(atm_1=["job_site", 0.99], atm_2=["_absent", 0.99])
    rec: list[dict] = []
    got = c3.quote_sites([a, b], record=rec)
    assert sorted(s["site"] for s in got) == ["site:akron", "site:reno"]
    assert rec == []


def test_a_veto_below_the_bar_never_held_anything_back():
    a = _atom("atm_1", "4250 Executive Square, La Jolla, CA 92037", ["site:la_jolla"])
    _reads(atm_1=["vendor_or_billing_address", 0.6])
    rec: list[dict] = []
    got = c3.quote_sites([a], record=rec)
    assert len(got) == 1 and rec == []


def test_several_held_back_lines_all_come_back_when_nothing_else_stands():
    a = _atom("atm_1", "1 Lockbox Plaza, Chicago, IL 60603", ["site:chicago"])
    b = _atom("atm_2", "99 Billing Way, Reno, NV 89501", ["site:reno"])
    _reads(atm_1=["vendor_or_billing_address", 0.95], atm_2=["vendor_or_billing_address", 0.93])
    rec: list[dict] = []
    got = c3.quote_sites([a, b], record=rec)
    assert len(got) == 2
    assert {r["atom_id"] for r in rec} == {"atm_1", "atm_2"}
    assert all("no site" in r["why"] for r in rec)
