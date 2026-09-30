# -*- coding: utf-8 -*-
"""A taught answer may delete a job site. A model's guess may only ask.

`suppress_vendor_sites` exists to stop the vendor's OWN letterhead address
being filed as a place where work happens. It asked `decide()` and deleted any
site the answer called `vendor_or_billing_address` at >= 0.6 confidence --
whether that answer came from a stored correction someone made, or from a model
guessing.

Live 000113 (Columbus AFB premise wiring), dev compile a257441f, with models
ON -- the only configuration where this path runs at all:

    "Customer:: Address: | Park Place Tech LLC: 6500 Hollister Ave 210"
    "Customer:: City/St: | Park Place Tech LLC: Goleta, Ca 93117"

both deleted, and NO surviving atom in the envelope mentioned Park Place,
Hollister or Goleta -- in raw_text or in value. The address is not on
`vendor_site_ban`'s list. It is precisely the failure the code's own comment
predicts: "the party address counts as the second site, the LLM is asked to
pick the vendor among two, and it can pick the customer's HQ".

The audit could not see this for months because every audit tool sets
SOWSMITH_DISABLE_LLM, which switches this path off entirely.

Why flag instead of delete: there IS a site-role head for this question, and
the way it learns is by someone judging a proposal. A deletion teaches nothing
-- the atom is gone, so nobody is ever asked, and the same guess repeats on
every compile forever. A kept-and-flagged atom reaches the labelling workspace,
gets a verdict, and that verdict lands in the store -- after which the source is
"store" and the deletion happens for free, everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core import site_geo_fallback as SGF


@dataclass
class _Decision:
    verdict: str | None
    confidence: float
    source: str
    correction_id: str | None = None


class _Site:
    def __init__(self, aid, text, value=None):
        self.id = aid
        self.atom_type = "physical_site"
        self.raw_text = text
        self.text = text
        self.value = dict(value or {})
        self.confidence = 0.8
        self.source_refs = []
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def _run(monkeypatch, decision, sites):
    """Drive suppress_vendor_sites with a canned decision.

    `decide` and the ban list are imported INSIDE the function, so they are
    patched at their own modules rather than on this one.
    """
    from app.core import decide as decide_mod
    from app.core import party_address_veto, vendor_site_ban

    # The verdict applies to the PARK PLACE address only. Answering the same
    # thing for every site trips the stage's "never strip the deal down to zero
    # sites" guard, which would make these tests pass without exercising
    # anything -- the first version of this fixture did exactly that.
    def _decide(kind, addr, *a, **k):
        if "Hollister" in str(addr):
            return decision
        return _Decision("job_site", 0.9, "store")

    monkeypatch.setattr(decide_mod, "decide", _decide)
    monkeypatch.setattr(vendor_site_ban, "drop_banned_vendor_physical_sites",
                        lambda atoms: (atoms, 0))
    monkeypatch.setattr(party_address_veto, "veto_party_page_sites",
                        lambda atoms: None)
    monkeypatch.setattr(SGF, "_is_roster_site", lambda a: False)
    monkeypatch.setattr(SGF, "_site_address_text", lambda a: (a.raw_text, ""))
    return SGF.suppress_vendor_sites(list(sites), project_id="000113")


def _pair():
    return [
        _Site("s1", "Park Place Tech LLC: 6500 Hollister Ave 210",
              {"street_address": "6500 Hollister Ave 210"}),
        _Site("s2", "Columbus AFB, MS Building 704",
              {"street_address": "680 Seventh St"}),
    ]


def test_a_model_guess_keeps_the_site_and_asks(monkeypatch) -> None:
    sites = _pair()
    out, dropped = _run(monkeypatch,
                        _Decision("vendor_or_billing_address", 0.75, "llm"),
                        sites)
    assert dropped == 0, "a guess deleted a site"
    assert len(out) == 2, "the customer's address left the compile"
    assert SGF.VENDOR_SUSPECT_FLAG in sites[0].review_flags
    proposal = sites[0].value["vendor_role_proposal"]
    assert proposal["source"] == "llm" and proposal["confidence"] == 0.75


def test_a_taught_answer_still_deletes(monkeypatch) -> None:
    """The stage must keep working for what it was built for."""
    sites = _pair()
    out, dropped = _run(monkeypatch,
                        _Decision("vendor_or_billing_address", 0.9, "store"),
                        sites)
    assert dropped == 1
    assert [a.id for a in out] == ["s2"]


def test_a_fallback_is_a_guess_too(monkeypatch) -> None:
    sites = _pair()
    out, dropped = _run(monkeypatch,
                        _Decision("vendor_or_billing_address", 0.99, "fallback"),
                        sites)
    assert dropped == 0 and len(out) == 2


def test_a_job_site_verdict_changes_nothing(monkeypatch) -> None:
    sites = _pair()
    out, dropped = _run(monkeypatch, _Decision("job_site", 0.9, "llm"), sites)
    assert dropped == 0 and len(out) == 2
    assert sites[0].review_flags == []


def test_a_low_confidence_taught_answer_does_not_delete(monkeypatch) -> None:
    sites = _pair()
    out, dropped = _run(monkeypatch,
                        _Decision("vendor_or_billing_address", 0.3, "store"),
                        sites)
    assert dropped == 0 and len(out) == 2


def test_the_flag_is_idempotent_across_compiles(monkeypatch) -> None:
    sites = _pair()
    for _ in range(3):
        _run(monkeypatch, _Decision("vendor_or_billing_address", 0.8, "llm"), sites)
    assert sites[0].review_flags.count(SGF.VENDOR_SUSPECT_FLAG) == 1
