"""Complaint intake: a PM's correction becomes a store rule — on confirm only.

These tests pin the contract the learning loop depends on:

* a complaint localizes to the offending atom across BOTH accepted and retained
  *suppressed* atoms, and the diagnosis names the stage that dropped it (the
  Phase 1 retention payoff — an omission is now pointable-at);
* ``intake()`` proposes but NEVER writes; the store is untouched until
  ``confirm()``;
* a confirmed correction is active and immediately resolves a *paraphrase* of
  the complaint (end-to-end: complaint → correction → store hit);
* guards: no exemplar can't be confirmed; reject discards; intake never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from app.core.complaint_intake import (
    KIND_WRONGLY_DROPPED,
    KIND_WRONGLY_KEPT,
    Complaint,
    confirm,
    intake,
    reject,
)
from app.core.decide import DecisionScope
from app.core.feedback_store import SCOPE_DEAL, SCOPE_GLOBAL, FeedbackStore


# ── lightweight atom + result stand-ins (intake only reads a few fields) ──
@dataclass
class _Atom:
    id: str
    raw_text: str = ""
    value: dict = field(default_factory=dict)
    atom_type: str = "physical_site"


@dataclass
class _Result:
    atoms: list[Any] = field(default_factory=list)
    suppressed_atoms: list[Any] = field(default_factory=list)


# deterministic concept embedder (same shape as test_feedback_store)
_CONCEPTS = {
    "vendor": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "site": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "other": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def _detect(t: str) -> str:
    tl = t.lower()
    if any(k in tl for k in ("purtera", "alpharetta", "letterhead")):
        return "vendor"
    if any(k in tl for k in ("santa fe", "field rd", "memorial")):
        return "site"
    return "other"


def _fake_embed(texts: list[str]) -> np.ndarray:
    return np.array([_CONCEPTS[_detect(t)] for t in texts], dtype=np.float32)


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)


# ── localization across accepted + suppressed ──────────────────────────

def test_localizes_to_suppressed_atom_and_names_stage():
    # An atom that site_geo_fallback wrongly dropped, retained by Phase 1.
    dropped = _Atom(
        id="atm_santafe",
        raw_text="location Santa Fe, NM 87506",
        value={
            "_suppression": {
                "stage": "site_geo_fallback",
                "reason": "vendor / selling-party letterhead address, not a job site",
            }
        },
    )
    result = _Result(atoms=[], suppressed_atoms=[dropped])
    c = Complaint(
        relation="physical_site",
        desired_verdict="job_site",
        text="Santa Fe, NM 87506",
        kind=KIND_WRONGLY_DROPPED,
    )
    res = intake(c, result=result)
    assert len(res.localized) == 1
    loc = res.localized[0]
    assert loc.bucket == "suppressed"
    assert loc.suppression_stage == "site_geo_fallback"
    assert "site_geo_fallback" in res.diagnosis
    assert "SUPPRESSED" in res.diagnosis


def test_localizes_to_accepted_atom():
    kept = _Atom(id="atm_purtera", raw_text="PurTera LLC, Alpharetta GA 30009")
    result = _Result(atoms=[kept], suppressed_atoms=[])
    c = Complaint(
        relation="physical_site",
        desired_verdict="vendor_or_billing_address",
        text="PurTera",
        kind=KIND_WRONGLY_KEPT,
    )
    res = intake(c, result=result)
    assert res.localized and res.localized[0].bucket == "accepted"
    assert "ACCEPTED" in res.diagnosis


def test_localizes_by_atom_id_when_given():
    a1 = _Atom(id="a1", raw_text="something")
    a2 = _Atom(id="a2", raw_text="PurTera LLC")
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            atom_id="a2",
        ),
        result=_Result(atoms=[a1, a2]),
    )
    assert len(res.localized) == 1 and res.localized[0].atom_id == "a2"


def test_intake_without_result_still_proposes():
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera, Alpharetta GA",
        )
    )
    assert res.localized == []
    assert res.proposed_correction.exemplars == ["PurTera, Alpharetta GA"]
    assert "Could not localize" in res.diagnosis


# ── intake proposes but does not commit ─────────────────────────────────

def test_intake_does_not_write_to_store():
    s = _store()
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera, Alpharetta GA",
        ),
        store=s,
    )
    assert res.proposed_correction.status == "proposed"
    assert res.committed is False
    assert s.all_corrections(active_only=False) == []  # nothing written


# ── end-to-end: complaint → confirm → store resolves a paraphrase ───────

def test_confirm_makes_correction_fire_on_paraphrase():
    s = _store()
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera LLC, 11720 Amber Park Dr, Alpharetta GA 30009",
            scope=SCOPE_GLOBAL,
        ),
        store=s,
    )
    c = confirm(s, res)
    assert c.status == "active"
    assert res.committed is True

    # The store now resolves an unseen paraphrase to the corrected verdict.
    d = s.resolve(
        relation="physical_site",
        text="PurTera letterhead, Alpharetta GA",
        candidates=["job_site", "vendor_or_billing_address"],
        context="",
        scope=DecisionScope(),
        instruction="",
        relations=None,
    )
    assert d is not None
    assert d.verdict == "vendor_or_billing_address"
    assert d.source == "store"
    assert d.correction_id == c.id


def test_confirm_respects_deal_scope():
    s = _store()
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="job_site",
            text="Santa Fe, NM 87506",
            scope=SCOPE_DEAL,
            scope_key="deal-yonah",
        ),
        store=s,
    )
    confirm(s, res)
    # Fires inside the deal...
    inside = s.resolve(
        relation="physical_site",
        text="Santa Fe NM",
        candidates=["job_site", "vendor_or_billing_address"],
        context="",
        scope=DecisionScope(deal_id="deal-yonah"),
        instruction="",
        relations=None,
    )
    assert inside is not None and inside.verdict == "job_site"
    # ...but not in a different deal.
    outside = s.resolve(
        relation="physical_site",
        text="Santa Fe NM",
        candidates=["job_site", "vendor_or_billing_address"],
        context="",
        scope=DecisionScope(deal_id="deal-other"),
        instruction="",
        relations=None,
    )
    assert outside is None


# ── guards ──────────────────────────────────────────────────────────────

def test_confirm_is_idempotent():
    s = _store()
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera, Alpharetta GA",
        ),
        store=s,
    )
    c1 = confirm(s, res)
    c2 = confirm(s, res)
    assert c1 is c2
    assert len(s.all_corrections(active_only=False)) == 1


def test_cannot_confirm_without_exemplar():
    s = _store()
    # No text and no localizable atom → no exemplar to embed.
    res = intake(
        Complaint(relation="physical_site", desired_verdict="vendor_or_billing_address")
    )
    with pytest.raises(ValueError):
        confirm(s, res)
    assert s.all_corrections(active_only=False) == []


def test_reject_discards_and_writes_nothing():
    s = _store()
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera, Alpharetta GA",
        ),
        store=s,
    )
    reject(res)
    assert res.proposed_correction.status == "rejected"
    assert s.all_corrections(active_only=False) == []


def test_preview_is_human_readable():
    res = intake(
        Complaint(
            relation="physical_site",
            desired_verdict="vendor_or_billing_address",
            text="PurTera, Alpharetta GA",
            scope=SCOPE_GLOBAL,
        )
    )
    assert "PROPOSED" in res.preview
    assert "vendor_or_billing_address" in res.preview
    assert "global" in res.preview


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
