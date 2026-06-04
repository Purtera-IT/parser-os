"""The correction gate: a fix earns its way in, or it's refused.

Two outcomes matter most:

* a clean, well-scoped correction clears all nine invariants and
  ``gated_confirm`` writes it to the live store;
* a correction that would disturb a control case fails invariant D (no
  collateral) and ``gated_confirm`` REFUSES it — the live store is untouched.

Resolution runs on throwaway twins, so neither outcome can leak a half-tested
rule into production. The deterministic concept embedder lets us construct an
exact collateral case: a control probe that shares the corrected concept must
be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.complaint_intake import Complaint, intake
from app.core.correction_eval import Probe, evaluate, gated_confirm
from app.core.decide import DecisionScope
from app.core.feedback_store import SCOPE_DEAL, SCOPE_GLOBAL, FeedbackStore

CANDS = ["job_site", "vendor_or_billing_address"]

_CONCEPTS = {
    "vendor": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "site": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "other": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def _detect(t: str) -> str:
    tl = t.lower()
    if any(k in tl for k in ("purtera", "alpharetta", "letterhead", "billing")):
        return "vendor"
    if any(k in tl for k in ("santa fe", "field rd", "memorial", "job site")):
        return "site"
    return "other"


def _fake_embed(texts):
    return np.array([_CONCEPTS[_detect(t)] for t in texts], dtype=np.float32)


def _store():
    return FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)


def _probe(text, expect, scope=None):
    return Probe(text, "physical_site", expect, CANDS, scope or DecisionScope())


def _purtera_complaint(scope=SCOPE_GLOBAL, scope_key=""):
    return Complaint(
        relation="physical_site",
        desired_verdict="vendor_or_billing_address",
        text="PurTera LLC, 11720 Amber Park Dr, Alpharetta GA 30009",
        scope=scope,
        scope_key=scope_key,
    )


# ── a clean fix clears all nine invariants and commits ──────────────────

def test_clean_correction_passes_all_invariants_and_commits():
    s = _store()
    res = intake(_purtera_complaint(), store=s)
    committed, report = gated_confirm(
        s,
        res,
        fix_probes=[_probe("PurTera, Alpharetta GA", "vendor_or_billing_address")],
        generalization_probes=[
            _probe("PurTera letterhead office", "vendor_or_billing_address"),
            _probe("billing address, Alpharetta", "vendor_or_billing_address"),
        ],
        # Control: a real site concept that must stay silent.
        collateral_probes=[_probe("Memorial Hospital, Santa Fe", None)],
    )
    assert committed is True
    assert report.passed, report.summary()
    assert report.failed() == []
    # Names every invariant A–I.
    assert {r.name for r in report.results} == set("ABCDEFGHI")
    # And it really landed in the live store.
    assert any(c.relation == "physical_site" for c in s.all_corrections())


# ── a fix that disturbs a control is refused (invariant D) ──────────────

def test_collateral_damage_is_refused_and_store_untouched():
    s = _store()
    res = intake(_purtera_complaint(), store=s)
    # This control shares the corrected (vendor) concept, so the rule WILL
    # change it from silent→vendor — exactly the over-generalization the gate
    # must catch.
    committed, report = gated_confirm(
        s,
        res,
        fix_probes=[_probe("PurTera, Alpharetta GA", "vendor_or_billing_address")],
        collateral_probes=[_probe("our corporate billing letterhead", None)],
    )
    assert committed is False
    assert "D" in report.failed()
    # Nothing was written to the live store.
    assert s.all_corrections(active_only=False) == []


# ── evaluate() runs on twins — never touches the live store ─────────────

def test_evaluate_does_not_mutate_live_store():
    s = _store()
    res = intake(_purtera_complaint(), store=s)
    evaluate(
        s,
        res.proposed_correction,
        fix_probes=[_probe("PurTera, Alpharetta GA", "vendor_or_billing_address")],
        collateral_probes=[_probe("Santa Fe site", None)],
        resolution=res,
    )
    assert s.all_corrections(active_only=False) == []


# ── deal-scoped fix: invariant H proves no cross-deal leak ──────────────

def test_deal_scoped_correction_does_not_leak_across_deals():
    s = _store()
    res = intake(_purtera_complaint(scope=SCOPE_DEAL, scope_key="deal-x"), store=s)
    committed, report = gated_confirm(
        s,
        res,
        fix_probes=[
            _probe(
                "PurTera, Alpharetta GA",
                "vendor_or_billing_address",
                scope=DecisionScope(deal_id="deal-x"),
            )
        ],
    )
    assert committed is True
    h = next(r for r in report.results if r.name == "H")
    assert h.passed, h.detail


# ── invariant B fails loudly when the fix simply doesn't take ───────────

def test_fix_efficacy_failure_blocks_commit():
    s = _store()
    res = intake(_purtera_complaint(), store=s)
    # Ask the probe to expect a verdict the correction does not produce.
    committed, report = gated_confirm(
        s,
        res,
        fix_probes=[_probe("PurTera, Alpharetta GA", "job_site")],
    )
    assert committed is False
    assert "B" in report.failed()
    assert s.all_corrections(active_only=False) == []


def test_report_summary_is_compact():
    s = _store()
    res = intake(_purtera_complaint(), store=s)
    _, report = gated_confirm(
        s,
        res,
        fix_probes=[_probe("PurTera, Alpharetta GA", "vendor_or_billing_address")],
    )
    summ = report.summary()
    assert summ.startswith("PASS") or summ.startswith("FAIL")
    assert "A" in summ and "I" in summ


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
