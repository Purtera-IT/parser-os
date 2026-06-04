"""Integration: the neural head firing *through* FeedbackStore.resolve().

Proves the wiring, not just the head in isolation:
  * a relation with a real boundary (>=2 verdicts, enough exemplars) is decided
    by the learned head — source="store", rationale cites "neural head";
  * an in-distribution query on EITHER side is decided without the LLM
    (resolve returns a Decision, so decide() never reaches the model);
  * a novel (OOD) query makes the head abstain AND the cosine path miss, so
    resolve() returns None → decide() routes it to the LLM (the hard case);
  * a single-class relation (PurTera) does NOT build a head — the legacy
    cosine path still decides it, behavior unchanged.
"""

from __future__ import annotations

import numpy as np

from app.core.decide import DecisionScope
from app.core.feedback_store import (
    SCOPE_GLOBAL,
    Correction,
    FeedbackStore,
)

_D = 32


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic embedder: a big shared 'corporate address' component
    (dim0) that makes raw cosine conflate the classes, a small role signal
    (dim1: vendor=+1 / site=-1 / neither=0), and per-text jitter for identity."""
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = t.lower()
        shared = 0.0 if "nowhere" in tl else 3.0
        if any(w in tl for w in ("vendor", "billing", "letterhead", "purtera")):
            sig = +1.0
        elif any(w in tl for w in ("site", "project", "install", "premises")):
            sig = -1.0
        else:
            sig = 0.0
        out[i, 0] = shared
        out[i, 1] = sig
        h = abs(hash(t))
        out[i, 2 + (h % (_D - 2))] = 0.05  # unique-ish jitter
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 1e-9, norms, 1.0)


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)


def _boundary_corrections() -> list[Correction]:
    vendor = Correction(
        id="c_vendor", relation="physical_site",
        verdict="vendor_or_billing_address", scope=SCOPE_GLOBAL,
        exemplars=[
            "vendor billing address on the letterhead",
            "PurTera vendor billing office",
            "service provider letterhead billing address",
            "vendor remittance billing address",
        ],
    )
    site = Correction(
        id="c_site", relation="physical_site",
        verdict="job_site", scope=SCOPE_GLOBAL,
        exemplars=[
            "project install site premises",
            "customer job site for the install",
            "site where the project work happens",
            "field install site premises address",
        ],
    )
    return [vendor, site]


def test_head_decides_vendor_side_without_llm():
    s = _store()
    for c in _boundary_corrections():
        s.add(c)
    d = s.resolve(
        relation="physical_site",
        text="the vendor billing letterhead address",
        candidates=["job_site", "vendor_or_billing_address"],
        context="", scope=DecisionScope(), instruction="role?", relations=None,
    )
    assert d is not None
    assert d.verdict == "vendor_or_billing_address"
    assert d.source == "store"
    assert "neural head" in d.rationale
    assert d.correction_id == "c_vendor"


def test_head_decides_jobsite_side_without_llm():
    s = _store()
    for c in _boundary_corrections():
        s.add(c)
    d = s.resolve(
        relation="physical_site",
        text="the customer project install site premises",
        candidates=["job_site", "vendor_or_billing_address"],
        context="", scope=DecisionScope(), instruction="role?", relations=None,
    )
    assert d is not None
    assert d.verdict == "job_site"          # confident NEGATIVE handled too
    assert d.source == "store"


def test_ood_query_abstains_and_routes_to_llm():
    s = _store()
    for c in _boundary_corrections():
        s.add(c)
    # "nowhere" zeroes the shared component → unlike any trained address.
    d = s.resolve(
        relation="physical_site",
        text="nowhere abstract token unrelated",
        candidates=["job_site", "vendor_or_billing_address"],
        context="", scope=DecisionScope(), instruction="role?", relations=None,
    )
    assert d is None                         # head abstains + cosine misses → LLM


def test_single_class_relation_builds_no_head():
    s = _store()
    s.add(Correction(
        id="c_only", relation="physical_site",
        verdict="vendor_or_billing_address", scope=SCOPE_GLOBAL,
        exemplars=["vendor billing letterhead", "PurTera vendor billing"],
    ))
    head = s._relation_head(
        "physical_site", {"vendor_or_billing_address", "job_site"}, DecisionScope()
    )
    assert head is None                      # one verdict → no boundary → no head
