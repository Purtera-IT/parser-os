"""The feedback store: a PM's correction, resolved by meaning not keywords.

These tests pin the store's contract that ``decide()`` depends on:

* a confident, in-candidate, in-scope hit returns ``source="store"`` citing the
  correction id — and it fires on a *paraphrase*, not a string match
  (generalization);
* anything unsure returns ``None`` so ``decide()`` falls through to the LLM —
  dissimilar text, offline endpoint, a verdict outside the caller's candidate
  set, a disabled correction, or a threshold the query can't clear;
* narrowest scope wins (a deal correction overrides a global one for that deal);
* it never raises and never persists embeddings.

The embedder and reachability probe are injected, so resolution is exercised
deterministically with no network. The fake maps text to one of a few
orthonormal concept axes: paraphrases of the same concept embed identically
(cosine 1.0), unrelated text is orthogonal (cosine 0.0). That is exactly the
semantic behavior the real qwen3-embedding model approximates — it lets these
tests prove the *resolution logic* without depending on embedding quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decide import DecisionScope
from app.core.feedback_store import (
    SCOPE_DEAL,
    SCOPE_GLOBAL,
    Correction,
    FeedbackStore,
    seed_default_corrections,
)

CANDS = ["job_site", "vendor_or_billing_address"]

# ── deterministic concept embedder (stands in for qwen3-embedding) ──────
_CONCEPTS = {
    "vendor": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "site": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "other": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


def _detect(t: str) -> str:
    tl = t.lower()
    if any(k in tl for k in ("purtera", "letterhead", "billing", "alpharetta")):
        return "vendor"
    if any(k in tl for k in ("santa fe", "field rd", "job site", "customer location")):
        return "site"
    return "other"


def _fake_embed(texts: list[str]) -> np.ndarray:
    return np.array([_CONCEPTS[_detect(t)] for t in texts], dtype=np.float32)


def _store(*, reachable: bool = True) -> FeedbackStore:
    return FeedbackStore(
        ":memory:", embed_fn=_fake_embed, reachable_fn=lambda: reachable
    )


def _purtera_correction(
    cid: str = "corr_purtera",
    *,
    verdict: str = "vendor_or_billing_address",
    scope: str = SCOPE_GLOBAL,
    scope_key: str = "",
    threshold: float = 0.82,
    status: str = "active",
) -> Correction:
    return Correction(
        id=cid,
        relation="physical_site",
        verdict=verdict,
        scope=scope,
        scope_key=scope_key,
        exemplars=["PurTera LLC, 11720 Amber Park Dr, Alpharetta GA 30009"],
        threshold=threshold,
        status=status,
        instruction="Classify the role of this address.",
    )


def _resolve(store: FeedbackStore, text: str, *, scope=None, candidates=CANDS):
    return store.resolve(
        relation="physical_site",
        text=text,
        candidates=candidates,
        context="",
        scope=scope or DecisionScope(),
        instruction="Classify the role of this address.",
        relations=None,
    )


# ── generalization: fires on a paraphrase, cites the correction ─────────

def test_hit_on_paraphrase_cites_correction():
    s = _store()
    s.add(_purtera_correction())
    # Different wording than the stored exemplar — same meaning.
    d = _resolve(s, "PurTera, Alpharetta GA")
    assert d is not None
    assert d.verdict == "vendor_or_billing_address"
    assert d.source == "store"
    assert d.correction_id == "corr_purtera"
    assert d.confidence >= 0.82


def test_hit_increments_hit_count():
    s = _store()
    s.add(_purtera_correction())
    _resolve(s, "PurTera, Alpharetta GA")
    assert s.get("corr_purtera").hit_count == 1


# ── undecided cases: store returns None, decide() will fall through ─────

def test_dissimilar_text_no_hit():
    s = _store()
    s.add(_purtera_correction())
    assert _resolve(s, "location Santa Fe, NM 87506") is None


def test_offline_endpoint_no_hit():
    s = _store(reachable=False)
    s.add(_purtera_correction())
    assert _resolve(s, "PurTera, Alpharetta GA") is None


def test_verdict_outside_candidates_no_hit():
    s = _store()
    s.add(_purtera_correction())
    # Caller doesn't offer the correction's verdict as an option.
    assert _resolve(s, "PurTera, Alpharetta GA", candidates=["job_site"]) is None


def test_disabled_correction_no_hit():
    s = _store()
    s.add(_purtera_correction(status="active"))
    s.set_status("corr_purtera", "disabled")
    assert _resolve(s, "PurTera, Alpharetta GA") is None


def test_threshold_gates_a_weak_match():
    s = _store()
    # Concept matches (cosine 1.0) but an impossible threshold blocks it.
    s.add(_purtera_correction(threshold=1.01))
    assert _resolve(s, "PurTera, Alpharetta GA") is None


def test_empty_inputs_no_hit():
    s = _store()
    s.add(_purtera_correction())
    assert _resolve(s, "") is None
    assert _resolve(s, "PurTera", candidates=[]) is None


def test_relation_must_match():
    s = _store()
    c = _purtera_correction()
    c.relation = "atom_type"  # governs a different decision family
    s.add(c)
    assert _resolve(s, "PurTera, Alpharetta GA") is None


# ── scope precedence: deal overrides global for that deal only ──────────

def test_deal_scope_overrides_global():
    s = _store()
    s.add(_purtera_correction("corr_global", verdict="vendor_or_billing_address"))
    s.add(
        _purtera_correction(
            "corr_deal",
            verdict="job_site",
            scope=SCOPE_DEAL,
            scope_key="deal-x",
        )
    )
    # Within deal-x, the deal correction wins.
    d = _resolve(s, "PurTera, Alpharetta GA", scope=DecisionScope(deal_id="deal-x"))
    assert d.correction_id == "corr_deal"
    assert d.verdict == "job_site"
    # A different deal falls back to the global correction.
    d2 = _resolve(s, "PurTera, Alpharetta GA", scope=DecisionScope(deal_id="deal-y"))
    assert d2.correction_id == "corr_global"
    assert d2.verdict == "vendor_or_billing_address"


# ── resilience: a broken embedder never breaks resolution ───────────────

def test_broken_embedder_returns_none():
    def _boom(_texts):
        raise RuntimeError("embed endpoint exploded")

    s = FeedbackStore(":memory:", embed_fn=_boom, reachable_fn=lambda: True)
    s.add(_purtera_correction())
    assert _resolve(s, "PurTera, Alpharetta GA") is None


# ── persistence: a correction is a durable, inspectable row ─────────────

def test_correction_row_roundtrip():
    s = _store()
    c = _purtera_correction()
    c.relations = {"owner": "selling_party"}
    s.add(c)
    got = s.get("corr_purtera")
    assert got is not None
    assert got.verdict == "vendor_or_billing_address"
    assert got.exemplars == c.exemplars
    assert got.relations == {"owner": "selling_party"}


def test_all_corrections_active_only():
    s = _store()
    s.add(_purtera_correction("a"))
    s.add(_purtera_correction("b"))
    s.set_status("b", "disabled")
    active = {c.id for c in s.all_corrections(active_only=True)}
    assert active == {"a"}
    every = {c.id for c in s.all_corrections(active_only=False)}
    assert every == {"a", "b"}


# ── few-shot: nearest exemplars prime the LLM when there's no firm hit ──

def test_few_shot_returns_nearest_exemplars():
    s = _store()
    s.add(_purtera_correction())
    ex = s.few_shot(
        relation="physical_site",
        text="PurTera, Alpharetta GA",
        scope=DecisionScope(),
        k=3,
    )
    assert ex and ex[0]["verdict"] == "vendor_or_billing_address"
    assert "PurTera" in ex[0]["text"]


def test_few_shot_offline_is_empty():
    s = _store(reachable=False)
    s.add(_purtera_correction())
    assert s.few_shot(
        relation="physical_site", text="PurTera", scope=DecisionScope()
    ) == []


# ── the seeded global PurTera rule resolves end to end ──────────────────

def test_seeded_purtera_rule_resolves():
    s = _store()
    n = seed_default_corrections(s)
    assert n >= 1
    d = _resolve(s, "PurTera LLC letterhead, Alpharetta GA 30009")
    assert d is not None
    assert d.verdict == "vendor_or_billing_address"
    assert d.source == "store"


def test_seed_is_idempotent():
    s = _store()
    seed_default_corrections(s)
    seed_default_corrections(s)
    # Re-seeding replaces, never duplicates, the global rule.
    purtera = [
        c for c in s.all_corrections(active_only=False)
        if c.id == "global_purtera_self_address"
    ]
    assert len(purtera) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
