"""Plain-English rule compiler (upgrade #7).

A PM types one English sentence; an INJECTED LLM synthesizer turns it into a
structured proposal; the compiler runs that proposal through the SAME nine-
invariant verify-gate every other correction must clear. A rule commits only on
a clean pass — never on the sentence's say-so alone.

Hermetic: a deterministic marker-clustering embedder makes the exemplar and its
paraphrases share one vector (so invariant C — generalization — can pass) while
control cases land on orthogonal vectors (so invariant D — no collateral — can
be exercised both ways). The synthesizer is a plain Python fake. No network, no
model, no Ollama.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decide import DecisionScope
from app.core.feedback_store import SCOPE_DEAL, FeedbackStore
from app.core.plain_rule_compiler import RuleProposal, compile_rule

_D = 64
# Texts containing one of these markers cluster onto a shared basis vector, so an
# exemplar and its paraphrases (all carrying the same marker) embed identically →
# cosine 1.0 → a taught rule fires on all of them. Marker-free text is unique.
_MARKERS = ["purtera", "pricebook"]


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = t.lower()
        placed = False
        for j, m in enumerate(_MARKERS):
            if m in tl:
                out[i, j] = 1.0
                placed = True
        if not placed:
            h = abs(hash(tl))
            out[i, len(_MARKERS) + (h % (_D - len(_MARKERS)))] = 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)


def _resolve(store: FeedbackStore, text: str, *, scope: DecisionScope | None = None):
    return store.resolve(
        relation="physical_site",
        text=text,
        candidates=["keep", "drop"],
        context="",
        scope=scope or DecisionScope(),
        instruction="",
        relations=None,
    )


# A well-formed proposal: "PurTera is our company, never a site." The exemplar
# and paraphrases all carry the 'purtera' marker (cluster together); the controls
# are real sites carrying no marker (must stay untouched).
def _clean_synth(sentence: str) -> dict:
    return {
        "relation": "physical_site",
        "verdict": "drop",
        "candidates": ["keep", "drop"],
        "exemplar": "PurTera headquarters",
        "paraphrases": [
            "our company PurTera",
            "PurTera Inc main office",
        ],
        "controls": [
            "Atlanta data center MDF closet",
            "Santa Fe distribution warehouse",
        ],
        "scope": "global",
        "scope_key": "",
    }


def test_clean_rule_commits_and_fires():
    """A well-formed rule clears all nine invariants, commits, and afterwards the
    store decides the exemplar (and a paraphrase) WITHOUT the LLM, citing the
    learned correction's id."""
    store = _store()
    out = compile_rule(
        "PurTera is our own company, it is never a site.",
        store,
        synthesize=_clean_synth,
        created_by="pm",
    )
    assert out.committed is True
    assert out.report.passed is True, out.report.summary()

    # The committed rule now fires on the exemplar — store-decided, with id.
    d = _resolve(store, "PurTera headquarters")
    assert d is not None and d.verdict == "drop"
    assert d.source == "store"
    assert d.correction_id == out.proposal_correction_id

    # And it generalizes to a paraphrase (same marker cluster).
    g = _resolve(store, "our company PurTera")
    assert g is not None and g.verdict == "drop"

    # A real site (a control) is untouched — the store stays silent there.
    assert _resolve(store, "Atlanta data center MDF closet") is None


def test_collateral_failure_refuses_rule():
    """If a control case would be swept up by the rule (invariant D fails), the
    gate REFUSES it: nothing is committed and the store stays silent everywhere."""
    def _greedy_synth(sentence: str) -> dict:
        p = _clean_synth(sentence)
        # Poison a control with the SAME marker as the exemplar → the rule would
        # fire on it too → collateral damage → invariant D must fail.
        p["controls"] = ["PurTera adjacent real site", "Santa Fe warehouse"]
        return p

    store = _store()
    out = compile_rule(
        "PurTera is our company.", store, synthesize=_greedy_synth
    )
    assert out.committed is False
    assert "D" in out.report.failed()
    # Refused → the store learned nothing; even the exemplar is undecided.
    assert _resolve(store, "PurTera headquarters") is None


def test_malformed_proposal_raises():
    """A proposal that could never gate-pass fails loudly at synthesis time."""
    store = _store()

    with pytest.raises(ValueError):
        compile_rule("x", store, synthesize=lambda s: {"relation": "physical_site", "verdict": "drop"})  # no exemplar

    with pytest.raises(ValueError):
        compile_rule(
            "x", store,
            synthesize=lambda s: {
                "relation": "physical_site",
                "verdict": "banana",  # not in candidates
                "candidates": ["keep", "drop"],
                "exemplar": "PurTera HQ",
            },
        )

    with pytest.raises(ValueError):
        compile_rule("   ", store, synthesize=_clean_synth)  # empty sentence


def test_deal_scoped_rule_does_not_leak():
    """A deal-scoped rule commits and fires inside its deal, but never leaks to
    another deal (invariant H), proving scope precedence holds."""
    def _deal_synth(sentence: str) -> dict:
        p = _clean_synth(sentence)
        p["scope"] = "deal"
        p["scope_key"] = "deal-123"
        return p

    store = _store()
    out = compile_rule(
        "On this deal, PurTera is not a site.", store, synthesize=_deal_synth
    )
    assert out.committed is True
    assert out.proposal.scope == SCOPE_DEAL

    # Fires inside its own deal.
    here = _resolve(store, "PurTera headquarters", scope=DecisionScope(deal_id="deal-123"))
    assert here is not None and here.verdict == "drop"

    # Silent in a different deal — no cross-deal leak.
    elsewhere = _resolve(
        store, "PurTera headquarters", scope=DecisionScope(deal_id="deal-999")
    )
    assert elsewhere is None


def test_default_scope_inheritance():
    """When the synthesizer omits a scope, the caller's default is applied."""
    def _no_scope_synth(sentence: str) -> dict:
        p = _clean_synth(sentence)
        p.pop("scope", None)
        p.pop("scope_key", None)
        return p

    store = _store()
    out = compile_rule(
        "PurTera is ours.",
        store,
        synthesize=_no_scope_synth,
        default_scope=SCOPE_DEAL,
        default_scope_key="deal-abc",
    )
    assert out.proposal.scope == SCOPE_DEAL
    assert out.proposal.scope_key == "deal-abc"


def test_proposal_from_raw_recovers_empty_candidates():
    """A verdict with no explicit candidate list is recoverable: the verdict is
    itself a valid candidate."""
    p = RuleProposal.from_raw(
        {"relation": "physical_site", "verdict": "drop", "exemplar": "PurTera HQ"}
    )
    assert p.candidates == ["drop"]
    assert p.verdict == "drop"
