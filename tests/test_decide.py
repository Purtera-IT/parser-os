"""The decide() chokepoint: one resolver for every meaning-judgment.

Precedence is store -> llm -> safe fallback. Phase 2 ships with no store, so
decide() must be a transparent pass-through to semantic_role.classify_role
(byte-identical to calling it directly). Phase 3 wires a store; these tests pin
the precedence and the safe-fallback contract so that wiring can't regress it.
"""

from __future__ import annotations

import app.core.semantic_role as semantic_role
from app.core.decide import Decision, DecisionScope, decide, resolve_or, set_store

CANDS = ["job_site", "vendor_or_billing_address"]


def setup_function(_):
    set_store(None)
    semantic_role.reset_reachability()


def teardown_function(_):
    set_store(None)


# ── Phase 2: no store → transparent pass-through to the LLM primitive ──

def test_llm_pass_through(monkeypatch):
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        return ("vendor_or_billing_address", 0.91)
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    d = decide("physical_site", "PurTera LLC, Alpharetta GA", CANDS, instruction="x")
    assert d.verdict == "vendor_or_billing_address"
    assert d.confidence == 0.91
    assert d.source == "llm"
    assert d.correction_id is None


def test_fallback_when_model_undecided(monkeypatch):
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        return (None, 0.0)
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    d = decide("physical_site", "Santa Fe, NM 87506", CANDS, instruction="x")
    assert d.verdict is None
    assert d.source == "fallback"
    assert d.confidence == 0.0


def test_empty_input_is_fallback():
    assert decide("physical_site", "", CANDS, instruction="x").verdict is None
    assert decide("physical_site", "text", [], instruction="x").verdict is None


def test_model_override_is_forwarded(monkeypatch):
    seen = {}
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        seen["model"] = model
        return ("job_site", 0.8)
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    decide("physical_site", "x", CANDS, instruction="x", model="qwen3:14b")
    assert seen["model"] == "qwen3:14b"


# ── Phase 3 forward-contract: a confident store hit pre-empts the LLM ──

class _FakeStore:
    def __init__(self, hit: Decision | None, examples=None):
        self._hit = hit
        self._examples = examples or []
        self.few_shot_called = False

    def resolve(self, **kw):
        return self._hit

    def few_shot(self, **kw):
        self.few_shot_called = True
        return self._examples


def test_store_hit_preempts_llm(monkeypatch):
    def _clf(*a, **k):  # would be wrong; must NOT be called
        raise AssertionError("LLM called despite confident store hit")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    set_store(_FakeStore(Decision(
        verdict="vendor_or_billing_address", confidence=0.97,
        source="store", correction_id="corr_purtera",
    )))
    d = decide("physical_site", "PurTera, Alpharetta GA", CANDS, instruction="x")
    assert d.source == "store"
    assert d.verdict == "vendor_or_billing_address"
    assert d.correction_id == "corr_purtera"


def test_store_miss_falls_through_to_llm_with_fewshot(monkeypatch):
    captured = {}
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        captured["instruction"] = instruction
        return ("job_site", 0.85)
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    store = _FakeStore(None, examples=[{"text": "123 Main St (customer)", "verdict": "job_site"}])
    set_store(store)
    d = decide("physical_site", "456 Field Rd", CANDS, instruction="Classify role.")
    assert d.source == "llm"
    assert store.few_shot_called  # store consulted for priming
    assert "Worked examples" in captured["instruction"]  # few-shot injected


def test_store_exception_never_breaks_decide(monkeypatch):
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        return ("job_site", 0.8)
    monkeypatch.setattr(semantic_role, "classify_role", _clf)

    class _Boom:
        def resolve(self, **kw):
            raise RuntimeError("store down")
        def few_shot(self, **kw):
            raise RuntimeError("store down")
    set_store(_Boom())
    d = decide("physical_site", "x", CANDS, instruction="x")
    assert d.verdict == "job_site"  # gracefully fell through to LLM
    assert d.source == "llm"


def test_scope_default():
    s = DecisionScope()
    assert s.deal_id == "" and s.pack == ""


# ── store-fronts-regex seam: decide(llm=False) + resolve_or ─────────────

def test_llm_false_skips_model_entirely(monkeypatch):
    def _clf(*a, **k):  # must NOT be called when llm=False
        raise AssertionError("LLM called despite llm=False")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    # No store → store undecided → safe fallback, never the LLM.
    d = decide("physical_site", "456 Field Rd", CANDS, instruction="x", llm=False)
    assert d.verdict is None
    assert d.source == "fallback"


def test_llm_false_still_honors_confident_store_hit(monkeypatch):
    def _clf(*a, **k):
        raise AssertionError("LLM called despite confident store hit")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    set_store(_FakeStore(Decision(
        verdict="vendor_or_billing_address", confidence=0.97,
        source="store", correction_id="corr_purtera",
    )))
    d = decide("physical_site", "PurTera, Alpharetta GA", CANDS, instruction="x", llm=False)
    assert d.source == "store"
    assert d.verdict == "vendor_or_billing_address"


def test_resolve_or_returns_lexical_when_no_store(monkeypatch):
    def _clf(*a, **k):
        raise AssertionError("resolve_or must never call the LLM")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    # No store wired → byte-identical to the caller's own regex verdict.
    verdict, decision = resolve_or(
        "physical_site", "456 Field Rd", CANDS, lexical="job_site"
    )
    assert verdict == "job_site"
    assert decision is None


def test_resolve_or_store_overrides_lexical(monkeypatch):
    def _clf(*a, **k):
        raise AssertionError("resolve_or must never call the LLM")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    set_store(_FakeStore(Decision(
        verdict="vendor_or_billing_address", confidence=0.96,
        source="store", correction_id="corr_purtera",
    )))
    # The regex thinks it's a site; the store's PM correction overrides it.
    verdict, decision = resolve_or(
        "physical_site", "PurTera, Alpharetta GA", CANDS, lexical="job_site"
    )
    assert verdict == "vendor_or_billing_address"
    assert decision is not None and decision.correction_id == "corr_purtera"


def test_resolve_or_keeps_lexical_when_store_silent(monkeypatch):
    def _clf(*a, **k):
        raise AssertionError("resolve_or must never call the LLM")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    set_store(_FakeStore(None))  # store has nothing confident to say
    verdict, decision = resolve_or(
        "physical_site", "456 Field Rd", CANDS, lexical="job_site"
    )
    assert verdict == "job_site"
    assert decision is None


def test_resolve_or_passes_through_none_lexical(monkeypatch):
    def _clf(*a, **k):
        raise AssertionError("resolve_or must never call the LLM")
    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    set_store(_FakeStore(None))
    # Caller's own gate was undecided too → undecided stays undecided.
    verdict, decision = resolve_or(
        "physical_site", "ambiguous", CANDS, lexical=None
    )
    assert verdict is None
    assert decision is None
