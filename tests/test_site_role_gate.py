"""Universal ghost-rejection gate for site keys via the decide() store.

``semantic_site_role_drops`` is the universal replacement for the deal-specific
``_is_obvious_non_site`` denylist. A PM teaches the ROLE of a roster value
presentationally (an escort contact / an MDF-IDF closet / a work window is NOT a
site) and the store drops the same and structurally-similar ghosts on every
deal — without a regex or a hardcoded place name.

The gate is **store-only and one-sided** by design:
  * only a CONFIDENT store reject ({site_attribute, not_a_site}) drops a key;
  * a bare site CODE the store has never seen as canonical ABSTAINS → KEEP;
  * the LLM is never consulted, so it can't guess a drop on a context-free code.

These tests are hermetic: a deterministic content-addressed embedder gives each
distinct text its own near-orthogonal vector (cosine 1.0 only on an exact taught
exemplar), so a ghost equal to a taught value fires while an untaught site code
abstains. No network. The LLM tier is forced to abstain so only the store path
is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.semantic_role as semantic_role
from app.core.decide import set_store
from app.core.entity_resolution import semantic_site_role_drops
from app.core.feedback_store import Correction, FeedbackStore

_D = 128
_REL = "site_candidate_role"


def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower().strip()))
        out[i, h % _D] = 1.0
        out[i, (h // _D) % _D] += 0.5
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force the LLM tier to abstain so ONLY a store hit can drop — the gate is
    store-only anyway, but this guards against a regression that re-enables it."""
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: (None, 0.0))
    yield
    set_store(None)


def _store_with(*corrections: Correction) -> FeedbackStore:
    s = FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)
    for c in corrections:
        s.add(c)
    return s


# Single exemplar: with the deterministic embedder, distinct exemplars are
# near-orthogonal, so a multi-exemplar mean prototype would blur below threshold
# (that mean-pool effect is covered by test_maxsim_scoring). One exemplar gives a
# clean cosine 1.0 on an exact match so this file tests the GATE's drop/keep
# logic. Semantic generalization across paraphrases is proven in the live harness.
_ESCORT = Correction(
    id="r_esc", relation=_REL, verdict="site_attribute", created_by="pm",
    exemplars=["optbot facilities"],
)
# Real site keys (bare code slugs) — never taught as canonical, so the store
# abstains on them and they must survive.
_REAL = {"site:atl_hq_01", "site:atl_air_03"}
# A ghost whose phrase equals a taught attribute exemplar.
_GHOST = "site:optbot_facilities"


def test_flag_off_is_noop(monkeypatch):
    # The gate is ON by default now (the learned path fronts a trimmed denylist),
    # so DISABLING requires an explicit "0"/"false". With it off, no drops fire.
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", "0")
    set_store(_store_with(_ESCORT))
    assert semantic_site_role_drops(_REAL | {_GHOST}) == set()


def test_default_unset_runs_gate(monkeypatch):
    """Unset env → gate RUNS (ON by default): the taught ghost drops."""
    monkeypatch.delenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", raising=False)
    set_store(_store_with(_ESCORT))
    drops = semantic_site_role_drops(_REAL | {_GHOST})
    assert _GHOST in drops
    assert drops & _REAL == set()


def test_no_store_is_noop(monkeypatch):
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", "1")
    set_store(None)
    assert semantic_site_role_drops(_REAL | {_GHOST}) == set()


def test_taught_ghost_dropped_real_sites_kept(monkeypatch):
    """The taught attribute ghost drops; the untaught real site codes survive
    (store abstains on them) — zero false kills."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", "1")
    set_store(_store_with(_ESCORT))
    drops = semantic_site_role_drops(_REAL | {_GHOST})
    assert _GHOST in drops
    assert drops & _REAL == set()  # no real site dropped


def test_canonical_verdict_is_kept(monkeypatch):
    """A key the store classifies canonical_site is never dropped (one-sided)."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", "1")
    canon = Correction(
        id="r_can", relation=_REL, verdict="canonical_site", created_by="pm",
        exemplars=["atl hq 01"],
    )
    set_store(_store_with(_ESCORT, canon))
    drops = semantic_site_role_drops({"site:atl_hq_01", _GHOST})
    assert "site:atl_hq_01" not in drops
    assert _GHOST in drops


def test_non_site_keys_ignored(monkeypatch):
    """Only ``site:`` keys are considered; other entity keys pass through."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_ROLE_GATE", "1")
    set_store(_store_with(_ESCORT))
    drops = semantic_site_role_drops({"customer:acme", "stakeholder:jane_doe"})
    assert drops == set()
