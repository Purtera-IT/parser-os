"""Base learning: the lexical gates become the store, semantically.

These tests prove the bootstrap mechanism (deterministically, no network):

* harvesting imports the LIVE gate constants — no duplicated lists — and seeds
  one kNN prototype per phrase;
* the seeded store reproduces the gate's verdicts AND generalizes to a
  paraphrase the regex never contained (the whole point);
* ``verify_gate`` measures reproduction + collateral and yields the
  ``safe_to_delete`` criterion the regex deletion is gated on;
* seeding is idempotent (content-stable ids) and the live store is opt-in
  (untouched until bootstrap runs).

The deterministic embedder maps text to concept axes so paraphrases of the same
concept collide (cosine 1.0) and different concepts are orthogonal — the
behavior the real qwen3-embedding model approximates.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decide import DecisionScope
from app.core.feedback_store import FeedbackStore
from app.core.gate_bootstrap import (
    RELATION_IS_SITE,
    SITE_CANDIDATES,
    GateSpec,
    bootstrap_default_store,
    bootstrap_store,
    default_gate_specs,
    seed_corrections_from_spec,
    verify_gate,
)

# concept axes: a facility word, a non-site fragment, a vendor brand, other.
_AXES = {
    "facility": np.array([1.0, 0, 0, 0], dtype=np.float32),
    "nonsite": np.array([0, 1.0, 0, 0], dtype=np.float32),
    "vendor": np.array([0, 0, 1.0, 0], dtype=np.float32),
    "other": np.array([0, 0, 0, 1.0], dtype=np.float32),
}

_FACILITY = ("school", "hospital", "warehouse", "campus", "clinic", "academy",
             "depot", "stadium", "library", "datacenter", "fieldhouse",
             "elementary", "lower school", "prek 5 building")
_NONSITE = ("facility", "bid opening", "consumption", "energy costs", "covid",
            "quality assurance", "go live", "phase i", "level ii", "first aid",
            "the building", "this site", "department of defense")
_VENDOR = ("cisco", "aruba", "genetec", "milestone", "verkada", "axis")


def _concept(t: str) -> str:
    tl = t.lower()
    if any(k in tl for k in _VENDOR):
        return "vendor"
    if any(k in tl for k in _FACILITY):
        return "facility"
    if any(k in tl for k in _NONSITE):
        return "nonsite"
    return "other"


def _fake_embed(texts):
    return np.array([_AXES[_concept(t)] for t in texts], dtype=np.float32)


def _store():
    return FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)


def _resolve(store, text, candidates=SITE_CANDIDATES, relation=RELATION_IS_SITE):
    return store.resolve(
        relation=relation,
        text=text,
        candidates=candidates,
        context="",
        scope=DecisionScope(),
        instruction="",
        relations=None,
    )


# A small, self-contained spec (so the test doesn't depend on the exact live
# frozenset contents, which evolve).
def _site_spec() -> GateSpec:
    return GateSpec(
        id="site",
        relation=RELATION_IS_SITE,
        candidates=SITE_CANDIDATES,
        groups={
            "site": ["school", "hospital", "warehouse", "campus"],
            "not_site": ["the facility", "bid opening", "energy costs"],
        },
        source="test",
    )


# ── harvest imports live constants, seeds one prototype per phrase ──────

def test_default_specs_import_live_constants():
    specs = {s.id: s for s in default_gate_specs()}
    assert "site" in specs
    site = specs["site"]
    # The real blocklist/anchor lists are non-trivial.
    assert len(site.groups["site"]) > 20
    assert len(site.groups["not_site"]) > 100  # _OBVIOUS_NON_SITES alone is ~300
    assert site.relation == RELATION_IS_SITE


def test_seed_one_correction_per_phrase_with_stable_ids():
    spec = _site_spec()
    corrs = seed_corrections_from_spec(spec)
    assert len(corrs) == 7  # 4 site + 3 not_site
    ids = [c.id for c in corrs]
    assert len(set(ids)) == 7  # unique
    # Stable + content-addressed (human-readable slug + content hash suffix).
    assert any(i.startswith("gate:site:site:school_") for i in ids)
    # Re-seeding yields identical ids (idempotent replace).
    assert {c.id for c in seed_corrections_from_spec(spec)} == set(ids)


# ── the payoff: reproduces the gate AND generalizes past the regex ──────

def test_seeded_store_reproduces_and_generalizes():
    s = _store()
    bootstrap_store(s, [_site_spec()])

    # Reproduces a seeded entry.
    d = _resolve(s, "warehouse")
    assert d is not None and d.verdict == "site" and d.source == "store"

    # Generalizes: "elementary" / "lower school" are NOT in the seed list, but
    # embed to the facility concept — the regex would have missed them.
    for unseen in ("elementary", "lower school", "prek 5 building"):
        d = _resolve(s, unseen)
        assert d is not None and d.verdict == "site", unseen

    # A non-site fragment resolves to not_site, not site.
    d = _resolve(s, "bid opening")
    assert d is not None and d.verdict == "not_site"


def test_unrelated_text_stays_silent():
    s = _store()
    bootstrap_store(s, [_site_spec()])
    # "other" concept matches no prototype → store stays silent → decide()
    # would fall through to the LLM/lexical fallback. Never a guess.
    assert _resolve(s, "lorem ipsum dolor") is None


# ── verify_gate yields the deletion criterion ───────────────────────────

def test_verify_gate_reports_safe_to_delete_on_clean_reproduction():
    s = _store()
    spec = _site_spec()
    bootstrap_store(s, [spec])
    v = verify_gate(s, spec)
    assert v.total == 7
    assert v.misassigned == 0
    assert v.reproduction_rate == 1.0
    assert v.safe_to_delete is True
    assert "SAFE-TO-DELETE" in v.summary()


def test_verify_gate_generalization_probes():
    s = _store()
    spec = _site_spec()
    bootstrap_store(s, [spec])
    # Held-out paraphrases (not in the seed) — the real generalization test.
    v = verify_gate(
        s,
        spec,
        probes={
            "site": ["elementary", "lower school", "campus"],
            "not_site": ["energy costs", "consumption"],
        },
    )
    assert v.misassigned == 0
    assert v.reproduced == 5


def test_verify_gate_flags_collateral():
    s = _store()
    spec = _site_spec()
    bootstrap_store(s, [spec])
    # A probe that the gate would WRONGLY classify: feed a facility-concept word
    # but label it not_site → it resolves to "site" → misassigned (collateral).
    v = verify_gate(s, spec, probes={"not_site": ["hospital", "campus"]})
    assert v.misassigned == 2
    assert v.safe_to_delete is False
    assert "KEEP-REGEX" in v.summary()


# ── opt-in: the live store is untouched until bootstrap runs ────────────

def test_bootstrap_is_explicit_and_idempotent():
    s = _store()
    assert s.all_corrections(active_only=False) == []  # empty until asked
    n1 = bootstrap_store(s, [_site_spec()])
    n2 = bootstrap_store(s, [_site_spec()])
    assert n1 == n2 == 7
    # Idempotent: re-seeding replaces, never duplicates.
    assert len(s.all_corrections(active_only=False)) == 7


def test_bootstrap_default_store_skips_unreproducible_gates(monkeypatch):
    # The seeding criterion IS the deletion criterion: a gate the store can't
    # reproduce cleanly is skipped, not leaked as collateral-prone bare tokens.
    import app.core.gate_bootstrap as gb

    clean = _site_spec()  # orthogonal concepts → reproduces 7/7, 0 collateral
    # A gate whose two verdicts share one concept: both exemplars embed to the
    # facility axis, so resolution can't separate them → collateral → unsafe.
    dirty = GateSpec(
        id="dirty",
        relation="dirty_rel",
        candidates=["a", "b"],
        groups={"a": ["school"], "b": ["hospital"]},
        source="test",
    )
    monkeypatch.setattr(gb, "default_gate_specs", lambda: [clean, dirty])

    verified = bootstrap_default_store(":memory:", embed_fn=_fake_embed, verify=True)
    forced = bootstrap_default_store(":memory:", embed_fn=_fake_embed, verify=False)
    # verify=True drops the 2 dirty exemplars; verify=False seeds everything.
    assert forced - verified == 2


def test_offline_store_stays_silent_even_when_bootstrapped():
    s = FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: False)
    bootstrap_store(s, [_site_spec()])
    assert _resolve(s, "warehouse") is None  # safe fallback when embed endpoint down


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
