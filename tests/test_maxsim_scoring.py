"""Max-similarity (late-interaction-lite) scoring for the feedback store.

The default scorer collapses each correction's exemplars to a single MEAN
prototype, then takes cosine(query, mean). That blurs a *heterogeneous*
correction: when a PM lumps several unrelated surface forms under one verdict
(the exact shape ``gate_bootstrap`` seeds from a frozenset — Cisco / Genetec /
Securitas / Honeywell all as "vendor"), the mean sits in the middle of nowhere.
A query that matches ONE exemplar strongly gets dragged below threshold by the
other four, and — worse — the blurred mean can accidentally clear threshold for
a form that matches *nothing* taught.

``SOWSMITH_NEURAL_MAXSIM`` swaps cosine-to-mean for the MAX cosine over the
correction's individual exemplars. A single strong exemplar match fires; a
heterogeneous correction can no longer dilute itself or spuriously match.

These tests are hermetic: a deterministic content-addressed embedder gives each
distinct exemplar its own near-orthogonal basis vector, so the mean of N
exemplars has cosine ~1/sqrt(N) to any one of them (below 0.82 for N>=2) while
max-sim is ~1.0 on an exact match. No network.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decide import DecisionScope
from app.core.feedback_store import Correction, FeedbackStore

_D = 256


def _basis_embed(texts: list[str]) -> np.ndarray:
    """Each distinct text -> its own one-hot basis vector (near-orthogonal to
    every other text). Identical text -> identical vector (cosine 1.0)."""
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, abs(hash(t.lower().strip())) % _D] = 1.0
    return out


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=_basis_embed, reachable_fn=lambda: True)


_REL = "is_vendor_token"
_CANDS = ["vendor"]
# Five unrelated brands lumped under one verdict (the frozenset shape).
_EXEMPLARS = ["cisco", "genetec", "aruba", "securitas", "honeywell"]


def _resolve(store: FeedbackStore, text: str):
    return store.resolve(
        relation=_REL, text=text, candidates=_CANDS,
        context="", scope=DecisionScope(), instruction="", relations=None,
    )


def test_mean_path_blurs_heterogeneous_correction(monkeypatch):
    """Default (mean prototype): an exact exemplar match scores ~1/sqrt(5)
    ≈ 0.447, below the 0.82 threshold, so the correction does NOT fire even on
    a token it was literally taught. This is the blur the flag fixes."""
    monkeypatch.delenv("SOWSMITH_NEURAL_MAXSIM", raising=False)
    store = _store()
    store.add(Correction(
        id="c_vendor", relation=_REL, verdict="vendor",
        exemplars=list(_EXEMPLARS), created_by="pm",
    ))
    # "securitas" is one of the taught exemplars — yet the mean blurs it away.
    assert _resolve(store, "securitas") is None


def test_maxsim_recovers_the_taught_exemplar(monkeypatch):
    """With max-sim on, the SAME heterogeneous correction fires on the taught
    exemplar — the strong single-exemplar match is no longer diluted."""
    monkeypatch.setenv("SOWSMITH_NEURAL_MAXSIM", "1")
    store = _store()
    store.add(Correction(
        id="c_vendor", relation=_REL, verdict="vendor",
        exemplars=list(_EXEMPLARS), created_by="pm",
    ))
    d = _resolve(store, "securitas")
    assert d is not None and d.verdict == "vendor"
    assert d.confidence >= 0.82
    assert "max-sim" in (d.rationale or "")


def test_maxsim_does_not_fire_on_untaught_token(monkeypatch):
    """Max-sim stays precise: a token near NONE of the exemplars does not fire
    (no accidental match from a blurred mean)."""
    monkeypatch.setenv("SOWSMITH_NEURAL_MAXSIM", "1")
    store = _store()
    store.add(Correction(
        id="c_vendor", relation=_REL, verdict="vendor",
        exemplars=list(_EXEMPLARS), created_by="pm",
    ))
    assert _resolve(store, "verkada") is None


def test_single_exemplar_identical_under_both_paths(monkeypatch):
    """For a single-exemplar correction, mean == that one vector, so max-sim and
    the mean path are identical — the flag only changes heterogeneous cases."""
    for flag in ("", "1"):
        if flag:
            monkeypatch.setenv("SOWSMITH_NEURAL_MAXSIM", flag)
        else:
            monkeypatch.delenv("SOWSMITH_NEURAL_MAXSIM", raising=False)
        store = _store()
        store.add(Correction(
            id="c1", relation=_REL, verdict="vendor",
            exemplars=["cisco"], created_by="pm",
        ))
        d = _resolve(store, "cisco")
        assert d is not None and d.verdict == "vendor"


def test_flag_off_is_default_mean_behavior(monkeypatch):
    """Sanity: unset flag == mean path (byte-identical default pipeline)."""
    monkeypatch.delenv("SOWSMITH_NEURAL_MAXSIM", raising=False)
    store = _store()
    assert store._enable_maxsim is False
