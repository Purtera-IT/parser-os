"""The purpulse labeler rebuilds decide-text in JavaScript from envelope atoms
(Platform-infra shared/atom-labeling.js ``decideText``). Human labels train on
that string, so it must equal what ``_atom_decide_text`` serves the heads.
Both sides assert the same vectors (copy in Platform-infra
shared/decide-text-vectors.json). Change both or neither.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.typed_atom_classifier import _atom_decide_text

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "decide_text_vectors.json").read_text(encoding="utf-8"))


def _as_evidence_atom(env_atom: dict) -> SimpleNamespace:
    # The envelope writes EvidenceAtom.value as ``structured`` and the first
    # source ref's locator as ``locator``.
    return SimpleNamespace(
        raw_text=env_atom.get("text", ""),
        value=env_atom.get("structured") or {},
        source_refs=[SimpleNamespace(locator=env_atom.get("locator") or {})],
    )


@pytest.mark.parametrize("vec", VECTORS, ids=[v["name"] for v in VECTORS])
def test_parser_decide_text_matches_shared_vector(vec, monkeypatch):
    monkeypatch.setenv("SOWSMITH_ATOM_BIND_HEADERS", "1")
    assert _atom_decide_text(_as_evidence_atom(vec["atom"])) == vec["expected"]
