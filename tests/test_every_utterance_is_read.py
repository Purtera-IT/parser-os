"""Every substantive Fireflies utterance reaches the labeller -- as an atom,
or, when a dedup pass folds it, as a suppression entry that says so.

The speech collapse ran ahead of the semantic_dedup ledger snapshot, so a
folded utterance left no atom and no suppression entry. And it folded
untyped turns by word overlap: "the only region that won't have a stack
coordinator" vanished into a longer earlier turn sharing its words.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.compiler import compile_project

_UTTERANCES = [
    ("Victor", "Okay so let's go through the site list for the rollout."),
    ("Saga", "Yeah."),
    ("Saga", "The Dallas site is not ready, the floor is still being poured."),
    ("Saga", "could we swap to a nearby ready site"),
    ("Victor", "So Frankfurt is the only region that won't have a stack coordinator"),
    ("Victor", "the only region that won't have a stack coordinator"),
    ("Saga", "Send me NewBold's quote and the updated site count."),
    ("Victor", "Okay."),
    ("Saga", "Send me NewBold's quote and the updated site count please."),
]


def _compile(tmp_path: Path):
    payload = {
        "schema": "fireflies.transcript.utterances.v1",
        "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
        "title": "Rollout sync",
        "utterances": [
            {"speaker": s, "text": t, "start": float(i * 5), "index": i}
            for i, (s, t) in enumerate(_UTTERANCES)
        ],
    }
    (tmp_path / "010087-fireflies-01M1KWDX5FJCYZ5BAF5JC8W0QC-transcript.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)


def _text(a) -> str:
    if isinstance(a, dict):
        return str(a.get("raw_text") or "")
    return str(getattr(a, "raw_text", "") or "")


def test_untyped_utterances_are_not_folded_away(tmp_path: Path) -> None:
    r = _compile(tmp_path)
    texts = {_text(a) for a in r.atoms}
    assert "could we swap to a nearby ready site" in texts
    assert "the only region that won't have a stack coordinator" in texts
    assert "So Frankfurt is the only region that won't have a stack coordinator" in texts


def test_every_utterance_is_an_atom_or_a_suppression(tmp_path: Path) -> None:
    r = _compile(tmp_path)
    seen = {_text(a) for a in r.atoms} | {_text(a) for a in r.suppressed_atoms}
    for _speaker, text in _UTTERANCES:
        if text in {"Yeah.", "Okay."}:
            continue
        assert text in seen, f"utterance has neither an atom nor a suppression entry: {text!r}"
