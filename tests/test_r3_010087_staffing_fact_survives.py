"""010087: a short staffing fact on a call survives the substance gate.

"So that'll be the only region that won't have a stack coordinator." was
dropped as OCR debris: "that'll", "won't" and "coordinator" are not in the
debris judge's wordlist, so the line read as half non-words. Speech is not
OCR, a contraction is a word, an agent noun reads as its verb, and a line
stating a site / quantity / staffing fact is held to the bar of real debris.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.atom_substance_gate import drop_unreadable_text
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.text_quality import is_unreadable

VICTOR = "So that'll be the only region that won't have a stack coordinator."


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _atom(text: str, art: ArtifactType) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"a{abs(hash((text, art))) % 10**8}", project_id="p", artifact_id="a", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={}, entity_keys=[],
        source_refs=[SourceRef(id="s", artifact_id="a", artifact_type=art, filename="f",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.meeting_note, confidence=0.4,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def test_contractions_and_agent_nouns_are_words() -> None:
    assert not is_unreadable(VICTOR)
    assert not is_unreadable("we don't have a stack coordinator")
    assert not is_unreadable("They're short two techs and won't have a coordinator there.")
    assert is_unreadable("Tes aks wilenur tht projetcompen mee egutements")


def test_gate_keeps_speech_and_stated_facts() -> None:
    odd = "the integrattor wilenur sez the dock is shut"  # misheard words, still speech
    speech = _atom(odd, ArtifactType.transcript)
    debris = _atom("Tes aks wilenur tht projetcompen mee egutements", ArtifactType.pdf)
    kept, dropped = drop_unreadable_text([speech, debris])
    assert speech in kept and debris in dropped


def test_victor_line_survives_a_compile(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    turns = [("Victor", "Okay so let's go through the site list for the rollout."), ("Victor", VICTOR)]
    payload = {"schema": "fireflies.transcript.utterances.v1", "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
               "title": "Rollout sync",
               "utterances": [{"speaker": s, "text": t, "start": float(i * 5), "index": i}
                              for i, (s, t) in enumerate(turns)]}
    (tmp_path / "010087-fireflies-01M1KWDX5FJCYZ5BAF5JC8W0QC-transcript.json").write_text(json.dumps(payload))
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    by = {a.raw_text: a for a in r.atoms}
    assert VICTOR in by, sorted(by)
    assert str(getattr(by[VICTOR].atom_type, "value", by[VICTOR].atom_type)) != "deal_metadata"
