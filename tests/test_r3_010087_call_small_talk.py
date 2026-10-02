"""010087 re-run: a call's small talk is typed small_talk, and a short
staffing fact survives the substance gate.

* 478 of the call's 717 lines came out ``deal_metadata``: "Yeah.", "Okay.",
  "Right, right." (gate-demoted turns) and the banter beside them. Small
  talk is now typed ``small_talk`` -- the labeler's reject type -- at the end
  of the compile, chatter, with deal_metadata kept as an alternative.
* Greetings and signatures were flagged chatter but typed deal_metadata;
  they take the same type.
* "So that'll be the only region that won't have a stack coordinator." was
  dropped by the substance gate as OCR debris: "that'll", "won't" and
  "coordinator" are not in its wordlist. Speech is not OCR, a contraction is
  a word, and a line stating a staffing fact is not filler.
"""
from __future__ import annotations

import json
from email.message import EmailMessage
from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.core.text_quality import is_unreadable


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


VICTOR = "So that'll be the only region that won't have a stack coordinator."
TURNS = [
    ("Victor", "Okay so let's go through the site list for the rollout."),
    ("Saga", "Yeah."),
    ("Victor", VICTOR),
    ("Victor", "we don't have a stack coordinator"),
    ("Saga", "Okay."),
    ("Saga", "Right, right."),
    ("Victor", "Hope you had a great 4th of July!"),
    ("Octavian", "Is the loading dock open on Saturdays?"),
    ("Saga", "Yes."),
]


def _t(a) -> str:
    return str(getattr(a.atom_type, "value", a.atom_type))


def _call(d: Path) -> None:
    payload = {"schema": "fireflies.transcript.utterances.v1", "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
               "title": "Rollout sync",
               "utterances": [{"speaker": s, "text": t, "start": float(i * 5), "index": i}
                              for i, (s, t) in enumerate(TURNS)]}
    (d / "010087-fireflies-01M1KWDX5FJCYZ5BAF5JC8W0QC-transcript.json").write_text(json.dumps(payload))


def test_call_small_talk_is_small_talk_and_the_staffing_fact_survives(tmp_path: Path) -> None:
    _call(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    by = {a.raw_text: a for a in r.atoms}
    for line in (VICTOR, "we don't have a stack coordinator"):
        assert line in by, (line, sorted(by))
        assert _t(by[line]) not in ("deal_metadata", "small_talk"), _t(by[line])
        assert "unreadable_ocr" not in by[line].review_flags
    for line in ("Yeah.", "Okay.", "Right, right.", "Hope you had a great 4th of July!"):
        a = by[line]
        assert _t(a) == "small_talk", (line, _t(a), a.review_flags)
        assert "chatter" in a.review_flags and a.value.get("chatter") is True
        assert "deal_metadata" in a.value.get("alt_atom_types", [])
    # "Yes." straight after a question may be its answer: not small talk.
    assert _t(by["Yes."]) != "small_talk"
    assert not any(_t(a) == "deal_metadata" and "chatter" in a.review_flags for a in r.atoms
                   if a.raw_text in {t for _, t in TURNS})


def test_email_greeting_and_signature_are_small_talk(tmp_path: Path) -> None:
    m = EmailMessage()
    m["From"] = "alec@vendor.example"
    m["To"] = "trent@purtera-it.com"
    m["Subject"] = "Install"
    m.set_content("Hi Trent,\n\nThe riser room is locked after 6pm and the super has the only key.\n\n"
                  "Thank you,\n\nAlec Burns\n")
    (tmp_path / "m.eml").write_bytes(bytes(m))
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    by = {a.raw_text: a for a in r.atoms}
    for line in ("Hi Trent,", "Thank you,"):
        assert line in by, sorted(by)
        assert _t(by[line]) == "small_talk", (line, _t(by[line]), by[line].value)
    assert _t(by["The riser room is locked after 6pm and the super has the only key."]) != "small_talk"


def test_ocr_debris_is_still_debris() -> None:
    assert not is_unreadable(VICTOR)
    assert not is_unreadable("They're short two techs and won't have a coordinator there.")
    assert is_unreadable("Tes aks wilenur tht projetcompen mee egutements")
