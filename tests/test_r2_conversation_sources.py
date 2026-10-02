"""Round-2 fixes for conversation sources: email chains, HubSpot notes and
Fireflies calls, re-parsed from real deals.

Each test mirrors what the labeler saw:

* a Fireflies call whose turns were all atoms but none typed;
* real questions on a call taken out of the atom stream as "answered" or
  "not actionable" (010087: devices per school, swap to a ready site, no
  stack coordinator);
* "Hope you had a great 4th of July!" typed deal_metadata context;
* a "Let's go!!" credited to the message it answered, and one "Hi Megan,"
  per reply that quoted the message it opened;
* a stakeholder fold that kept the contact row without the email (010246);
* "Well, ..." read as a promise, so pleasantries could never be chatter.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _type(a) -> str:
    t = a.get("atom_type") if isinstance(a, dict) else getattr(a, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def _text(a) -> str:
    return str((a.get("raw_text") if isinstance(a, dict) else getattr(a, "raw_text", "")) or "")


def _flags(a) -> list[str]:
    return list((a.get("review_flags") if isinstance(a, dict) else getattr(a, "review_flags", None)) or [])


def _value(a) -> dict:
    v = a.get("value") if isinstance(a, dict) else getattr(a, "value", None)
    return v if isinstance(v, dict) else {}


# ---------------------------------------------------------------------------
# 1. Every Fireflies utterance gets a type
# ---------------------------------------------------------------------------

_UTTERANCES = [
    ("Victor", "Okay so let's go through the site list for the rollout."),
    ("Saga", "Yeah."),
    ("Octavian", "How many devices per school are we looking at?"),
    ("Saga", "The Dallas site is not ready, the floor is still being poured."),
    ("Saga", "could we swap to a nearby ready site"),
    ("Saga", "Can we swap to a nearby ready site?"),
    ("Victor", "There is no stack coordinator."),
    ("Victor", "So Frankfurt is the only region that won't have a stack coordinator"),
    ("Victor", "the only region that won't have a stack coordinator"),
    ("Victor", "Hope you had a great 4th of July!"),
    ("Victor", "We'll schedule the techs once the floor is done."),
    ("Saga", "Okay."),
]


def _write_call(directory: Path) -> Path:
    payload = {
        "schema": "fireflies.transcript.utterances.v1",
        "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
        "title": "Rollout sync",
        "utterances": [
            {"speaker": s, "text": t, "start": float(i * 5), "index": i}
            for i, (s, t) in enumerate(_UTTERANCES)
        ],
    }
    p = directory / "010087-fireflies-01M1KWDX5FJCYZ5BAF5JC8W0QC-transcript.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_every_utterance_is_typed_at_parse(tmp_path: Path) -> None:
    from app.core.utterance_typing import FALLBACK_TYPED_FLAG
    from app.parsers.transcript_parser import TranscriptParser

    atoms = TranscriptParser().parse_artifact("p", "art_ff", _write_call(tmp_path))
    turns = [a for a in atoms if (a.value or {}).get("kind") != "transcript_header"
             and a.atom_type != AtomType.physical_site]
    assert {a.raw_text for a in turns} >= {t for _, t in _UTTERANCES}
    assert not [a.raw_text for a in turns if a.atom_type == AtomType.raw_utterance]
    by_text = {a.raw_text: a for a in turns}
    # No pattern fired: a fallback type, flagged as a guess, low confidence.
    dallas = by_text["The Dallas site is not ready, the floor is still being poured."]
    assert FALLBACK_TYPED_FLAG in dallas.review_flags and dallas.confidence <= 0.45
    # A question with its question mark lost is still a question.
    assert by_text["could we swap to a nearby ready site"].atom_type == AtomType.open_question
    # A pleasantry is an admission-chatter atom, not scope.
    hope = by_text["Hope you had a great 4th of July!"]
    assert hope.atom_type == AtomType.deal_metadata
    assert "chatter" in hope.review_flags and "admission_regex" in hope.review_flags


def test_every_utterance_is_a_typed_atom_after_compile(tmp_path: Path) -> None:
    _write_call(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    by_text = {_text(a): a for a in r.atoms}
    for _speaker, text in _UTTERANCES:
        assert text in by_text, f"utterance has no atom: {text!r}"
        assert _type(by_text[text]) not in ("", "raw_utterance"), f"untyped: {text!r}"
    # 010087: real asks and the staffing constraint keep a substantive type.
    assert _type(by_text["How many devices per school are we looking at?"]) == "open_question"
    assert _type(by_text["Can we swap to a nearby ready site?"]) == "open_question"
    assert "transcript_smalltalk_demoted" not in _flags(by_text["There is no stack coordinator."])
    assert _type(by_text["There is no stack coordinator."]) != "deal_metadata"
    # The PM's commitment stays a commitment.
    sched = by_text["We'll schedule the techs once the floor is done."]
    assert _type(sched) == "action_item"
    assert "chatter" not in _flags(sched)
    # A fallback guess never anchors a packet.
    from app.core.utterance_typing import is_untyped_speech

    fallback_ids = {str(getattr(a, "id", "")) for a in r.atoms if is_untyped_speech(a)}
    assert fallback_ids
    for p in r.packets:
        anchors = set(getattr(p, "governing_atom_ids", None) or [])
        assert not (anchors & fallback_ids)


def test_demoted_small_talk_is_typed_not_raw() -> None:
    from app.core.atom_substance_gate import demote_transcript_smalltalk

    a = EvidenceAtom(
        id="atm_x", project_id="p", artifact_id="ff", atom_type=AtomType.open_question,
        raw_text="And they play Youngstown State, right?", normalized_text="x",
        value={"text": "x"}, entity_keys=[],
        source_refs=[SourceRef(id="s", artifact_id="ff", artifact_type=ArtifactType.transcript,
                               filename="call.json", extraction_method="t", parser_version="t",
                               locator={"utterance_index": 1})],
        authority_class=AuthorityClass.meeting_note, confidence=0.74,
        review_status=ReviewStatus.needs_review, parser_version="t",
    )
    demote_transcript_smalltalk([a])
    assert a.atom_type == AtomType.deal_metadata
    assert "chatter" in a.review_flags
    assert any(r["key"] == "small_talk" for r in a.value["reads"])


# ---------------------------------------------------------------------------
# 2. open_question_quality_filter
# ---------------------------------------------------------------------------

def _q(text: str, keys: list[str], rid: str, atype=AtomType.open_question) -> EvidenceAtom:
    return EvidenceAtom(
        id=rid, project_id="p", artifact_id="a", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value={"text": text}, entity_keys=keys,
        source_refs=[SourceRef(id="s" + rid, artifact_id="a", artifact_type=ArtifactType.email,
                               filename="a.eml", extraction_method="x", parser_version="x", locator={})],
        authority_class=AuthorityClass.customer_current_authored, confidence=0.8,
        review_status=ReviewStatus.needs_review, parser_version="x",
    )


def test_a_device_line_does_not_answer_how_many_devices_per_school() -> None:
    from app.core.open_question_resolution import (
        filter_unhelpful_open_questions,
        is_answered_question,
        resolve_open_questions,
    )

    q = _q("How many devices per school?", ["device:chromebook"], "atm_q")
    fact = _q("Chromebook carts for the classrooms", ["device:chromebook"], "atm_f", AtomType.scope_item)
    stream = [fact, q]
    resolve_open_questions(stream)
    assert not is_answered_question(q)
    kept, dropped = filter_unhelpful_open_questions(stream)
    assert q in kept and q not in dropped
    # A count does answer it.
    q2 = _q("How many devices per school?", ["device:chromebook", "quantity:chromebook"], "atm_q2")
    fact2 = _q("40 Chromebooks per school", ["device:chromebook", "quantity:chromebook"], "atm_f2",
               AtomType.quantity)
    resolve_open_questions([fact2, q2])
    assert is_answered_question(q2)


def test_filtered_questions_stay_atoms(tmp_path: Path) -> None:
    body = (
        "Hi Eddie,\n\n"
        "We need 40 Cat6 drops on the second floor.\n\n"
        "Eddie, is there anything else you need in order to start getting the price?\n\n"
        "Thanks,\nPatrick\n"
    )
    (tmp_path / "ask.eml").write_text(
        "From: Patrick <patrick@purtera-it.com>\nTo: Eddie <eddie@cust.com>\nSubject: Drops\n"
        "Date: Mon, 07 Jul 2025 09:00:00 -0400\nContent-Type: text/plain; charset=utf-8\n\n" + body,
        encoding="utf-8",
    )
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    q = "Eddie, is there anything else you need in order to start getting the price?"
    hits = [a for a in r.atoms if _text(a) == q and _type(a) == "open_question"]
    assert hits, "the filtered question must still be an atom"
    assert all("not_pm_actionable_question" in _flags(a) for a in hits)
    assert not [a for a in r.suppressed_atoms if _text(a) == q and _type(a) == "open_question"]


# ---------------------------------------------------------------------------
# 3. Pleasantries are chatter, not deal_metadata context
# ---------------------------------------------------------------------------

def test_sentence_kind_reads_a_holiday_as_banter_not_a_figure() -> None:
    from app.core.sentences import sentence_kind

    assert sentence_kind("Hope you had a great 4th of July!") == "banter"
    assert sentence_kind("Have a great Labor Day weekend!") == "banter"
    assert sentence_kind("Install must finish before the 4th of July.") == "work"
    assert sentence_kind("We need 40 drops by July 4th, hope that works") == "work"


def test_email_pleasantry_is_an_admission_chatter_atom(tmp_path: Path) -> None:
    from app.parsers.email_parser import EmailParser

    p = tmp_path / "a.eml"
    p.write_text(
        "From: Patrick <patrick@purtera-it.com>\nTo: Megan <megan@cust.com>\nSubject: Rollout\n"
        "Date: Tue, 15 Jul 2025 13:37:00 -0400\nContent-Type: text/plain; charset=utf-8\n\n"
        "Hi Megan,\n\nHope you had a great 4th of July!\n\nWe need 40 drops at the Delphos site.\n\nThanks,\nPatrick\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    hope = [a for a in atoms if a.raw_text == "Hope you had a great 4th of July!"]
    assert len(hope) == 1
    assert hope[0].atom_type == AtomType.deal_metadata
    assert hope[0].value.get("kind") == "admission_reject"
    assert "chatter" in hope[0].review_flags


def test_hubspot_note_pleasantry_is_an_admission_chatter_atom(tmp_path: Path) -> None:
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "000132-hs-note-1103.txt"
    p.write_text(
        "HubSpot Note: Rollout update\nHubSpot Note ID: 1103\nDate: 2026-07-07T18:58:14.007Z\n"
        "Author: Trent Torrence\n\nHope you had a great 4th of July! We need 40 drops at the Delphos site.\n",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("p", "a", p)
    by_text = {a.raw_text: a for a in atoms}
    hope = by_text["Hope you had a great 4th of July!"]
    assert hope.atom_type == AtomType.deal_metadata
    assert "chatter" in hope.review_flags and "admission_regex" in hope.review_flags
    assert by_text["We need 40 drops at the Delphos site."].atom_type == AtomType.scope_item


# ---------------------------------------------------------------------------
# 4 + 6. Email chain: credit and quoted greetings
# ---------------------------------------------------------------------------

_P_BODY = "Hi Megan,\n\nAttached is signed SOW and PO for the Delphos refresh.\n\nThanks,\nPatrick Doyle\n"
_M_BODY = "Hi Patrick,\n\nLet's go!!\n\nMegan Blevins\n"
_C_BODY = "Hi Megan,\n\nWe will pull 40 Cat6 drops to the second floor IDF.\n\nChase\n"


def _q_hdr(name: str, addr: str, sent: str, to: str, subj: str) -> str:
    return f"From: {name} <{addr}>\nSent: {sent}\nTo: {to}\nSubject: {subj}\n\n"


def _eml(frm: str, to: str, date: str, subj: str, mid: str, body: str, irt: str | None = None) -> str:
    h = f"From: {frm}\nTo: {to}\nSubject: {subj}\nDate: {date}\nMessage-ID: <{mid}>\n"
    if irt:
        h += f"In-Reply-To: <{irt}>\nReferences: <{irt}>\n"
    return h + "Content-Type: text/plain; charset=utf-8\n\n" + body


def _write_chain(d: Path, *, with_middle: bool = True) -> None:
    q1 = _q_hdr("Patrick Doyle", "patrick@purtera-it.com", "Tuesday, July 15, 2025 1:37 PM",
                "Megan Blevins <megan@cust.com>", "SOW signed") + _P_BODY
    q2 = _q_hdr("Megan Blevins", "megan@cust.com", "Tuesday, July 15, 2025 2:05 PM",
                "Patrick Doyle <patrick@purtera-it.com>", "RE: SOW signed") + _M_BODY + "\n" + q1
    (d / "a_sow_signed.eml").write_text(_eml(
        "Patrick Doyle <patrick@purtera-it.com>", "Megan Blevins <megan@cust.com>",
        "Tue, 15 Jul 2025 13:37:00 -0400", "SOW signed", "m1@x", _P_BODY), encoding="utf-8")
    if with_middle:
        (d / "b_re_sow_signed.eml").write_text(_eml(
            "Megan Blevins <megan@cust.com>", "Patrick Doyle <patrick@purtera-it.com>",
            "Tue, 15 Jul 2025 14:05:00 -0400", "RE: SOW signed", "m2@x", _M_BODY + "\n" + q1,
            irt="m1@x"), encoding="utf-8")
    (d / "c_re_re_sow_signed.eml").write_text(_eml(
        "Chase Miller <chase@purtera-it.com>", "Megan Blevins <megan@cust.com>",
        "Tue, 15 Jul 2025 15:00:00 -0400", "RE: SOW signed", "m3@x", _C_BODY + "\n" + q2,
        irt="m2@x"), encoding="utf-8")


def _credited_to(a) -> str:
    v = _value(a)
    msg = (v.get("email_thread") or {}).get("message") or {}
    return str(msg.get("author") or v.get("author") or "").lower()


def test_each_chain_line_is_credited_to_the_message_that_wrote_it(tmp_path: Path) -> None:
    _write_chain(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    lets_go = [a for a in r.atoms if _text(a) == "Let's go!!"]
    assert len(lets_go) == 1, [(_credited_to(a)) for a in lets_go]
    assert "megan@cust.com" in _credited_to(lets_go[0])
    assert not _value(lets_go[0]).get("quoted")
    # One "Hi Megan," per message that wrote one: Patrick's and Chase's.
    hi = [a for a in r.atoms if _text(a) == "Hi Megan,"]
    assert sorted("patrick" in _credited_to(a) for a in hi) == [False, True], [_credited_to(a) for a in hi]
    assert len([a for a in r.atoms if _text(a) == "Hi Patrick,"]) == 1
    assert len([a for a in r.atoms if _text(a) == "Thanks,"]) == 1


def test_a_quoted_only_line_is_credited_to_its_quoted_author(tmp_path: Path) -> None:
    """Megan's own email is not in the deal: her "Let's go!!" exists only as a
    quote inside Chase's reply, and must still read as hers -- not as part of
    Patrick's 1:37 PM message it answers."""
    _write_chain(tmp_path, with_middle=False)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    lets_go = [a for a in r.atoms if _text(a) == "Let's go!!"]
    assert len(lets_go) == 1
    assert "megan@cust.com" in _credited_to(lets_go[0])


# ---------------------------------------------------------------------------
# 5. Stakeholder dedup keeps the row with the email
# ---------------------------------------------------------------------------

def _person(aid: str, raw: str, val: dict, conf: float = 0.8) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_" + aid + raw[:6], project_id="p", artifact_id=aid, atom_type=AtomType.stakeholder,
        raw_text=raw, normalized_text=raw.lower(), value=dict(kind="person", **val), entity_keys=[],
        source_refs=[SourceRef(id="s" + aid, artifact_id=aid, artifact_type=ArtifactType.email,
                               filename=aid + ".eml", extraction_method="x", parser_version="x", locator={})],
        authority_class=AuthorityClass.customer_current_authored, confidence=conf,
        review_status=ReviewStatus.auto_accepted, parser_version="x",
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_stakeholder_fold_keeps_the_row_with_the_email(reverse: bool) -> None:
    from app.core.semantic_dedup import dedupe_stakeholder_atoms

    atoms = [
        _person("note", "Megan Blevins | 555-111-2222", {"name": "Megan Blevins", "phone": "555-111-2222"}, 0.9),
        _person("mail", "Megan Blevins <megan@cust.com>", {"name": "Megan Blevins", "email": "megan@cust.com"}),
    ]
    if reverse:
        atoms.reverse()
    out = dedupe_stakeholder_atoms(copy.deepcopy(atoms))
    assert len(out) == 1
    assert out[0].raw_text == "Megan Blevins <megan@cust.com>"
    assert out[0].value["email"] == "megan@cust.com"
    assert out[0].value["phone"] == "555-111-2222"


# ---------------------------------------------------------------------------
# 7. "we'll" as a promise
# ---------------------------------------------------------------------------

def test_promise_rule_keeps_commitments_and_lets_pleasantries_go() -> None:
    from app.core.deal_chatter import is_chatter

    for commitment in (
        "We'll schedule the techs once the floor is done.",
        "we'll get the techs out there next week",
        "Thanks, we'll send the updated quote Monday",
        "I will get a conversation going with the club owner",
        "We will get back to you with the quote",
    ):
        assert not is_chatter(commitment), commitment
    for talk in (
        "Well, thanks again for your time",
        "Well, looking forward to it",
        "We'll get back to you",
        "Thanks! We'll keep you posted.",
    ):
        assert is_chatter(talk), talk
