"""Per-source briefs (C1 of purpulse-frontend docs/design/sources-reader/CALM-PLAN.md).

Shaped like deal 000112 (Phillips Connect - Xtra Lease): a Fireflies kickoff
call, an install guide PDF, an email thread and HubSpot notes.
"""
import json

import pytest

from app.core import c3_brief, source_briefs as SB


def _atom(i, artifact, text, atype="scope_item", conf=0.68, **loc):
    return {"id": f"atm_{i}", "artifact_id": artifact, "atom_type": atype, "confidence": conf,
            "text": text, "locator": loc, "structured": {}, "review_flags": []}


ENV = {
    "documents": [
        {"artifact_id": "call", "filename": "000112-fireflies-01KZVW7RTH-XTRA-Install-Kickoff-Call-transcript.json",
         "artifact_type": "transcript", "authored_at": "2026-08-14T15:00:00Z"},
        {"artifact_id": "pdf", "filename": "Xtra Lease K298_StealthNet_MidTurn v1.7.pdf", "artifact_type": "pdf"},
        {"artifact_id": "m1", "filename": "000112-hs-email-1.eml", "artifact_type": "email"},
        {"artifact_id": "m2", "filename": "000112-hs-email-2.eml", "artifact_type": "email"},
        {"artifact_id": "n1", "filename": "000112-hs-note-110508283233-From Cliff Creech.txt", "artifact_type": "txt",
         "authored_at": "2026-06-02T13:57:43Z"},
        {"artifact_id": "n2", "filename": "000112-hs-note-117847550751-Waiting on PCT MSA.txt", "artifact_type": "txt",
         "authored_at": "2026-10-01T21:28:49Z"},
    ],
    "email_threads": [{"thread_id": "t1", "name": "Install guides for the K298 kit", "artifact_ids": ["m1", "m2"]}],
    "atoms": [
        _atom(1, "call", "Good morning.", "small_talk", 0.2, timestamp_start=22),
        _atom(2, "call", "It was the K298 Dry Van, StealthNet, Mid Turn ALT Power, ABS Reader, CargoVision.",
              timestamp_start=351),
        _atom(3, "call", "Box two would be the sensor kit, which would only go on 2016 and newer dry vans.",
              timestamp_start=442),
        _atom(4, "call", "So it's just a 4 inch hole to access the camera into the trailer and then four rivets.",
              timestamp_start=483),
        _atom(5, "call", "You grab them and go, is it 50 at a time?", "open_question", timestamp_start=944),
        _atom(6, "call", "It's Friday.", conf=0.62, timestamp_start=49, **{}),
        _atom(7, "call", "Confidential and privileged", conf=0.0),
        _atom(8, "pdf", "Connect the 77-E103 12V Car Adapter harness to the StealthNet.", page=2),
        _atom(9, "m1", "From: Cliff Creech | Sent: Monday", "deal_metadata", message_index=0),
        _atom(10, "n1", "note_id=110508283233 | author=Megan Blevins | date=2026-06-02", "deal_metadata"),
        _atom(11, "n1", "we allow 1.5 hours to install and verify this kit.", conf=0.78),
        _atom(12, "n2", "Waiting on PCT MSA - Getting direct MVA done to hopefully start doing some of these",
              conf=0.78),
    ],
}
ENV["atoms"][6]["structured"] = {"chatter": True}


def test_sources_follow_the_reader():
    got = {s["key"]: s for s in SB.sources_of(ENV)}
    assert got["call"]["kind"] == "call" and got["call"]["title"] == "XTRA Install Kickoff Call"
    assert got["pdf"]["kind"] == "file"
    # One thread, not two messages; the notes together, oldest first.
    assert got["thread:t1"]["artifact_ids"] == ["m1", "m2"] and "m1" not in got
    assert got["notes"]["artifact_ids"] == ["n1", "n2"] and "n1" not in got


def test_lines_drop_boilerplate_and_say_where():
    call = next(s for s in SB.sources_of(ENV) if s["key"] == "call")
    lines = SB.lines_of(ENV, call)
    ids = [x["id"] for x in lines]
    assert "atm_1" not in ids  # small talk
    assert "atm_7" not in ids  # chatter the parser refused
    assert ids == ["atm_2", "atm_3", "atm_4", "atm_5", "atm_6"]  # the envelope's reading order
    assert {x["id"]: x["where"] for x in lines}["atm_3"] == "7:22"
    notes = next(s for s in SB.sources_of(ENV) if s["key"] == "notes")
    nl = SB.lines_of(ENV, notes)
    assert [x["id"] for x in nl] == ["atm_11", "atm_12"]  # the note_id= metadata line is out
    assert nl[0]["where"] == "2026-06-02"
    thread = next(s for s in SB.sources_of(ENV) if s["key"] == "thread:t1")
    assert SB.lines_of(ENV, thread) == []  # only a header line


def test_where_reads_seconds_ms_pages_and_messages():
    assert SB.where({"locator": {"timestamp_start": 378.4}}) == "6:18"
    assert SB.where({"locator": {"start_ms": 944000}}) == "15:44"
    assert SB.where({"locator": {"page": 2}}) == "p. 3"
    assert SB.where({"locator": {"message_index": 1}}) == "message 2"
    assert SB.where({"locator": {}}) == ""


def test_check_keeps_only_cited_text_under_known_headings():
    lines = [{"id": "atm_2", "where": "5:51", "text": "x"}, {"id": "atm_3", "where": "7:22", "text": "y"},
             {"id": "atm_5", "where": "15:44", "text": "z"}]
    doc = {
        "lead": {"text": "The kickoff for the 77-K298 dry-van installs.", "cites": ["atm_2"]},
        "sections": [
            {"heading": "Scope", "items": [
                {"text": "Install the 77-K298 kit.", "cites": ["[atm_2]"]},
                {"text": "Sensor kit only on 2016-and-newer dry vans.", "cites": ["atm_3", "atm_2"]},
                {"text": "Invented line.", "cites": ["atm_999"]},
                {"text": "No citation.", "cites": []}]},
            {"heading": "Weather", "items": [{"text": "Sunny.", "cites": ["atm_2"]}]},
            {"heading": "open questions", "items": [{"text": "How many trailers per batch?", "cites": "atm_5"}]},
        ],
        "inputs": [
            {"label": "Batch size", "value": "Not answered", "open": True, "cites": ["atm_5"]},
            {"label": "Kit", "value": "77-K298", "open": False, "cites": ["atm_2"]},
            {"label": "Guess", "value": "40 trailers", "cites": ["atm_1"]},
        ],
    }
    got = SB.check(doc, lines)
    assert got["lead"]["cites"] == ["atm_2"]
    assert [s["heading"] for s in got["sections"]] == ["Scope", "Open questions"]
    scope = got["sections"][0]["items"]
    assert [x["text"] for x in scope] == ["Install the 77-K298 kit.", "Sensor kit only on 2016-and-newer dry vans."]
    assert scope[1]["where"] == ["7:22", "5:51"]
    assert got["sections"][1]["items"][0]["cites"] == ["atm_5"]
    assert [x["label"] for x in got["inputs"]] == ["Batch size", "Kit"]
    assert got["inputs"][0]["open"] is True and got["inputs"][1]["open"] is False
    assert got["dropped_uncited"] == 3  # atm_999, no citation, atm_1
    assert got["dropped_heading"] == 1


def test_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("SOWSMITH_SOURCE_BRIEFS", raising=False)
    assert SB.for_envelope(ENV) is None


def _reply(prompt, model, tokens):
    ids = [s.split("]")[0].lstrip("[") for s in json.loads(prompt.split("LINES: ", 1)[1])]
    return json.dumps({"lead": {"text": "What this source says.", "cites": ids[:1]},
                       "sections": [{"heading": "Scope", "items": [{"text": "A line.", "cites": ids[:1]}]}],
                       "inputs": [{"label": "Kit", "value": "77-K298", "open": False, "cites": ids[:1]}]})


@pytest.fixture
def store(monkeypatch, tmp_path):
    saved = {}
    from app.storage import repositories

    monkeypatch.setattr(repositories, "load_cache_payload", lambda k: saved.get(k))
    monkeypatch.setattr(repositories, "save_cache_payload", lambda k, p: saved.__setitem__(k, p))
    monkeypatch.setenv("SOWSMITH_SOURCE_BRIEFS", "on")
    monkeypatch.setattr(SB, "MIN_LINES", 1)
    return saved


def test_build_writes_each_source_once_then_reads_the_cache(monkeypatch, store):
    calls = []
    monkeypatch.setattr(c3_brief, "_complete", lambda p, m, t: calls.append(p) or _reply(p, m, t))
    first = SB.for_envelope(ENV)
    assert set(first["by_source"]) == {"call", "pdf", "notes"}
    assert first["skipped"] == [{"key": "thread:t1", "why": "0 line(s) to read"}]
    assert first["calls"] == 3 and first["cached"] == 0 and len(calls) == 3
    call = first["by_source"]["call"]
    assert call["lead"]["text"] == "What this source says." and call["inputs"][0]["label"] == "Kit"
    assert "Good morning" not in calls[0] and "Confidential" not in calls[0]
    again = SB.for_envelope(ENV)
    assert len(calls) == 3 and again["cached"] == 3 and again["calls"] == 0
    assert again["by_source"]["call"]["cached"] is True
    assert again["by_source"]["call"]["inputs"] == call["inputs"]


def test_a_changed_source_is_written_again(monkeypatch, store):
    calls = []
    monkeypatch.setattr(c3_brief, "_complete", lambda p, m, t: calls.append(p) or _reply(p, m, t))
    SB.for_envelope(ENV)
    env2 = json.loads(json.dumps(ENV))
    env2["atoms"].append(_atom(13, "pdf", "Remove the IMEI stickers and apply one to the mounting plate.", page=2))
    out = SB.for_envelope(env2)
    assert out["calls"] == 1 and out["cached"] == 2


def test_an_unusable_reply_is_an_error_not_a_brief(monkeypatch, store):
    monkeypatch.setattr(c3_brief, "_complete", lambda p, m, t: "sorry, I cannot")
    out = SB.for_envelope(ENV)
    assert out["errors"] == 3
    assert out["by_source"]["call"]["error"] == "the writer returned no usable JSON"
    assert out["by_source"]["call"]["sections"] == []
    assert store == {}  # nothing cached


def test_short_sources_are_skipped_with_why(monkeypatch, store):
    monkeypatch.setattr(SB, "MIN_LINES", 3)
    monkeypatch.setattr(c3_brief, "_complete", _reply)
    out = SB.for_envelope(ENV)
    assert set(out["by_source"]) == {"call"}
    assert {s["key"] for s in out["skipped"]} == {"pdf", "notes", "thread:t1"}
