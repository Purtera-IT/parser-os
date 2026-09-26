"""The PM is asked each thing once, and never something the deal answers.

Three streams feed the question queue -- this deal's SRL gaps, the standing
head-start checklist, and the missing-field checklist -- and they overlap. On
010288 the overlap was not subtle: `site_contact` and `payment_terms` arrived
from all three under the SAME field_id, all thirteen deal gaps reappeared in the
checklist, and the near-misses were one topic under two spellings --
`work_hours` and `workhours`.

The dedup compared the TEXT, so a question asked in two wordings survived twice
and the PM was told to chase it twice. Nine of eighty-one candidates on that
deal were duplicates of another candidate in the same queue.

`covered` made it worse: the head-start computes, per topic, whether the deal's
own words already address it, and that verdict dropped the head-start copy while
the identical SRL gap sailed past. The deal says "1 external access point [front
door]" and the PM was still asked to confirm the device count.
"""
from __future__ import annotations

from app.core.orbitbrief_core import _dedupe_questions, _question_topic


def q(kind: str, field_id: str, text: str) -> dict:
    return {"kind": kind, "field_id": field_id, "text": text}


def test_one_question_per_topic_however_it_is_worded():
    asked = [
        q("generated_gap", "kickoff_date", "What is the kickoff / project start date?"),
        q("headstart", "kickoff", "What is the confirmed kickoff / start date?"),
    ]
    out = _dedupe_questions(asked, set())
    assert len(out) == 1


def test_the_deals_own_wording_wins_over_the_standing_list():
    """Both ask for the site contact. The deal's gap is phrased the way this
    customer would answer it; the standing item is generic."""
    asked = [
        q("headstart", "site_contact", "Who is the on-site contact (name + mobile) at each location?"),
        q("generated_gap", "site_contact", "Who is the on-site contact (name + phone) for each location?"),
    ]
    out = _dedupe_questions(asked, set())
    assert len(out) == 1
    assert out[0]["kind"] == "generated_gap"


def test_a_topic_the_deal_already_covers_is_not_asked():
    """The deal states "1 external access point [front door]"."""
    asked = [q("generated_gap", "device_qty_per_site", "What is the confirmed device count per site?")]
    assert _dedupe_questions(asked, {"f:device_count"}) == []


def test_an_underscore_is_not_a_different_question():
    assert _question_topic(q("generated_gap", "work_hours", "x")) == \
           _question_topic(q("headstart", "workhours", "y"))


def test_different_topics_are_both_kept():
    """The dedup must not swallow questions that merely sound similar. Who
    APPROVES access and WHO opens the door on the day are two people."""
    asked = [
        q("generated_gap", "site_access_terms", "What are the site access, escort, and badging terms?"),
        q("headstart", "security_contact", "Who approves site/badge access and security clearance?"),
        q("headstart", "permits", "Are permits / AHJ approvals / inspections required?"),
    ]
    assert len(_dedupe_questions(asked, set())) == 3


def test_a_question_with_no_topic_falls_back_to_its_words():
    asked = [
        {"kind": "open_question", "text": "4 parts are claimed by both sides. Who supplies them?"},
        {"kind": "open_question", "text": "4 parts are claimed by both sides. Who supplies them?"},
        {"kind": "open_question", "text": "Something else entirely?"},
    ]
    assert len(_dedupe_questions(asked, set())) == 2
