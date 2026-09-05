"""A PM's reason must survive their punctuation.

`extract_rationale` reads a reason only after an explicit connective —
because / since / so that. Live on 2026-09-05:

    "Say SLO not SLA — our contracts are written in service objectives."

returned an empty rationale, so the lesson would later fire with nothing to
show for itself. A suppressed question whose reason nobody can see is
indistinguishable from one that was lost.

Widening the pattern is the wrong fix: it means guessing which half of a
sentence is the reason. The reason is shown back to the PM wherever the lesson
fires, so a sentence they did not write reads as the system inventing their
judgment. The model may POINT at the reason; it may never author one, and its
answer is only accepted when the PM's note actually contains it.
"""

from __future__ import annotations

from app.core.pm_note_router import _verbatim_reason, extract_rationale

NOTE = "Say SLO not SLA — our contracts are written in service objectives."


def test_the_connective_reader_still_wins_when_there_is_one() -> None:
    assert extract_rationale("Say SLO because our contracts use objectives") == (
        "our contracts use objectives"
    )


def test_a_reason_after_a_dash_is_accepted_when_the_pm_wrote_it() -> None:
    assert _verbatim_reason("our contracts are written in service objectives", NOTE) == (
        "our contracts are written in service objectives"
    )


def test_a_reason_the_pm_never_wrote_is_refused() -> None:
    """The whole guard. A plausible invention is worse than no reason, because
    it is shown back to them as their own words."""
    assert _verbatim_reason("because SLAs are legally risky", NOTE) == ""
    assert _verbatim_reason("the customer prefers it", NOTE) == ""


def test_a_rephrasing_of_their_words_is_still_an_invention() -> None:
    assert _verbatim_reason("our contracts use service objectives", NOTE) == ""


def test_punctuation_and_case_do_not_decide_it() -> None:
    assert _verbatim_reason("Our Contracts Are Written In Service Objectives.", NOTE) != ""
    assert _verbatim_reason("our  contracts   are written in service objectives", NOTE) != ""


def test_a_fragment_too_short_to_be_a_reason_is_refused() -> None:
    assert _verbatim_reason("SLO", NOTE) == ""
    assert _verbatim_reason("", NOTE) == ""
    assert _verbatim_reason(None, NOTE) == ""


def test_the_model_is_asked_for_the_pms_own_words() -> None:
    from app.core import pm_note_router as m

    assert '"reason"' in m._LLM_PROMPT
    assert "character for" in m._LLM_PROMPT, "it must be told to copy, not compose"


def test_the_connective_reader_takes_precedence_over_the_model() -> None:
    """An explicit `because` is unambiguous; the model is the fallback, never
    an override of what the PM plainly said."""
    import inspect

    from app.core import pm_note_router as m

    src = inspect.getsource(m._llm_lessons)
    assert "rationale or _verbatim_reason(" in src


# ── the same defect on the pattern path, which is the common one ────────────
#
# The live smoke test on 2026-09-05 showed the model-pointed reason was not
# enough: the lesson came back `source: "pattern"`, so `_verbatim_reason` was
# never consulted. And because the reason was never separated, it stayed glued
# to the subject:
#
#   exemplar: "Who the onsite contact is — Carl Painter meets the tech at the
#              door every time?"                              grounded: False
#
# Two failures, one cause. A longer list of connectives would still be guessing
# which half is the reason. The CARD decides instead.

from app.core.pm_note_router import split_subject_and_reason

CARD = (
    "Who is the day-of onsite contact for 2970 brandywine rd ste 200, "
    "and how do we reach them — Yealink phones?"
)


def test_the_card_decides_where_the_reason_starts() -> None:
    subject, reason = split_subject_and_reason(
        "Stop asking who the onsite contact is — Carl Painter meets the tech at the door every time",
        [CARD],
    )
    assert subject == "Stop asking who the onsite contact is"
    assert reason == "Carl Painter meets the tech at the door every time"


def test_a_note_with_no_reason_keeps_its_whole_subject() -> None:
    subject, reason = split_subject_and_reason("Stop asking who the onsite contact is", [CARD])
    assert subject == "Stop asking who the onsite contact is"
    assert reason == ""


def test_a_note_that_matches_no_card_is_left_alone() -> None:
    """Never split on punctuation alone. With nothing to ground against there
    is no evidence which half is the reason, and inventing one is the failure
    this whole design exists to avoid."""
    subject, reason = split_subject_and_reason(
        "Say SLO not SLA — our contracts are written in service objectives", [CARD]
    )
    assert reason == ""
    assert subject.startswith("Say SLO not SLA")


def test_the_subject_is_the_leading_run_not_the_trailing_one() -> None:
    """A PM says what they are judging before they say why. Taking the trailing
    match would make the reason the thing we learn from."""
    subject, reason = split_subject_and_reason(
        "Stop asking who the onsite contact is — the day-of onsite contact never changes",
        [CARD],
    )
    assert subject.startswith("Stop asking")
    assert "never changes" in reason


def test_a_one_part_clause_is_untouched() -> None:
    assert split_subject_and_reason("Stop asking about badging", [CARD]) == (
        "Stop asking about badging", ""
    )


def test_no_questions_means_no_split() -> None:
    assert split_subject_and_reason("a — b", []) == ("a — b", "")


def test_the_pattern_path_uses_it() -> None:
    import inspect

    from app.core import pm_note_router as m

    src = inspect.getsource(m.route_note)
    assert "split_subject_and_reason(clause, questions)" in src
    assert "if not rationale and questions:" in src, "an explicit because still wins"
