"""A note about a SOURCE has to be learned from the source, not the sentence.

The `sheet` head can only ever get its first exemplar from a PM saying so. The
note router is that path — but a lesson stored as the PM's prose ("that Chipotle
reporting file is just a customer dump") matches nothing next month, because
next month's export has the same identity and entirely different rows.

Same lesson as gap grounding: learn from the thing being judged. A sheet's
identity is its tab name and header row, which is exactly what recurs across
deals.
"""

from __future__ import annotations

from app.core.pm_note_router import route_note
from app.parsers.sheet_classifier import sheet_exemplar

NO_LLM = lambda *a, **k: None  # noqa: E731

SHEET = (
    "SSRS-SL-CUS001-CustomerOut | Customer Code INT | Customer Code | "
    "Customer Description | Contact Name | Order Date | Order Status"
)
OTHER_SHEET = "Site Roster | Site | Floor | Hostname | Quantity | Access Window"


def _sheet_lessons(lessons):
    return [l for l in lessons if l.head == "sheet"]


def test_the_prompt_tells_the_model_when_a_note_is_about_a_worksheet() -> None:
    from app.core import pm_note_router as m

    assert 'head "sheet"' in m._LLM_PROMPT
    assert "judges the SOURCE" in m._LLM_PROMPT, "the distinction from a gap lesson is the point"


def test_a_sheet_lesson_is_grounded_in_the_sheet() -> None:
    """Fed a lesson whose exemplar is the PM's paraphrase, the router swaps in
    the sheet it is plainly about."""
    from app.core.pm_note_router import Lesson, NoteRouting

    routing = NoteRouting()
    routing.lessons = [
        Lesson(head="sheet", exemplar="SSRS-SL-CUS001-CustomerOut customer dump",
               new_value="reference", scope="deal")
    ]
    # Exercise the grounding the router applies at the end of route_note.
    from app.core.pm_note_router import ground_in_questions

    grounded = ground_in_questions(routing.lessons[0].exemplar, [SHEET, OTHER_SHEET])
    assert grounded == SHEET, "learn from the sheet's identity, not the sentence"


def test_it_does_not_ground_a_sheet_lesson_in_an_unrelated_sheet() -> None:
    from app.core.pm_note_router import ground_in_questions

    assert ground_in_questions("SSRS customer out export", [OTHER_SHEET]) == ""


def test_the_router_accepts_sheets_alongside_questions() -> None:
    import inspect

    from app.core import pm_note_router as m

    sig = inspect.signature(m.route_note)
    assert "sheets" in sig.parameters
    assert "sheets" in inspect.signature(m.apply_note).parameters


def test_a_note_with_no_sheets_behaves_exactly_as_before() -> None:
    """Adding the parameter must not change any existing note."""
    r = route_note("Stop asking who signs acceptance", questions=[], synthesize=NO_LLM)
    assert _sheet_lessons(r.lessons) == []


def test_the_endpoint_carries_sheets_through() -> None:
    import inspect

    from app.api import routes_feedback as rf

    assert "sheets" in rf.NoteRequest.model_fields
    src = inspect.getsource(rf.feedback_note)
    assert src.count("sheets=req.sheets") == 2, "both the dry-run and the commit path"


def test_the_sheet_identity_is_what_recurs_across_deals() -> None:
    """The rows differ every month; the tab name and header do not."""
    june = [["CHIPOTLE MEXICAN GRILL", "SSRS-SL-CUS001-CustomerOut"], [None],
            ["Customer Code INT", "Customer Code", "Customer Description", "Contact Name"],
            ["1", "1", "CHIPOTLE", "A Buyer"]]
    sept = [["CHIPOTLE MEXICAN GRILL", "SSRS-SL-CUS001-CustomerOut"], [None],
            ["Customer Code INT", "Customer Code", "Customer Description", "Contact Name"],
            ["9", "9", "SOMEBODY ELSE", "B Buyer"]]
    assert sheet_exemplar("SSRS-SL-CUS001-CustomerOut", june) == sheet_exemplar(
        "SSRS-SL-CUS001-CustomerOut", sept
    )
