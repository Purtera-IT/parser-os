"""Live 010289: three sentences that are not scope, three facts we were
throwing away -- the job is small, there is a diagram we do not hold, and the
account has room to grow."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.deal_chatter import mark_chatter
from app.core.deal_signals import extract_deal_signals
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

ASK = "Here are the details for the small job I was discussing earlier."
GROW = "If this is a successful implementation, it could lead to many more of the same opportunity."
DIAGRAM = "Diagram: https://huzzard.com/wp-content/uploads/2025/11/BPW061725-Rev-1-Diagram-768x593.png"


def _atom(text: str, value: dict | None = None, atom_type: AtomType = AtomType.deal_metadata):
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**8}", project_id="p", artifact_id="art_m", atom_type=atom_type,
        raw_text=text, normalized_text=text.lower(), value=value if value is not None else {"kind": "email_body_line"},
        entity_keys=[],
        source_refs=[SourceRef(id="src_1", artifact_id="art_m", artifact_type=ArtifactType.txt, filename="m.eml",
                               locator={}, extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="test",
    )


def test_the_job_size_is_what_the_sentence_MEANS_not_a_new_atom():
    """Nobody said "the sender calls this a small job" -- it is what their
    sentence means, so it rides on that sentence. A head reads an atom and
    says what it implies; that is the shape it has to learn."""
    src = _atom(ASK)
    made = extract_deal_signals([src], project_id="p")
    assert made == []  # no sentence invented
    read = next(r for r in src.value["reads"] if r["key"] == "job_scale")
    assert read["value"] == "small" and read["source"] == "rule"
    assert read["why"] == "small job"
    # and the sentence it came from is no longer small talk
    assert src.value["signals"] == ["scale:small"]
    assert mark_chatter([src]) == 0


def test_a_diagram_we_do_not_hold_becomes_something_to_go_and_get():
    url = DIAGRAM.split(" ", 1)[1]
    src = _atom(DIAGRAM, {"kind": "note_field_image", "media_type": "image", "image_url": url})
    made = extract_deal_signals([src], project_id="p", filenames=["m.eml", "010289-note-The Ask w diagram link.txt"])
    assert made == []  # nothing invented: the reading rides on the line
    reads = {r["key"]: r["value"] for r in src.value["reads"]}
    assert reads["points_at_artifact"] == "diagram"
    assert reads["needs_artifact"] == "diagram"
    assert src.value["artifact_url"] == url
    # a mail NAMED "…diagram link.txt" is not the diagram; a real drawing is
    held = _atom(DIAGRAM, {"image_url": url})
    extract_deal_signals([held], project_id="p", filenames=["BPW061725-Rev-1-Diagram.png"])
    assert not any(r["key"] == "needs_artifact" for r in held.value["reads"])


def test_room_to_grow_is_recorded_on_the_line_that_says_it():
    src = _atom(GROW)
    assert extract_deal_signals([src], project_id="p") == []
    assert [r["key"] for r in src.value["reads"]] == ["expansion"]
    assert mark_chatter([src]) == 0


def test_pure_courtesy_is_still_small_talk():
    src = _atom("Thank you for bringing this our way.")
    assert extract_deal_signals([src], project_id="p") == []
    assert not (src.value or {}).get("reads")
    assert mark_chatter([src]) == 1


# ---------------------------------------------------------------------------
# A conditional is the critical path, and an announcement scopes what follows.
# Both were invisible: one flattened into the type, the other into nothing.
# ---------------------------------------------------------------------------

def _reads_of(atom, key):
    for r in (atom.value.get("reads") or []):
        if r["key"] == key:
            return r
    return None


def _spoken(text, *, by_side="theirs", by_role="reseller", to_side="ours"):
    """One atom, said by somebody to somebody -- which is what decides who a
    conditional falls on. "If YOU all would be able to" means us when a
    reseller writes it to us and means them when we write it to them."""
    return SimpleNamespace(
        raw_text=text,
        text=text,
        value={
            "said_by": {"side": by_side, "role": by_role},
            "said_to": [{"side": to_side, "role": "internal" if to_side == "ours" else "reseller"}],
        },
        project_id="010288",
    )


def test_a_promise_waiting_on_us_says_so():
    a = _spoken(
        "If you all would be able to do something like this, I will get a conversation "
        "going with the club owner."
    )
    extract_deal_signals([a], project_id="010288", filenames=[])
    r = _reads_of(a, "blocked_on")
    assert r is not None and r["value"] == "us"
    assert "conditional" in r["why"]


def test_the_same_words_from_our_side_wait_on_them():
    a = _spoken(
        "If you can confirm the lock type, we will finalise the quote.",
        by_side="ours", by_role="internal", to_side="theirs",
    )
    extract_deal_signals([a], project_id="010288", filenames=[])
    assert _reads_of(a, "blocked_on")["value"] == "partner"


def test_a_sentence_that_merely_contains_if_is_not_blocked_on_anyone():
    for text in ["Let me know if you want the floorplan.", "They are intending to use a maglock."]:
        a = _spoken(text)
        extract_deal_signals([a], project_id="010288", filenames=[])
        assert _reads_of(a, "blocked_on") is None, text


def test_an_announcement_says_what_it_opens():
    a = _spoken("Here are the details for the small job I was discussing earlier.")
    extract_deal_signals([a], project_id="010288", filenames=[])
    r = _reads_of(a, "opens_block")
    assert r is not None
    assert r["value"].startswith("the details for the small job")


def test_a_line_that_merely_mentions_details_opens_nothing():
    for text in ["Thanks for the details.", "The details are in the attached quote."]:
        a = _spoken(text)
        extract_deal_signals([a], project_id="010288", filenames=[])
        assert _reads_of(a, "opens_block") is None, text


def test_expansion_says_what_would_repeat_not_just_that_it_might():
    a = _spoken("If this is a successful implementation, it could lead to many more of the same opportunity.")
    extract_deal_signals([a], project_id="010288", filenames=[])
    r = _reads_of(a, "expansion")
    assert r is not None
    # Not `True`, and not a truncated "lead to many more" either: the thing
    # that repeats is the half worth having.
    assert isinstance(r["value"], str)
    assert r["value"] == "lead to many more of the same opportunity"
