"""Live 010289: three sentences that are not scope, three facts we were
throwing away -- the job is small, there is a diagram we do not hold, and the
account has room to grow."""
from __future__ import annotations

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
    # a drawing we do not hold is real WORK, so it stays an item of its own
    dep = next(a for a in made if a.value["kind"] == "missing_artifact")
    assert next(r for r in src.value["reads"] if r["key"] == "points_at_artifact")["value"] == "diagram"
    assert dep.atom_type == AtomType.dependency
    assert dep.value["artifact_kind"] == "diagram" and dep.value["url"] == url
    # a mail NAMED "…diagram link.txt" is not the diagram; a real drawing is
    have = extract_deal_signals([_atom(DIAGRAM, {"image_url": url})], project_id="p",
                                filenames=["BPW061725-Rev-1-Diagram.png"])
    assert not [a for a in have if a.value["kind"] == "missing_artifact"]


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
