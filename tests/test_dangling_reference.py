"""'the small job I was discussing earlier' says there was a call the deal
does not hold. Live 010288: nobody would have noticed by reading the thread."""
from __future__ import annotations

from app.core.dangling_reference import find_dangling_references
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text: str, value: dict | None = None, filename: str = "m.eml"):
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**8}", project_id="p", artifact_id="art_m", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value=value or {"kind": "email_body_line"}, entity_keys=[],
        source_refs=[SourceRef(id="s1", artifact_id="art_m", artifact_type=ArtifactType.email, filename=filename,
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def test_a_call_we_do_not_have_becomes_something_to_chase():
    a = _atom("Here are the details for the small job I was discussing earlier.",
              {"kind": "email_body_line",
               "said_by": {"name": "Alec Burns", "email": "alecbur@cdw.com", "company": "cdw.com", "side": "theirs"}})
    made = find_dangling_references([a], project_id="p", filenames=["m.eml"])
    assert len(made) == 1
    v = made[0].value
    assert v["reference_kind"] == "conversation" and v["wants"] == "chase-conversation"
    assert "Alec Burns" in made[0].raw_text
    assert made[0].atom_type == AtomType.dependency
    assert made[0].review_status == ReviewStatus.needs_review


def test_one_chase_item_per_kind_not_per_sentence():
    # "as discussed" appears in six mails of one thread; it is one missing call.
    atoms = [_atom(f"As we discussed, item {i} is on us.") for i in range(6)]
    assert len(find_dangling_references(atoms, project_id="p", filenames=["m.eml"])) == 1


def test_an_attachment_we_actually_hold_is_not_missing():
    a = _atom("Please see attached for the layout.")
    assert find_dangling_references([a], project_id="p", filenames=["m.eml"])
    assert not find_dangling_references(a and [a], project_id="p", filenames=["m.eml", "layout.pdf"])


def test_plain_prose_and_small_talk_raise_nothing():
    assert find_dangling_references([_atom("They are intending to use a maglock.")], project_id="p") == []
    chat = _atom("As we discussed, thanks again!", {"kind": "email_body_line", "chatter": True})
    assert find_dangling_references([chat], project_id="p") == []


def test_a_chase_for_someone_we_know_does_not_send_a_pm_hunting_this_thread():
    """AJ had worked with this rep before, so the "earlier" conversation
    predates the deal: hunting 010288 for it would find nothing."""
    a = _atom("Here are the details for the small job I was discussing earlier.",
              {"kind": "email_body_line",
               "said_by": {"name": "Alec Burns", "email": "alecbur@cdw.com", "company": "cdw.com", "side": "theirs"}})
    made = find_dangling_references([a], project_id="p", filenames=["m.eml"])
    assert made[0].value["prior_relationship"] is True
    assert "may predate this deal" in made[0].raw_text

    cold = _atom("Per our call, the riser room is locked after 6pm.")
    out = find_dangling_references([cold], project_id="p", filenames=["m.eml"])
    assert out[0].value["prior_relationship"] is False
    assert "find it or ask what was agreed" in out[0].raw_text


def test_no_regex_in_this_module_holds_a_literal_control_character():
    """Shell escaping turned \b into a literal backspace twice today, and the
    pattern still compiled -- it just silently matched nothing."""
    import re as _re

    import app.core.dangling_reference as mod

    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, _re.Pattern):
            assert not any(ord(c) < 32 for c in obj.pattern), f"{name} holds a control character"
