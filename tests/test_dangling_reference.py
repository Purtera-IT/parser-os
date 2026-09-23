"""'the small job I was discussing earlier' says there was a call the deal
does not hold. Live 010288: nobody would have noticed by reading the thread.

Nothing here is minted. "Alecandrich refers to a conversation the deal does
not hold" was the parser's sentence, not his -- an atom nobody said. The
reading rides on the line that points at the missing thing.
"""
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


def _read(atom, key):
    return next((r for r in (atom.value or {}).get("reads") or [] if r["key"] == key), None)


def test_a_call_we_do_not_have_is_read_onto_the_line_that_points_at_it():
    a = _atom("Here are the details for the small job I was discussing earlier.",
              {"kind": "email_body_line",
               "said_by": {"name": "Alec Burns", "email": "alecbur@cdw.com", "company": "cdw.com", "side": "theirs"}})
    assert find_dangling_references([a], project_id="p", filenames=["m.eml"]) == 1
    assert _read(a, "chase")["value"] == "conversation"
    assert a.value["chase"]["prior_relationship"] is True
    # someone we know: the call predates the deal, so do not send a PM hunting
    assert "ask them what was agreed" in a.value["chase"]["ask"]


def test_one_chase_per_kind_not_per_sentence():
    atoms = [_atom(f"As we discussed, item {i} is on us.") for i in range(6)]
    assert find_dangling_references(atoms, project_id="p", filenames=["m.eml"]) == 1
    assert sum(1 for a in atoms if _read(a, "chase")) == 1


def test_an_attachment_we_actually_hold_is_not_missing():
    a = _atom("Please see attached for the layout.")
    assert find_dangling_references([a], project_id="p", filenames=["m.eml"]) == 1
    b = _atom("Please see attached for the layout.")
    assert find_dangling_references([b], project_id="p", filenames=["m.eml", "layout.pdf"]) == 0
    assert not _read(b, "chase")


def test_plain_prose_and_small_talk_raise_nothing():
    a = _atom("They are intending to use a maglock.")
    assert find_dangling_references([a], project_id="p") == 0
    chat = _atom("As we discussed, thanks again!", {"kind": "email_body_line", "chatter": True})
    assert find_dangling_references([chat], project_id="p") == 0


def test_a_cold_contact_is_told_to_find_the_thread():
    a = _atom("Per our call, the riser room is locked after 6pm.")
    assert find_dangling_references([a], project_id="p", filenames=["m.eml"]) == 1
    assert a.value["chase"]["prior_relationship"] is False
    assert "find it or ask what was agreed" in a.value["chase"]["ask"]


def test_no_regex_in_this_module_holds_a_literal_control_character():
    """Shell escaping turned \b into a literal backspace twice; the pattern
    still compiled, it just silently matched nothing."""
    import re as _re

    import app.core.dangling_reference as mod

    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, _re.Pattern):
            assert not any(ord(c) < 32 for c in obj.pattern), f"{name} holds a control character"
