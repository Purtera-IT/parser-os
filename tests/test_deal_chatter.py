"""Live 010289: 11 of 49 atoms were relationship talk, each one a card asking
the PM to type a sentence that says nothing about the work."""
from __future__ import annotations

from app.core.deal_chatter import is_chatter, mark_chatter
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text: str, atom_type: AtomType = AtomType.scope_item, value: dict | None = None, keys: list[str] | None = None):
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**8}", project_id="p", artifact_id="a", atom_type=atom_type,
        raw_text=text, normalized_text=text.lower(), value=value if value is not None else {"kind": "email_body_line"},
        entity_keys=keys or [],
        source_refs=[SourceRef(id="src_a", artifact_id="a", artifact_type=ArtifactType.txt, filename="a.txt",
                               locator={}, extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="test",
    )


def test_relationship_talk_is_chatter():
    for t in [
        "Thank you for bringing this our way.",
        "Sending it over to my solutions team now!",
        "If this is a successful implementation, it could lead to many more of the same opportunity.",
        "Here are the details for the small job I was discussing earlier.",
        "Sorry, left that part off.",
    ]:
        assert is_chatter(t), t


def test_anything_about_the_work_is_not_chatter():
    # A pipeline sentence that also names the hardware is about the hardware.
    for t in [
        "They are intending to use a maglock.",
        "The connectivity kit limits our power to 40-50ft.",
        "If this is successful we would run Cat6 to the second door as well.",
        "How many doors - 1 external access point [front door]",
    ]:
        assert not is_chatter(t), t
    assert not is_chatter("Looking forward to the rollout", entity_keys=["site:bethesda"])


def test_marking_keeps_the_atom_and_spares_list_items():
    atoms = [
        _atom("Thank you for bringing this our way."),
        _atom("Relay", value={"kind": "email_body_line", "list_item": True, "list_label": "Provided by us:"}),
        _atom("Mount the reader at the front door."),
    ]
    assert mark_chatter(atoms) == 1
    assert len(atoms) == 3  # nothing is deleted
    assert any(r["key"] == "small_talk" for r in atoms[0].value["reads"])
    assert "chatter" in atoms[0].review_flags
    assert atoms[0].atom_type.value == "scope_item"  # the rule does not retype
    assert not atoms[1].value.get("chatter") and not atoms[2].value.get("chatter")
    assert mark_chatter(atoms) == 0  # idempotent


def test_a_promise_is_never_hidden_as_banter():
    """The rule hid "I will get a conversation going with the club owner" --
    the only sentence naming this deal's decision maker -- because it contains
    a phrase that also shows up in pipeline chatter. A judgement a pattern
    cannot make is a head, so the rule now only PREDICTS."""
    promise = _atom("If you all would be able to do something like this, I will get a conversation going with the club owner.")
    assert not is_chatter(promise.raw_text)
    assert mark_chatter([promise]) == 0

    banter = _atom("Thank you for bringing this our way.")
    assert mark_chatter([banter]) == 1
    # nothing is hidden and nothing is retyped: it is a guess on the card
    assert banter.atom_type.value == "scope_item"
    read = next(r for r in banter.value["reads"] if r["key"] == "small_talk")
    assert read["value"] is True and read["confidence"] <= 0.5
    assert not banter.value.get("chatter")
