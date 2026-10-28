"""semantic_dedup never folds lines stating different values (audit 2026-10-28, wave 3 #5). Synthetic."""
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import semantic_dedup_atoms


def _atom(i, text, value, kind=AtomType.deal_metadata):
    return EvidenceAtom(
        id=f"atm_{i}", project_id="p", artifact_id="art_a", atom_type=kind, raw_text=text,
        normalized_text=text.lower(), value=value, entity_keys=[], source_refs=[SourceRef(
            id=f"src_{i}", artifact_id="art_a", artifact_type=ArtifactType.txt, filename="note.txt",
            locator={"line_start": i}, extraction_method="hubspot_note_parser", parser_version="t")],
        receipts=[], authority_class=AuthorityClass.meeting_note, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t")


def test_a_list_of_cities_is_not_folded_into_its_lead_in():
    lead = _atom(1, "Here is a list of the cities involved:", {"field_name": "city", "value": "list of cities involved"})
    cities = [_atom(2, "Lisbon - 16 Users", {"field_name": "city", "value": "Lisbon", "quantity": 16}),
              _atom(3, "Oslo, Norway - 30 Users", {"field_name": "city", "value": "Oslo", "quantity": 30})]
    out = semantic_dedup_atoms([lead, *cities])
    assert {a.raw_text for a in out} == {a.raw_text for a in [lead, *cities]}


def test_rom_ranges_without_a_single_amount_stay_apart():
    rows = [_atom(i, t, {"category": "ROM", "amounts": a, "rom_text": t}, AtomType.commercial_total)
            for i, (t, a) in enumerate([("$140 to $190", ["140", "190"]), ("$175 to $250", ["175", "250"]),
                                        ("$55 to $85", ["55", "85"])])]
    out = semantic_dedup_atoms(rows)
    assert len(out) == 3
    # the same range said twice is still one line
    twice = [_atom(7, "$140 to $190", {"category": "ROM", "amounts": ["140", "190"]}, AtomType.commercial_total),
             _atom(8, "$140 to $190", {"category": "ROM", "amounts": ["140", "190"]}, AtomType.commercial_total)]
    assert len(semantic_dedup_atoms(twice)) == 1
