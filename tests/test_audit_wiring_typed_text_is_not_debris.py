"""A typed line is never OCR debris (audit 2026-10-28, wave 12 W10): a CRM note naming four customer
accounts in lowercase was dropped as unreadable before the heads saw it. Synthetic text."""
from app.core.atom_substance_gate import drop_unreadable_text
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text, art, method="t"):
    return EvidenceAtom(
        id="a" + str(abs(hash((text, art)))), project_id="p", artifact_id="x", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={}, entity_keys=[],
        source_refs=[SourceRef(id="s", artifact_id="x", artifact_type=art, filename="f", locator={},
                               extraction_method=method, parser_version="t")],
        receipts=[], authority_class=AuthorityClass.meeting_note, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t")


def test_a_typed_list_of_account_names_survives():
    note = _atom("atrium kelso, novarro, dukane, wakeford medcen.", ArtifactType.txt, "hubspot_note_parser")
    kept, dropped = drop_unreadable_text([note])
    assert kept == [note] and dropped == []


def test_scanned_debris_is_still_dropped():
    debris = _atom("Tes aks wilenur tht projetcompen mee egutements", ArtifactType.pdf)
    ocr = _atom("Tes aks wilenur tht projetcompen mee egutements", ArtifactType.docx, "docx_image_ocr")
    kept, dropped = drop_unreadable_text([debris, ocr])
    assert dropped == [debris, ocr]
