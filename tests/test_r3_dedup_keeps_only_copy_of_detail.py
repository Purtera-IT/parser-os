"""A dedup fold never drops the only copy that carries a detail the survivor lacks.

Ox 010353 (JSON intake, compile b6911826) lost three things to three passes:

* ``semantic_dedup`` (physical_site): the one address with its ZIP folded into
  a higher-quality copy that had none;
* ``atom_type_sanity.merge_signature_rows``: Megan's full contact row (email,
  phone) on the signature page folded into a record of name and title only;
* ``dedupe_stakeholder_atoms``: the SOW step "Call Client Support Manager John
  Ozuna-Diaz ... upon arrival" keyed on John's name and folded into his person
  record -- no copy survived.

The rule is ``fold_invariants.detail_only_the_loser_states``: ZIP, phone,
email and instruction. The copy carrying the detail becomes the survivor, or
its detail is merged in, or it stays standing.
"""
from __future__ import annotations

from app.core import fold_invariants as fold
from app.core.atom_type_sanity import merge_signature_rows
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import dedupe_stakeholder_atoms, semantic_dedup_atoms


def _atom(aid, text, atype, value=None, *, art="a", page=None, conf=0.8):
    loc = {"table_index": 1, "row": len(aid)}
    if page is not None:
        loc["page"] = page
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id=art, atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value or {}, entity_keys=[],
        source_refs=[SourceRef(id=f"s{aid}", artifact_id=art, artifact_type=ArtifactType.docx,
                               filename=f"{art}.docx", locator=loc,
                               extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=conf,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def _all_text(atoms) -> str:
    return " ".join(f"{a.raw_text} {a.value}" for a in atoms)


# ── the invariant ───────────────────────────────────────────────────


def test_detail_names_zip_phone_email_and_instruction() -> None:
    w = _atom("w", "John Ozuna-Diaz | Client Support Manager", AtomType.stakeholder)
    l = _atom("l", "Call John Ozuna-Diaz at (555) 201-3344 or john@ox.com upon arrival, Findlay, OH 45840",
              AtomType.stakeholder)
    assert fold.detail_only_the_loser_states(w, l) == {
        "action:call", "phone:5552013344", "email:john@ox.com", "zip:45840"}
    assert fold.detail_only_the_loser_states(l, w) == frozenset()


def test_labels_and_street_numbers_are_not_detail() -> None:
    w = _atom("w", "Megan Lee", AtomType.stakeholder)
    for text in ("Megan Lee | Phone: | Contact:", "Megan Lee - Contact", "15733 US-224 Suite 12345"):
        assert not fold.detail_only_the_loser_states(w, _atom("l", text, AtomType.stakeholder)), text


# ── physical_site: the only address with its ZIP ────────────────────


def test_only_full_address_with_zip_survives_site_dedup() -> None:
    best = _atom("s1", "VC Links - 15733 US-224, Findlay, OH", AtomType.physical_site,
                 {"site_id": "VC Links", "name": "VC Links", "street_address": "15733 US-224",
                  "address": "15733 US-224, Findlay, OH", "city": "Findlay", "state": "OH"}, conf=0.9)
    with_zip = _atom("s2", "Site Address: 15733 US-224, Findlay, OH 45840", AtomType.physical_site,
                     {"site_id": "VC Links", "address": "15733 US-224, Findlay, OH 45840"}, art="b", conf=0.7)
    out = semantic_dedup_atoms([best, with_zip])
    sites = [a for a in out if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1, [a.raw_text for a in out]
    assert "45840" in sites[0].raw_text
    # ...and it still carries the structured fields of the copy it replaced.
    assert sites[0].value.get("city") == "Findlay" and sites[0].value.get("state") == "OH"


# ── atom_type_sanity: a contact row on the signature page ───────────


def test_contact_row_with_email_and_phone_is_not_folded_into_signature_record() -> None:
    rows = [
        _atom("r1", "OxBlue: Name: Megan Blevins | Customer: Name: Pat Doe", AtomType.signatory,
              {"name": "Megan Blevins"}, page=3),
        _atom("r2", "OxBlue: Title: Project Coordinator | Customer: Title: Director", AtomType.signatory, page=3),
        _atom("r3", "Name: Megan Blevins | Title: Project Coordinator | Email: megan.blevins@oxblue.com",
              AtomType.stakeholder, {"name": "Megan Blevins", "email": "megan.blevins@oxblue.com"}, page=3),
    ]
    merge_signature_rows(rows)
    assert "megan.blevins@oxblue.com" in _all_text(rows)
    # The signature rows themselves still merge into one record.
    assert sum(1 for a in rows if a.atom_type == AtomType.signatory) == 1


# ── stakeholder dedup: an action step is not a person record ────────


def test_action_step_naming_a_person_survives_stakeholder_dedup() -> None:
    person = _atom("p1", "John Ozuna-Diaz | Client Support Manager", AtomType.stakeholder,
                   {"name": "John Ozuna-Diaz", "role": "Client Support Manager"}, art="intake")
    step = _atom("p2", "Call Client Support Manager John Ozuna-Diaz upon arrival", AtomType.stakeholder,
                 {"name": "John Ozuna-Diaz"}, art="sow")
    out = dedupe_stakeholder_atoms([person, step])
    assert any("upon arrival" in a.raw_text for a in out), [a.raw_text for a in out]
    people = [a for a in out if a.atom_type == AtomType.stakeholder]
    assert [a.raw_text for a in people] == ["John Ozuna-Diaz | Client Support Manager"]
    kept = next(a for a in out if "upon arrival" in a.raw_text)
    assert kept.atom_type == AtomType.task and kept.value.get("contact_name") == "John Ozuna-Diaz"


def test_contact_row_with_phone_survives_stakeholder_dedup() -> None:
    bare = _atom("m1", "Megan Blevins | Project Coordinator", AtomType.stakeholder,
                 {"name": "Megan Blevins", "role": "Project Coordinator"}, art="intake")
    full = _atom("m2", "Megan Blevins | Project Coordinator | megan@oxblue.com | 706-555-0142",
                 AtomType.stakeholder, {"name": "Megan Blevins"}, art="sow")
    out = dedupe_stakeholder_atoms([bare, full])
    text = _all_text(out)
    assert "megan@oxblue.com" in text and "706-555-0142" in text


def test_plain_duplicates_still_fold() -> None:
    a = _atom("d1", "John Ozuna-Diaz | Client Support Manager", AtomType.stakeholder,
              {"name": "John Ozuna-Diaz"}, art="x")
    b = _atom("d2", "John Ozuna-Diaz, Client Support Manager", AtomType.stakeholder,
              {"name": "John Ozuna-Diaz"}, art="y")
    assert len(dedupe_stakeholder_atoms([a, b])) == 1
