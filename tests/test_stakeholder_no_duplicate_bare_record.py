"""A name-only stakeholder record must not survive alongside a fuller one.

Deal 010215: _structural_people_atoms' bare owner_re/delegate_re fallback
(role defaults to the literal string "stakeholder", no email, no phone) fired
a second time on the same ten on-site contacts contact_property_block had
already emitted correctly -- because it works from entity_keys + a raw-text
name match and has no way to know a fuller record for that identity already
exists. The two never deduped downstream: this fallback's key is a slug of
the name ("rosalyn_hemingway"); the real atom's dedup key is its email.
Different keys, both survive.
"""

from __future__ import annotations

from app.core.entity_extraction import enrich_atoms
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.domain import load_domain_pack

_PACK = load_domain_pack("default")


def _person_atom(artifact_id, name, email):
    return EvidenceAtom(
        id=f"atm_{artifact_id}_person",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.stakeholder,
        raw_text=f"On Site Contact: {name}",
        normalized_text=name.lower(),
        value={"role": "On Site Contact", "kind": "person", "name": name, "email": email},
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        entity_keys=[],
        parser_version="test",
    )


def _mention_atom(artifact_id, text, key="stakeholder:rosalyn_hemingway"):
    """A different atom in the same document that happens to mention the
    contact's name in passing -- the kind of text the bare fallback fires on."""
    return EvidenceAtom(
        id=f"atm_{artifact_id}_mention",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        value={},
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        entity_keys=[key],
        source_refs=[
            SourceRef(
                id=f"src_{artifact_id}_mention",
                artifact_id=artifact_id,
                artifact_type=ArtifactType.docx,
                filename=f"{artifact_id}.docx",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        parser_version="test",
    )


def test_a_bare_fallback_record_does_not_duplicate_a_real_one():
    atoms = [
        _person_atom("doc1", "Rosalyn Hemingway", "rosalyn.hemingway@sodexo.com"),
        _mention_atom("doc1", "Coordinate with Rosalyn Hemingway before Owner: Rosalyn Hemingway signs off"),
    ]
    enrich_atoms(atoms, _PACK)
    people = [
        a for a in atoms
        if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder"
        and a.value.get("kind") == "person"
        and a.value.get("name") == "Rosalyn Hemingway"
    ]
    assert len(people) == 1, f"expected exactly one Rosalyn Hemingway record, got {len(people)}: {[p.value for p in people]}"
    assert people[0].value.get("email") == "rosalyn.hemingway@sodexo.com"


def test_the_guard_only_fires_when_a_fuller_record_actually_exists():
    """One-sided: refuse a slug already covered by a fuller record, but never
    invent coverage that is not there. No _person_atom in this fixture, so the
    guard's _covered_slugs set is empty and must not suppress anything."""
    from app.core.entity_extraction import _structural_people_atoms

    atom = _mention_atom("doc2", "Owner: Jamie Delacroix approved the change order",
                          key="stakeholder:jamie_delacroix")
    out = _structural_people_atoms([atom], project_id="p")
    names = {a.value.get("name") for a in out}
    assert "Jamie Delacroix" in names, (
        f"the guard blocked a name it was never taught to cover: {names}"
    )
