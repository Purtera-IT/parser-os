"""Envelope invariance — the seam audit's fifth stop (2026-08-25).

Three root causes fixed: underscore word-boundaries in account matching,
leading whitespace defeating the foreign-artifact check, and a `continue`
that dropped non-quote-line task atoms from every entity index (and only
when a classifier import succeeded). The held invariants are pinned too.
"""

import json
import random

import pytest

from app.core.ids import stable_id
from app.core.orbitbrief_envelope import (
    _account_match,
    _build_indexes,
    _compact_atom,
    _deal_number_from_crm,
    _foreign_artifacts,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

_CRM = {"account_name": "CDW", "deal_name": "010999 - CDW Checkout Wireless"}


def _atom(atom_id, atom_type, entity_keys, *, text="text", artifact_id="art_1"):
    return EvidenceAtom(
        id=atom_id, project_id="proj_1", artifact_id=artifact_id,
        atom_type=atom_type, raw_text=text, normalized_text=text.lower(),
        value={"text": text}, entity_keys=entity_keys,
        source_refs=[SourceRef(
            id=stable_id("src", atom_id), artifact_id=artifact_id,
            artifact_type=ArtifactType.txt, filename="fixture.txt",
            locator={}, extraction_method="test", parser_version="test")],
        authority_class=AuthorityClass.customer_current_authored, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[],
        parser_version="test",
    )


class TestAccountMatchDress:
    @pytest.mark.parametrize("fname,want", [
        ("010999 - CDW Quote.xlsx", "same"),
        ("010999 - CDW_Quote.xlsx", "same"),   # FIXED: "_" is a word char,
        ("010999-cdw-quote.xlsx", "same"),     # so \bCDW\b never matched
        ("010999 - Deal Kit.xlsx", "unknown"),  # generic name is not evidence
    ])
    def test_separator_is_dress(self, fname, want):
        assert _account_match(_CRM, fname) == want


class TestForeignArtifactDress:
    def test_every_dressing_of_a_foreign_number_is_flagged(self):
        docs = [
            {"filename": "010500 - Marcos New Store Installs.pdf", "artifact_id": "a1"},
            {"filename": " 010500 - Marcos New Store Installs.pdf", "artifact_id": "a2"},
            {"filename": "010500_Marcos New Store Installs.pdf", "artifact_id": "a3"},
        ]
        flagged = {d["artifact_id"] for d in _foreign_artifacts(crm=_CRM, documents=docs)}
        assert flagged == {"a1", "a2", "a3"}

    def test_adjacent_numbers_stay_unflagged(self):
        docs = [{"filename": "011000 - CDW sibling.pdf", "artifact_id": "s1"}]
        assert _foreign_artifacts(crm=_CRM, documents=docs) == []

    @pytest.mark.parametrize("deal_name", [
        "010114 - CDW Checkout", "  010114 - CDW Checkout",
        "010114-CDW Checkout", "010114_CDW Checkout",
    ])
    def test_deal_number_survives_dress(self, deal_name):
        assert _deal_number_from_crm({"deal_name": deal_name}) == "010114"


class TestIndexInvariants:
    def test_task_atoms_keep_their_entity_keys(self):
        # FIXED: a `continue` dropped non-quote-line tasks from every
        # entity index -- and only when the tier-classifier import
        # succeeded, so behaviour depended on the environment.
        task = _atom("t1", AtomType.task, ["site:hq", "stakeholder:bob"],
                     text="Coordinate access with Bob at HQ")
        req = _atom("r1", AtomType.requirement, ["site:hq"])
        idx = _build_indexes(atoms=[task, req])
        assert "t1" in idx["atoms_by_entity_key"]["site:hq"]
        assert "t1" in idx["atoms_by_stakeholder_slug"]["bob"]
        assert "t1" in idx["atoms_by_site_slug"]["hq"]

    def test_indexes_are_input_order_invariant(self):
        atoms = [
            _atom("a1", AtomType.requirement, ["site:hq"], artifact_id="A"),
            _atom("a2", AtomType.exclusion, ["site:hq", "device:ap"], artifact_id="B"),
            _atom("a3", AtomType.scope_item, ["device:ap"], artifact_id="A"),
            _atom("a4", AtomType.constraint, ["stakeholder:kim"], artifact_id="C"),
        ]
        base = _build_indexes(atoms=list(atoms))
        for seed in (3, 9):
            shuffled = list(atoms)
            random.Random(seed).shuffle(shuffled)
            assert _build_indexes(atoms=shuffled) == base

    def test_compact_atom_serializes_with_plain_strings(self):
        c = _compact_atom(_atom("x1", AtomType.requirement, ["site:hq"]))
        json.dumps(c)  # must not raise
        assert isinstance(c["atom_type"], str)
        assert isinstance(c["review_status"], str)
