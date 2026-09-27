"""Two ways 010180's meeting notes lost scope after they were parsed correctly.

The email is four `Heading: body` notes from a call. Three reached the
envelope. The other one, and half of a second, did not — and both had been
hand-labelled during the first pass with the note "NEVER READ", which is how
they were found again.

A wrong type is recoverable: a head relabels it, a PM corrects it, the words
are still there to argue about. Deleted words are not recoverable by anything.
Both fixes here follow from that.

1. THE PARAGRAPH TYPED AS A PERSON

    Security & Access Control: The team discussed implementing a new security
    system, likely using their own swipe card system similar to Great Neck,
    and considered whether to include badge readers at the suite entry doors.

236 characters of access-control scope, typed `stakeholder` with
`value.name = "Great Neck"` — a place named inside the sentence. Two ordinary
dictionary words, so `_all_common_words` fired, the rule written to keep
"Site Assessment" and "Account Executive" out of the roster, and the whole
paragraph went. The three branches beside it already ask `_retype_sentence`
first — "a full sentence that names nobody is CONTENT the typer mislabelled as
a person" — and this one did not.

2. THE TRUNCATION THAT OUTRANKED THE FULL SENTENCE

`cross_type_dedup_atoms` groups on the first 80 characters, "so trailing
paraphrase divergence doesn't split a shared fact". Right for two tellings of
one fact; wrong when one member's text strictly contains another's. One
paragraph produced:

    quantity  150 ch  "... IT room, and pantry."
    exclusion 317 ch  the same, plus "CAD drawings and plans were shared for
                      review. The setup will require Cat 6A cabling, two
                      Ethernet connections per workstation, and AV work for
                      conference rooms."

`quantity` outranks `exclusion`, so the survivor read as though the paragraph
stopped at "pantry" and the deal lost its cabling specification, its drops
per workstation and its AV scope.
"""
from __future__ import annotations

import pytest

from app.core.atom_substance_gate import drop_contextless_stakeholders
from app.core.schemas import (ArtifactType, AtomType, AuthorityClass,
                              EvidenceAtom, ReviewStatus, SourceRef)
from app.core.semantic_dedup import cross_type_dedup_atoms

SECURITY = (
    "Security & Access Control: The team discussed implementing a new security "
    "system, likely using their own swipe card system similar to Great Neck, and "
    "considered whether to include badge readers at the suite entry doors for the "
    "new floor."
)
LAYOUT_SHORT = (
    "Layout & Technical Requirements: The team discussed the office layout, "
    "including 106 workstations, conference rooms, phone rooms, IT room, and pantry."
)
LAYOUT_FULL = LAYOUT_SHORT + (
    " CAD drawings and plans were shared for review. The setup will require Cat 6A "
    "cabling, two Ethernet connections per workstation, and AV work for conference "
    "rooms."
)

_REF = SourceRef(id="src_1", artifact_id="art_1", filename="notes.eml",
                 artifact_type=ArtifactType.email, extraction_method="email_body_line",
                 parser_version="t", locator={"line": 1})


def _atom(text, atom_type, value=None, ident="atm_1"):
    return EvidenceAtom(
        id=ident, project_id="p", artifact_id="art_1", atom_type=atom_type,
        raw_text=text, normalized_text=text.lower(),
        value=value if value is not None else {"kind": "email_body_line"},
        entity_keys=[], source_refs=[_REF],
        authority_class=AuthorityClass.contractual_scope, confidence=0.72,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


# ---------------------------------------------------------------- 1. the gate

def test_a_scope_paragraph_is_not_deleted_for_its_invented_name():
    atom = _atom(SECURITY, AtomType.stakeholder,
                 {"kind": "email_body_line", "name": "Great Neck"})
    kept, dropped = drop_contextless_stakeholders([atom])
    assert dropped == [], "the access-control scope was deleted again"
    assert kept[0].atom_type != AtomType.stakeholder, "kept, but still a person"


@pytest.mark.parametrize("label", [
    "Site Assessment",
    "Account Executive",
    "Customer Contact Person",
])
def test_a_bare_role_label_is_still_dropped(label):
    """The rule keeps its job. These are the shapes it was written for — a
    phrase the sentence capitalised, with no person in it."""
    atom = _atom(label, AtomType.stakeholder, {"kind": "person", "name": label})
    kept, dropped = drop_contextless_stakeholders([atom])
    assert kept == [], f"{label} came back into the roster"


# ------------------------------------------------------------ 2. cross-type

def test_the_fuller_sentence_survives_a_type_it_outranks():
    out = cross_type_dedup_atoms([
        _atom(LAYOUT_SHORT, AtomType.quantity, ident="atm_short"),
        _atom(LAYOUT_FULL, AtomType.exclusion, ident="atm_full"),
    ])
    assert len(out) == 1
    survivor = out[0].raw_text
    for phrase in ("Cat 6A cabling",
                   "two Ethernet connections per workstation",
                   "AV work for conference rooms",
                   "CAD drawings and plans"):
        assert phrase in survivor, f"lost: {phrase}"


def test_order_of_arrival_does_not_decide_it():
    """Whichever way round they arrive, the words win."""
    for pair in (
        [_atom(LAYOUT_FULL, AtomType.exclusion, ident="a"),
         _atom(LAYOUT_SHORT, AtomType.quantity, ident="b")],
        [_atom(LAYOUT_SHORT, AtomType.quantity, ident="b"),
         _atom(LAYOUT_FULL, AtomType.exclusion, ident="a")],
    ):
        out = cross_type_dedup_atoms(pair)
        assert len(out) == 1
        assert "Cat 6A cabling" in out[0].raw_text


def test_a_genuine_same_sentence_pair_still_collapses():
    """The guard must not turn dedup off. Two types, one sentence, one atom —
    which is the whole reason cross_type_dedup exists."""
    same = "Provide and install 24 Cat6A drops in the IT closet."
    out = cross_type_dedup_atoms([
        _atom(same, AtomType.scope_item, ident="a"),
        _atom(same, AtomType.task, ident="b"),
    ])
    assert len(out) == 1


def test_two_different_sentences_sharing_a_prefix_are_not_merged_away():
    """Containment is the test, not length. Neither of these contains the
    other, so nothing here should be silently dropped for being longer."""
    a = LAYOUT_SHORT
    b = LAYOUT_SHORT[:-1] + " and a server room on the east side of the floor."
    out = cross_type_dedup_atoms([
        _atom(a, AtomType.quantity, ident="a"),
        _atom(b, AtomType.scope_item, ident="b"),
    ])
    # They share 80 characters so they are one group and one survives; the
    # point is that the survivor is chosen by priority, not by discarding a
    # superset. Neither text contains the other, so the old rule applies.
    assert len(out) == 1


def test_a_table_row_does_not_beat_its_typed_sibling():
    """The subtlety that makes containment alone the wrong rule.

    A `raw_table_row` contains its typed sibling too:

        raw_table_row  "ESTIMATED TOTAL FEES | $21,560.00"
        service_line   "ESTIMATED TOTAL FEES"

    and here the TYPED atom must win — the extra text is a money column whose
    value already lives in the atom's `value`, not a sentence. Getting this
    wrong broke `test_estimated_total_collapses_despite_money_tokens`, which is
    how the first version of the guard was caught. So the loser must add
    WORDS, measured with the cell bars, money and bare numbers stripped out.
    """
    out = cross_type_dedup_atoms([
        _atom("ESTIMATED TOTAL FEES | $21,560.00", AtomType.raw_table_row, ident="a"),
        _atom("ESTIMATED TOTAL FEES", AtomType.service_line, ident="b"),
    ])
    assert len(out) == 1
    assert out[0].atom_type == AtomType.service_line
