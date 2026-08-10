"""The routing representation must be dense, complete, and stable.

Measured over 86 gold-labelled deals with live envelopes, fitting the same
TF-IDF classifier under identical folds on each representation:

    v1               accuracy 0.610   macro-F1 0.371   ~880 tokens
    v2 (40 lines)    accuracy 0.690   macro-F1 0.496   ~1,408 tokens

macro-F1 is the number that matters — it weights the small classes equally, and
the small classes are exactly where the router fails.
"""
from __future__ import annotations

from app.core.scope_summary import SCOPE_SUMMARY_VERSION, build_scope_summary


def atom(atom_type: str, text: str) -> dict:
    return {"atom_type": atom_type, "text": text}


ROSTER = [
    atom("physical_site", f"HC: {i} | Zone: {i % 5} | Address: {i} Main St | "
                          f"City: Town{i} | ST: TX | Zip: 7{i:04d}")
    for i in range(60)
]


def test_the_version_is_stamped():
    """A head distilled from a teacher must see the identical representation at
    inference; a corpus that mixes two formats is a corpus that rots."""
    out = build_scope_summary([atom("scope_item", "Install 40 access points")], [])
    assert SCOPE_SUMMARY_VERSION in out


def test_repeated_row_shapes_collapse_to_one_exemplar_and_a_count():
    """437 roster rows used to eat a third of the budget saying the same thing."""
    out = build_scope_summary(ROSTER, [])
    roster_lines = [ln for ln in out.splitlines() if "Zone:" in ln]
    assert len(roster_lines) == 1, f"expected one exemplar, got {len(roster_lines)}"
    assert "x60 similar rows" in roster_lines[0]


def test_the_count_survives_even_though_the_rows_do_not():
    """Collapsing must not hide scale — "there are 60 of these" is the signal."""
    out = build_scope_summary(ROSTER, [])
    assert "sites 60" in out


def test_label_only_rows_are_dropped():
    """Clayton spent three of nineteen lines on the word "Quantity"."""
    atoms = [atom("scope_item", "Quantity Yes"), atom("scope_item", "Quantity No"),
             atom("scope_item", "Pull cat6 to 40 drops and terminate")]
    out = build_scope_summary(atoms, [])
    assert "Quantity Yes" not in out
    assert "cat6" in out


def test_exclusions_get_their_own_section():
    """Short, highly discriminative, and none of Clayton's reached the router."""
    atoms = [
        atom("exclusion", "Cabling is out of scope and provided by others."),
        atom("scope_item", "Mount 12 displays in conference rooms"),
    ]
    out = build_scope_summary(atoms, [])
    assert "EXCLUSIONS:" in out
    assert "out of scope" in out


def test_commercial_shape_reads_people_pricing():
    """The load-bearing aggregate: staff-aug prices PEOPLE. "tech rate" appears
    2.06 times per staff-aug deal and 0.00 times in every other class."""
    atoms = [atom("rate_card", "Networking L1 Technician 2 hr. min: 73.5"),
             atom("scope_item", "Dispatch technicians to 437 retail stores")]
    out = build_scope_summary(atoms, [])
    assert "labour rate card present" in out
    assert "materials BOM present" not in out


def test_commercial_shape_reads_materials_pricing():
    atoms = [atom("bom_line", "Part #: CAB-6-BLU | Mfg: Panduit | Unit Price: 42.00"),
             atom("scope_item", "Install cabling at 12 sites")]
    out = build_scope_summary(atoms, [])
    assert "materials BOM present" in out


def test_bom_vocabulary_is_not_thrown_away():
    """Regression guard on a mistake made building this: excluding bom_line
    dropped `cables`, `runs`, `feet` and `surface` — cabling-domain words that
    live in BOM rows — and cost a discriminative token on 84 of 86 gold deals.
    Stratification below already stops BOM owning the budget."""
    atoms = [atom("bom_line", "Cat6 cable runs, 4200 feet, surface raceway")]
    out = build_scope_summary(atoms, [])
    assert "cable" in out.lower() and "feet" in out.lower()


def test_one_chatty_type_cannot_own_the_budget():
    """Stratified round-robin is what makes it safe to keep BOM in."""
    atoms = ROSTER + [atom("scope_item", f"Distinct scope statement number {i}")
                      for i in range(5)]
    out = build_scope_summary(atoms, [], max_scope_lines=8)
    assert "scope_item" in out, "the rare type was crowded out"


def test_aggregates_the_router_cannot_infer_from_a_sample():
    atoms = ROSTER + [atom("scope_item", "Install access points at each store")]
    out = build_scope_summary(atoms, [{"filename": "Dispatch_Readiness.xlsx"}])
    assert "SHAPE:" in out and "ATOM TYPES:" in out
    assert "physical_site 60" in out
    assert "Dispatch_Readiness" in out


def test_it_is_deterministic():
    """Required twice over: the result is cached per envelope, and it is the
    training-row key. A representation that varies run to run poisons both."""
    atoms = ROSTER + [atom("scope_item", "Install access points")]
    docs = [{"filename": "a.xlsx"}, {"filename": "b.docx"}]
    assert build_scope_summary(atoms, docs) == build_scope_summary(atoms, docs)


def test_an_empty_deal_does_not_explode():
    out = build_scope_summary([], [])
    assert SCOPE_SUMMARY_VERSION in out and "SCOPE:" in out


def test_it_reads_evidence_atom_objects_too():
    """Envelope dicts carry `text`; EvidenceAtom carries `raw_text`. Reading only
    one of them is how the router once got an empty summary and routed
    everything to wireless."""
    class _Atom:
        atom_type = "scope_item"
        raw_text = "Install 40 access points and survey coverage"

    out = build_scope_summary([_Atom()], [])
    assert "access points" in out
