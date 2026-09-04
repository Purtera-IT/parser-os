"""An atom minted from a table row must carry that row.

Grounded in deal 8aa9051a (Fortinet rollout, 142 physical_site atoms), where
89 site names could not be checked against their own source. Two distinct
faults hid behind that one number, and this file pins both apart:

1. THE ROW NEVER SHIPPED. The XLSX roster minter bound the full row onto the
   atom under ``value["cells"]`` — a key
   ``semantic_dedup._PHYSICAL_SITE_ALLOWED_FIELDS`` does not whitelist, so
   ``_clean_physical_site_value`` deleted it from every physical_site atom
   before the envelope was written. The row is now carried under the keys
   that set DOES whitelist (``raw_cells`` / ``row_index``), and the composed
   summary is completed with the columns the field map could not place, so
   ``atom_type_sanity.strip_unsupported_names`` can decide a roster name
   instead of abstaining on it.

2. THE NAME WAS NEVER IN THE ROW. On 8aa9051a the roster ships two columns —
   ``Site Code`` and ``Physical address``. "Richmond Office (RIC)" is in
   neither, nor anywhere else in either source artifact: it is synthesized
   downstream (``site_facility_head`` builds ``f"{city} Office"``). Carrying
   the source row must NOT make that name pass. ``test_a_fabricated_name_is
   _still_unsupported`` is the load-bearing test in this file: without it,
   "make the invariant decidable" degrades into "make everything decidable
   and true", which is a worse bug than the one being fixed.

Non-vacuity discipline follows ``tests/test_promotion_gate_invariants.py``:
every fixture is asserted to actually exercise its rule, so a regression that
silently stops the rule firing fails here rather than passing quietly.
"""

from __future__ import annotations

import random

from openpyxl import Workbook

from app.core.atom_type_sanity import strip_unsupported_names
from app.core.orbitbrief_envelope import _compact_atom
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.semantic_dedup import _clean_physical_site_value
from app.parsers.site_roster_extractor import (
    SOURCE_ROW_MAX_CELL_CHARS,
    SOURCE_ROW_MAX_CELLS,
    capped_source_row,
    map_columns_to_fields,
)
from app.parsers.xlsx_parser import XlsxParser

# ── The real 8aa9051a "Locations" sheet shape ────────────────────
#
# Verbatim: two columns the mapper places, one ("Ceiling Heights") it cannot,
# and NO facility-name column anywhere. That absence is the point — it is why
# every display name on this deal is manufactured rather than read.
_LOCATIONS_COLUMNS = ["Site Code", "Physical address", "Ceiling Heights"]
_LOCATIONS_ROWS = [
    ["RIC-6802-RICHMOND-VA-CORP", "6802 Paragon Pl, Richmond, VA 23230, USA", ""],
    ["8640-TAMPA-FL-LM", "8640 Elm Fair Boulevard, Tampa, FL 33610", "12\""],
    ["LFO-3680-FAIRFIELD-OH-LM", "3680 Port Union Road Suite 400, Fairfield, OH 45014", "15' - 17'"],
]

# ── A roster that DOES name its facilities, in a column the field
#    map cannot place. This is the shape the fix is for. ──────────
_NAMED_COLUMNS = ["Site ID", "Street Address", "Marquee", "Ceiling Heights"]
_NAMED_ROWS = [
    ["ATL-HQ-01", "1100 Peachtree St NE", "Peachtree Tower", "14'"],
    ["ATL-WEST-02", "455 Marietta St NW", "Westside Annex", "12'"],
    ["BHM-03", "2200 Richard Arrington Blvd", "Birmingham Operations Center", "16'"],
]


def _site_atoms(path):
    result = XlsxParser().parse_artifact(
        project_id="proj_test", artifact_id="art_test", path=path
    )
    atoms = result if isinstance(result, list) else result.atoms
    return [a for a in atoms if a.atom_type == AtomType.physical_site]


def _write(path, columns, rows, sheet="Locations"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(columns)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


def _atom(atom_id, text, value, *, artifact_id="art_real"):
    """A minimal physical_site atom in the shape the compiler hands the gate."""
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_test",
        artifact_id=artifact_id,
        atom_type=AtomType.physical_site,
        raw_text=text,
        normalized_text=text.lower(),
        value=dict(value),
        entity_keys=[],
        source_refs=[
            SourceRef(
                id=f"src_{atom_id}",
                artifact_id=artifact_id,
                artifact_type=ArtifactType.xlsx,
                filename="roster.xlsx",
                locator={"sheet": "Locations", "row": 2},
                extraction_method="xlsx_site_roster_v1",
                parser_version="xlsx_parser_v1",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.85,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="xlsx_parser_v1",
    )


def _name_probe(name: str, support: str) -> bool:
    """The gate's own question, spelled out so a test can assert it directly."""
    from app.core.atom_type_sanity import _normalize_name

    return _normalize_name(name) in _normalize_name(support)


# ══ 0. guard the guards ════════════════════════════════════════════


def test_fixtures_are_not_vacuous():
    """The named roster must have an UNMAPPABLE name column; the 8aa9051a
    roster must have no name column at all. Both claims are load-bearing."""
    named_map = map_columns_to_fields(_NAMED_COLUMNS)
    assert "facility_name" not in set(named_map.values()), (
        "'Marquee' must stay unmapped — otherwise this fixture "
        "tests the canonical path, not the gap the source row closes"
    )
    assert 3 not in named_map  # "Ceiling Heights" unmapped too

    loc_map = map_columns_to_fields(_LOCATIONS_COLUMNS)
    assert "facility_name" not in set(loc_map.values())
    assert not any(
        "office" in str(c).lower() or "name" in str(c).lower()
        for row in _LOCATIONS_ROWS
        for c in row
    ), "the 8aa9051a fixture must contain no facility name anywhere"


# ══ 1. the source row survives into the envelope ═══════════════════


def test_roster_atom_carries_its_source_row(tmp_path):
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    atoms = _site_atoms(path)
    assert len(atoms) == 3, "fixture must mint one atom per roster row"

    for atom in atoms:
        cells = atom.value.get("raw_cells")
        assert cells, f"{atom.id}: no source row bound"
        headers = [h for h, _ in cells]
        assert headers == _NAMED_COLUMNS, "row must keep column order and names"
        assert atom.value.get("row_index") is not None

    # Non-vacuity: the row carries a cell that NO canonical field holds.
    canonical = {"site_id", "facility_name", "street_address", "city", "state", "zip"}
    first = atoms[0]
    carried = {v for _, v in first.value["raw_cells"]}
    assert "Peachtree Tower" in carried
    assert not any(first.value.get(f) == "Peachtree Tower" for f in canonical)


def test_source_row_survives_dedup_and_reaches_the_envelope(tmp_path):
    """The regression that hid the row: a value key outside
    ``_PHYSICAL_SITE_ALLOWED_FIELDS`` is deleted before serialization."""
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    atoms = _site_atoms(path)
    assert atoms

    for atom in atoms:
        atom.value = _clean_physical_site_value(dict(atom.value))
        assert atom.value.get("raw_cells"), (
            f"{atom.id}: the physical_site whitelist dropped the source row"
        )

    compacted = [_compact_atom(a) for a in atoms]
    assert all(c["structured"].get("raw_cells") for c in compacted)
    assert compacted[0]["structured"]["raw_cells"][2] == [
        "Marquee",
        "Peachtree Tower",
    ]

    # Non-vacuity: prove the old key would NOT have survived, so this test
    # fails if someone reverts to an unwhitelisted field name.
    smuggled = _clean_physical_site_value(
        {"kind": "physical_site", "site_id": "ATL-HQ-01", "cells": {"a": "b"}}
    )
    assert "cells" not in smuggled


def test_text_keeps_its_composed_summary(tmp_path):
    """The summary downstream reads is extended, never replaced or reordered."""
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    atoms = _site_atoms(path)
    first = next(a for a in atoms if a.value.get("site_id") == "ATL-HQ-01")
    assert first.raw_text.startswith("site_id: ATL-HQ-01 | address: 1100 Peachtree St NE")
    assert _compact_atom(first)["text"] == first.raw_text


# ══ 2. a name in a roster column is decidably SUPPORTED ════════════


def test_name_in_an_unmapped_roster_column_is_decidably_supported(tmp_path):
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    atoms = _site_atoms(path)
    assert atoms

    # Give each atom the display name its own row carries — exactly what a
    # naming stage should do — then let the SHIPPED gate judge it.
    names = {
        "ATL-HQ-01": "Peachtree Tower",
        "ATL-WEST-02": "Westside Annex",
        "BHM-03": "Birmingham Operations Center",
    }
    for atom in atoms:
        atom.value["name"] = names[atom.value["site_id"]]
        atom.value["display_name"] = names[atom.value["site_id"]]

    # Non-vacuity: the CANONICAL-fields-only summary — what the minter
    # produced before this change — does not contain the name, so the gate
    # would have stripped every one of these legitimate names.
    for atom in atoms:
        canonical_only = " | ".join(
            f"{label}: {atom.value[key]}"
            for label, key in (("site_id", "site_id"), ("address", "street_address"))
            if atom.value.get(key)
        )
        assert not _name_probe(atom.value["name"], canonical_only), (
            "fixture must exercise the rule: the name must be absent from the "
            "old canonical-only summary"
        )
        assert _name_probe(atom.value["name"], atom.raw_text)

    changed = strip_unsupported_names(atoms)
    assert changed == 0, "a name read out of its own roster row must PASS"
    for atom in atoms:
        assert atom.value["name"] == names[atom.value["site_id"]]
        assert "unsupported_name_stripped" not in (atom.review_flags or [])


# ══ 3. THE CRITICAL TEST — a fabricated name is still caught ═══════


def test_a_fabricated_name_is_still_unsupported(tmp_path):
    """8aa9051a, verbatim. "Richmond Office (RIC)" is in no column of this
    roster; carrying the row must not launder it into a supported name."""
    path = _write(tmp_path / "010004 Deal Kit.xlsx", _LOCATIONS_COLUMNS, _LOCATIONS_ROWS)
    atoms = _site_atoms(path)
    assert len(atoms) == 3

    fabricated = {
        "RIC-6802-RICHMOND-VA-CORP": "Richmond Office (RIC)",
        "8640-TAMPA-FL-LM": "Tampa Office",
        "LFO-3680-FAIRFIELD-OH-LM": "Fairfield Office",
    }
    for atom in atoms:
        atom.value["name"] = fabricated[atom.value["site_id"]]
        atom.value["display_name"] = fabricated[atom.value["site_id"]]

    # Non-vacuity: the row IS bound — so a failure here is the gate going
    # blind, not the row going missing.
    assert all(a.value.get("raw_cells") for a in atoms)
    for atom in atoms:
        row_text = " ".join(v for _, v in atom.value["raw_cells"])
        assert not _name_probe(atom.value["name"], row_text), (
            "fixture must exercise the rule: the name must be absent from the row"
        )

    changed = strip_unsupported_names(atoms)
    assert changed == 3, "every manufactured name must still be caught"
    for atom in atoms:
        assert atom.value.get("name") != fabricated[atom.value["site_id"]]
        assert "unsupported_name_stripped" in atom.review_flags
        assert atom.review_status == ReviewStatus.needs_review


def test_carrying_the_row_does_not_make_everything_pass(tmp_path):
    """The two halves together: in ONE batch, real names survive and
    manufactured ones do not."""
    named = _write(tmp_path / "Named.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    unnamed = _write(tmp_path / "Unnamed.xlsx", _LOCATIONS_COLUMNS, _LOCATIONS_ROWS)

    real = _site_atoms(named)
    for atom in real:
        atom.value["name"] = dict(atom.value["raw_cells"])["Marquee"]
    ghosts = _site_atoms(unnamed)
    for atom in ghosts:
        atom.value["name"] = "Richmond Office (RIC)"

    batch = real + ghosts
    assert strip_unsupported_names(batch) == len(ghosts)
    assert all(a.value.get("name") for a in real)


# ══ 4. size cap ════════════════════════════════════════════════════


def test_source_row_respects_the_size_cap():
    wide = [[f"Column {i}", "x" * 500] for i in range(SOURCE_ROW_MAX_CELLS * 4)]
    # Non-vacuity: the input must actually exceed both caps.
    assert len(wide) > SOURCE_ROW_MAX_CELLS
    assert len(wide[0][1]) > SOURCE_ROW_MAX_CELL_CHARS

    capped = capped_source_row(wide)
    assert len(capped) == SOURCE_ROW_MAX_CELLS
    assert all(len(h) <= SOURCE_ROW_MAX_CELL_CHARS for h, _ in capped)
    assert all(len(v) <= SOURCE_ROW_MAX_CELL_CHARS for _, v in capped)


def test_empty_cells_are_dropped_not_carried():
    capped = capped_source_row(
        [["Site Code", "RIC-1"], ["Ceiling Heights", ""], ["Notes", None], ["Zip", "23230"]]
    )
    assert capped == [["Site Code", "RIC-1"], ["Zip", "23230"]]


def test_envelope_serializer_re_caps_a_merged_row():
    """``semantic_dedup`` concatenates list fields when it merges two atoms,
    so the mint-time cap alone cannot bound the envelope."""
    oversized = [[f"C{i}", "y" * 400] for i in range(SOURCE_ROW_MAX_CELLS * 3)]
    atom = _atom(
        "atm_merged",
        "site_id: ATL-HQ-01 | address: 1100 Peachtree St NE",
        {"kind": "physical_site", "site_id": "ATL-HQ-01", "raw_cells": oversized},
    )
    shipped = _compact_atom(atom)["structured"]["raw_cells"]
    assert len(shipped) == SOURCE_ROW_MAX_CELLS
    assert all(len(v) <= SOURCE_ROW_MAX_CELL_CHARS for _, v in shipped)


def test_roster_summary_stays_bounded(tmp_path):
    wide_columns = ["Site ID", "Street Address"] + [f"Extra {i}" for i in range(400)]
    wide_row = ["ATL-HQ-01", "1100 Peachtree St NE"] + ["z" * 300 for _ in range(400)]
    path = _write(tmp_path / "Wide Roster.xlsx", wide_columns, [wide_row, wide_row])
    atoms = _site_atoms(path)
    assert atoms, "fixture must still be recognised as a roster"
    assert len(" | ".join(wide_row)) > XlsxParser._SITE_ROW_TEXT_MAX_CHARS
    for atom in atoms:
        assert len(atom.raw_text) <= XlsxParser._SITE_ROW_TEXT_MAX_CHARS


# ══ 5. dress invariance — order and idempotency ════════════════════


def test_source_row_is_stable_across_repeated_serialization(tmp_path):
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)
    atoms = _site_atoms(path)
    assert atoms
    once = [_compact_atom(a) for a in atoms]
    twice = [_compact_atom(a) for a in atoms]
    assert once == twice
    # ...and the row is unchanged by being serialized.
    assert [a.value["raw_cells"] for a in atoms] == [c["structured"]["raw_cells"] for c in once]


def test_source_row_is_order_invariant(tmp_path):
    """Shuffling the atom list changes no atom's row, and the gate's verdict
    is per-atom rather than order-dependent."""
    path = _write(tmp_path / "Site Roster.xlsx", _NAMED_COLUMNS, _NAMED_ROWS)

    def build():
        atoms = _site_atoms(path)
        for atom in atoms:
            atom.value["name"] = dict(atom.value["raw_cells"])["Marquee"]
        return atoms

    straight = build()
    assert strip_unsupported_names(straight) == 0
    expected = {a.value["site_id"]: a.value["raw_cells"] for a in straight}

    shuffled = build()
    random.Random(11).shuffle(shuffled)
    assert strip_unsupported_names(shuffled) == 0
    assert {a.value["site_id"]: a.value["raw_cells"] for a in shuffled} == expected


def test_column_order_is_preserved_not_sorted():
    """Determinism the envelope relies on: the row ships as read."""
    row = [["Zip", "23230"], ["Site Code", "RIC-1"], ["Address", "6802 Paragon Pl"]]
    assert capped_source_row(row) == row
