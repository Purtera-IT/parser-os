"""Pre-enrich table rollup backstop — folds high-cardinality money-bearing
spreadsheet tables the parser's sheet-classifier missed, so they never hit
the per-atom LLM enrich pass or flood the training store.
"""

from __future__ import annotations

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.table_rollup import roll_up_table_rows


def _row_atom(
    *,
    idx: int,
    columns: list[str],
    row: list,
    sheet: str = "Rates",
    artifact_id: str = "art1",
    filename: str = "rates.xlsx",
) -> EvidenceAtom:
    text = " | ".join(str(c) for c in row if c not in (None, ""))
    aid = f"atm_row_{sheet}_{idx}"
    src = SourceRef(
        id=f"src_{aid}",
        artifact_id=artifact_id,
        artifact_type=ArtifactType.xlsx,
        filename=filename,
        locator={"sheet": sheet, "row": idx + 1},
        extraction_method="raw_table_row_v49_2",
        parser_version="xlsx_test",
    )
    return EvidenceAtom(
        id=aid,
        project_id="proj1",
        artifact_id=artifact_id,
        atom_type=AtomType.raw_table_row,
        raw_text=text,
        normalized_text=text.lower(),
        value={
            "_columns": list(columns),
            "_row": list(row),
            "_row_idx": idx,
            "_sheet": sheet,
            "_filename": filename,
            "_artifact_type": "xlsx",
        },
        source_refs=[src],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.5,
        review_status=ReviewStatus.needs_review,
        parser_version="xlsx_test",
    )


def _prose_atom(aid: str = "atm_prose") -> EvidenceAtom:
    src = SourceRef(
        id=f"src_{aid}",
        artifact_id="art2",
        artifact_type=ArtifactType.transcript,
        filename="kickoff.txt",
        locator={"line_start": 3},
        extraction_method="prose",
        parser_version="x",
    )
    return EvidenceAtom(
        id=aid,
        project_id="proj1",
        artifact_id="art2",
        atom_type=AtomType.scope_item,
        raw_text="Install cameras in the west wing.",
        normalized_text="install cameras in the west wing.",
        source_refs=[src],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.needs_review,
        parser_version="x",
    )


def _type(atom) -> str:
    at = atom.atom_type
    return at.value if hasattr(at, "value") else str(at)


def test_high_cardinality_money_table_folds_to_one_summary() -> None:
    cols = ["Part No", "Description", "Unit Price"]
    atoms = [
        _row_atom(idx=i, columns=cols, row=[f"PN-{i:04d}", f"Widget {i}", 10.0 + i])
        for i in range(60)
    ]
    out, stats = roll_up_table_rows(atoms)

    assert stats["groups_folded"] == 1
    assert stats["rows_folded"] == 60
    summaries = [a for a in out if _type(a) == "pricing_assumption"]
    assert len(summaries) == 1
    # The 60 per-row atoms collapse to the single summary.
    assert len([a for a in out if _type(a) == "raw_table_row"]) == 0
    assert len(out) == 1

    s = summaries[0]
    assert s.value["is_summary"] is True
    assert s.value["line_count"] == 60
    assert s.value["_rolled_up"] is True
    # Lossless drill-down: every row preserved in value.rows.
    assert isinstance(s.value["rows"], list)
    assert len(s.value["rows"]) == 60
    assert s.value["rows"][0]["cells"]  # cells retained
    assert "pricing_rollup" in s.review_flags


def test_small_table_left_granular() -> None:
    cols = ["Part No", "Unit Price"]
    atoms = [_row_atom(idx=i, columns=cols, row=[f"PN-{i}", 5.0 + i]) for i in range(10)]
    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 0
    assert len([a for a in out if _type(a) == "raw_table_row"]) == 10
    assert not [a for a in out if _type(a) == "pricing_assumption"]


def test_non_money_table_left_granular() -> None:
    # No money column → per-row extraction signal must survive.
    cols = ["Site", "Room", "Device"]
    atoms = [
        _row_atom(idx=i, columns=cols, row=[f"Site {i}", f"Room {i}", "Camera"])
        for i in range(80)
    ]
    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 0
    assert len([a for a in out if _type(a) == "raw_table_row"]) == 80


def test_distinct_schemas_grouped_separately_and_order_preserved() -> None:
    money_cols = ["Part No", "Unit Price"]
    big = [_row_atom(idx=i, columns=money_cols, row=[f"PN-{i}", 12.0 + i], sheet="Rates") for i in range(50)]
    small_other = [
        _row_atom(idx=i, columns=["Site", "Room"], row=[f"S{i}", f"R{i}"], sheet="Sites")
        for i in range(5)
    ]
    prose = _prose_atom()
    atoms = [prose, *big, *small_other]

    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 1
    # Prose atom stays first; big table folds to one summary; small non-money
    # table untouched.
    assert _type(out[0]) == "scope_item"
    assert len([a for a in out if _type(a) == "pricing_assumption"]) == 1
    assert len([a for a in out if _type(a) == "raw_table_row"]) == 5


def test_empty_and_no_table_inputs_are_noops() -> None:
    assert roll_up_table_rows([])[0] == []
    only_prose = [_prose_atom("p1"), _prose_atom("p2")]
    out, stats = roll_up_table_rows(only_prose)
    assert len(out) == 2
    assert stats["groups_folded"] == 0


def _scope_table_row_atom(
    *, idx: int, cells: dict, sheet: str = "USD", artifact_id: str = "artB", filename: str = "rates.xlsx"
) -> EvidenceAtom:
    text = " | ".join(f"{k}: {v}" for k, v in cells.items())
    aid = f"atm_scope_{sheet}_{idx}"
    src = SourceRef(
        id=f"src_{aid}",
        artifact_id=artifact_id,
        artifact_type=ArtifactType.xlsx,
        filename=filename,
        locator={"sheet": sheet, "row": idx + 1},
        extraction_method="xlsx_scope",
        parser_version="xlsx_test",
    )
    return EvidenceAtom(
        id=aid,
        project_id="proj1",
        artifact_id=artifact_id,
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        value={"kind": "table_row", "sheet": sheet, "row": idx + 1, "cells": dict(cells)},
        source_refs=[src],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.5,
        review_status=ReviewStatus.needs_review,
        parser_version="xlsx_test",
    )


def test_scope_table_rows_money_fold_to_pricing_assumption() -> None:
    # The Chipotle shape: scope_item kind=table_row with dict cells + a price column.
    atoms = [
        _scope_table_row_atom(idx=i, cells={"Site Name": f"S{i}", "Unit Price": 100.0 + i})
        for i in range(50)
    ]
    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 1
    assert stats["commercial"] == 1
    summaries = [a for a in out if _type(a) == "pricing_assumption"]
    assert len(summaries) == 1
    assert summaries[0].value["line_count"] == 50
    assert len(summaries[0].value["rows"]) == 50
    assert len(out) == 1


def test_scope_table_rows_bulk_nonmoney_fold_keeps_type() -> None:
    # A giant non-money store list (Chipotle SSRS): folds only above bulk
    # threshold, and keeps its original scope_item type.
    atoms = [
        _scope_table_row_atom(idx=i, cells={"Store": "CHIPOTLE MEXICAN GRILL", "Num": str(13000000 + i)}, sheet="SSRS")
        for i in range(250)
    ]
    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 1
    assert stats["bulk"] == 1
    # Original type preserved; collapsed to one atom.
    assert len([a for a in out if _type(a) == "scope_item"]) == 1
    assert len(out) == 1
    s = out[0]
    assert s.value["_rolled_up"] is True
    assert len(s.value["rows"]) == 250


def test_medium_nonmoney_table_stays_granular() -> None:
    # 100 rows, no money, below bulk threshold (200): keep per-row signal.
    atoms = [
        _scope_table_row_atom(idx=i, cells={"Requirement": f"Req {i}", "Owner": "PM"}, sheet="Reqs")
        for i in range(100)
    ]
    out, stats = roll_up_table_rows(atoms)
    assert stats["groups_folded"] == 0
    assert len([a for a in out if _type(a) == "scope_item"]) == 100


def _site_atom(*, idx: int, artifact_id: str = "artS") -> EvidenceAtom:
    addr = f"{1000+idx} Main St, City{idx % 50}"
    aid = f"atm_site_{idx}"
    src = SourceRef(
        id=f"src_{aid}", artifact_id=artifact_id, artifact_type=ArtifactType.xlsx,
        filename="stores.xlsx", locator={"row": idx + 1},
        extraction_method="xlsx_site", parser_version="xlsx_test",
    )
    return EvidenceAtom(
        id=aid, project_id="proj1", artifact_id=artifact_id,
        atom_type=AtomType.physical_site,
        raw_text=f"facility: Store {idx} | address: {addr}",
        normalized_text=f"store {idx} {addr}".lower(),
        value={"kind": "site", "name": f"Store {idx}", "address": addr},
        source_refs=[src], authority_class=AuthorityClass.approved_site_roster,
        confidence=0.6, review_status=ReviewStatus.needs_review, parser_version="xlsx_test",
    )


def test_mega_site_list_training_cap_vs_production(monkeypatch) -> None:
    """The site cap is TRAINING-ONLY (decoupled from production output).

    PRODUCTION (SOWSMITH_SITE_ROLLUP_KEEP unset): every site is emitted as its
    own atom so the deliverable is complete. TRAINING (env set): the tail folds
    into one roster (all sites still preserved in value.rows) so a single deal's
    thousands of near-identical sites don't dominate the kNN store.
    """
    # PRODUCTION — env unset → no cap → all 500 sites emitted, no roster.
    monkeypatch.delenv("SOWSMITH_SITE_ROLLUP_KEEP", raising=False)
    out, stats = roll_up_table_rows([_site_atom(idx=i) for i in range(500)])
    sites = [a for a in out if _type(a) == "physical_site"]
    rosters = [a for a in sites if "site_roster_rollup" in (a.review_flags or [])]
    assert len(sites) == 500            # every site present in production
    assert len(rosters) == 0            # no fold in production
    assert stats["sites_folded"] == 0

    # TRAINING — env set to 150 → 150 individual + 1 roster holding the other 350.
    monkeypatch.setenv("SOWSMITH_SITE_ROLLUP_KEEP", "150")
    out, stats = roll_up_table_rows([_site_atom(idx=i) for i in range(500)])
    sites = [a for a in out if _type(a) == "physical_site"]
    rosters = [a for a in sites if "site_roster_rollup" in (a.review_flags or [])]
    individual = [a for a in sites if "site_roster_rollup" not in (a.review_flags or [])]
    assert len(individual) == 150
    assert len(rosters) == 1
    assert stats["sites_folded"] == 350
    assert len(rosters[0].value["rows"]) == 350  # every folded site preserved


def test_small_site_list_untouched() -> None:
    atoms = [_site_atom(idx=i) for i in range(40)]
    out, stats = roll_up_table_rows(atoms)
    assert stats["site_groups_folded"] == 0
    assert len([a for a in out if _type(a) == "physical_site"]) == 40


def test_fold_is_deterministic() -> None:
    cols = ["Part No", "Unit Price"]
    mk = lambda: [_row_atom(idx=i, columns=cols, row=[f"PN-{i}", 9.0 + i]) for i in range(50)]
    out1, _ = roll_up_table_rows(mk())
    out2, _ = roll_up_table_rows(mk())
    ids1 = [a.id for a in out1]
    ids2 = [a.id for a in out2]
    assert ids1 == ids2
    assert len(out1) == 1
