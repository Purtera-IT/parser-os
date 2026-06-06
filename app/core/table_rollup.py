"""Pre-enrich table rollup — universal backstop for high-cardinality
spreadsheet tables that explode into one atom per row.

Why this exists
---------------
When the xlsx parser cannot confidently route a sheet to a commercial /
financial rollup, it falls back to emitting **one atom per data row**:

* ``raw_table_row`` atoms (``value._columns`` + ``value._row``), or
* ``scope_item`` atoms tagged ``value.kind == "table_row"`` whose
  ``value.cells`` is a ``{header: value}`` dict (the SCOPE-routed path).

A 9 MB store list / rate-card workbook then lands in the pipeline as tens of
thousands of per-row atoms.  Every one is dragged through the per-atom LLM
``enrich_entities`` pass (hours of wall-time) and the
``typed_atom_classification`` pass, and floods the training store with tens
of thousands of near-identical, low-diversity rows that bias the kNN /
neural head toward catalog noise.

This stage runs **before** ``enrich_entities`` and folds each
high-cardinality, homogeneous group of table-row atoms losslessly into a
single summary atom whose ``value.rows`` preserves every row.  Money-bearing
tables fold to ``pricing_assumption`` (the shape ``build_bill_of_materials``
and the ``commercial_summary`` packet already consume); other bulk tables
keep their original type but collapse to one rolled summary.  Output is
preserved (full drill-down in ``value.rows``) while the atom count — and the
LLM cost and the training-row flood — collapse to one atom per table.

Universality contract
----------------------
* No customer names, no sheet-name keyword lists, no per-deal tuning.
* Signals are purely structural: how many rows share a sheet + column
  schema, and whether the table carries money columns (reusing the parser's
  own ``_money_columns`` / ``_row_money_values`` detectors).
* Two thresholds protect real per-row signal:
    - commercial tables fold at >= ``COMMERCIAL_MIN`` rows (they are
      summarised downstream anyway), and
    - non-commercial tables fold only at >= ``BULK_MIN`` rows, far beyond
      any hand-authored requirements / roster matrix, so medium tables stay
      granular and keep their per-row extraction value.
* ``physical_site`` and other structured atoms are never folded — only
  ``raw_table_row`` atoms and ``kind == "table_row"`` rows qualify.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

# Commercial (money-bearing) tables fold at this row count: they are rolled
# into the commercial summary downstream regardless, so per-row atoms buy
# nothing but cost.
_DEFAULT_COMMERCIAL_MIN = 40

# Non-commercial tables fold only when this much larger: below it, a table
# may be a hand-authored requirements / roster / asset matrix whose rows
# carry real per-row extraction signal, so we leave it granular.
_DEFAULT_BULK_MIN = 200

# Hard cap on rows materialised into ``value.rows`` (mirrors the parser's
# ``_COMMERCIAL_FOLD_CAP``). Drill-down stays bounded; ``line_count`` still
# reports the true total.
_DEFAULT_FOLD_CAP = 5000

# Site capping: a single artifact that yields a huge number of near-identical
# ``physical_site`` atoms (e.g. a 4,000-store address list) floods every site
# stage (enrich, site-verify, geo-fallback) one site at a time AND swamps the
# training store with low-diversity rows. We keep a representative sample as
# individual sites and fold the rest into one roster summary (all sites still
# preserved in ``value.rows``). Gated high so normal deals are untouched.
_DEFAULT_SITE_KEEP = 150   # representative individual sites kept per artifact
_DEFAULT_SITE_MIN = 300    # only fold when an artifact yields at least this many


def _env_int(name: str, default: int) -> int:
    try:
        return max(2, int(os.environ.get(name, str(default))))
    except Exception:
        return default


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _table_row_view(atom: Any) -> tuple[str, tuple, list[str], list[Any]] | None:
    """If ``atom`` is a per-row table atom, return
    ``(sheet, column_signature, headers, cell_values)``; else ``None``.

    Handles both emitted shapes:
      * ``raw_table_row``  → ``value._columns`` (list) + ``value._row`` (list)
      * ``kind=="table_row"`` → ``value.cells`` ({header: value} dict)
    """
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return None

    # Shape A: raw_table_row
    if _atom_type_str(atom) == "raw_table_row" or ("_columns" in val and "_row" in val):
        columns = val.get("_columns") or []
        row = val.get("_row")
        if not columns or row is None:
            return None
        headers = [str(c) for c in columns]
        sheet = str(val.get("_sheet") or "")
        return sheet, tuple(headers), headers, list(row)

    # Shape B: SCOPE-routed table rows
    if val.get("kind") == "table_row":
        cells = val.get("cells")
        if not isinstance(cells, dict) or not cells:
            return None
        headers = [str(k) for k in cells.keys()]
        sheet = str(val.get("sheet") or "")
        values = [cells[k] for k in cells.keys()]
        return sheet, tuple(headers), headers, values

    return None


def roll_up_table_rows(atoms: list[Any]) -> tuple[list[Any], dict[str, int]]:
    """Fold high-cardinality, homogeneous table-row groups into one summary
    atom each.

    Returns ``(new_atom_list, stats)``.  ``new_atom_list`` preserves the
    original ordering of every atom that is *not* folded and substitutes each
    folded group with its single summary atom at the position of the group's
    first row.  The caller diffs old-vs-new to populate the suppression
    ledger, exactly as the other drop/fold stages do.

    Pure / deterministic: no LLM, no network, no randomness.
    """
    stats = {"groups_folded": 0, "rows_folded": 0, "summary_atoms": 0,
             "groups_seen": 0, "commercial": 0, "bulk": 0,
             "site_groups_folded": 0, "sites_folded": 0, "sites_kept": 0}
    if not atoms:
        return list(atoms), stats

    from app.core.schemas import (
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )
    from app.core.ids import stable_id
    from app.parsers.xlsx_parser import (
        _money_columns,
        _row_money_values,
        _MIN_MONEY_VALUE,
    )

    commercial_min = _env_int("SOWSMITH_TABLE_ROLLUP_MIN_ROWS", _DEFAULT_COMMERCIAL_MIN)
    bulk_min = _env_int("SOWSMITH_TABLE_ROLLUP_BULK_ROWS", _DEFAULT_BULK_MIN)
    fold_cap = _env_int("SOWSMITH_TABLE_ROLLUP_FOLD_CAP", _DEFAULT_FOLD_CAP)

    # 1. Bucket per-row table atoms by (artifact, sheet, column-signature),
    #    preserving first-seen order. Non-table atoms are left out of the map.
    groups: "OrderedDict[tuple, list[Any]]" = OrderedDict()
    views: dict[int, tuple] = {}  # id(atom) -> view
    for atom in atoms:
        view = _table_row_view(atom)
        if view is None:
            continue
        sheet, colsig, _headers, _values = view
        key = (getattr(atom, "artifact_id", "") or "", sheet, colsig)
        groups.setdefault(key, []).append(atom)
        views[id(atom)] = view

    stats["groups_seen"] = len(groups)

    fold_summary: dict[tuple, Any] = {}
    folded_ids: set[int] = set()

    for key, members in groups.items():
        if len(members) < commercial_min:
            continue

        headers = list(key[2])
        sheet_name = key[1]
        # Reconstruct the cell matrix for the parser's money detectors.
        row_values = [views[id(a)][3] for a in members]
        money_cols = _money_columns([headers, *row_values])
        is_commercial = bool(money_cols)

        # Fold decision: commercial tables fold at commercial_min; other
        # tables only at the much larger bulk_min (protects real per-row
        # content from being collapsed).
        if not is_commercial and len(members) < bulk_min:
            continue

        folded_rows: list[dict[str, Any]] = []
        all_values: list[float] = []
        for a in members:
            cells_raw = views[id(a)][3]
            cells = [("" if c is None else str(c).strip()) for c in cells_raw]
            values = _row_money_values(cells_raw, money_cols) if money_cols else []
            all_values.extend(values)
            money_keys = sorted({f"money:{int(round(v))}" for v in values})
            label = " ".join(
                c for c in cells
                if c and not c.replace(",", "").replace(".", "").lstrip("-").isdigit()
            ).strip()[:300]
            v = getattr(a, "value", None) or {}
            if len(folded_rows) < fold_cap:
                folded_rows.append({
                    "row": v.get("row") if v.get("row") is not None else (int(v.get("_row_idx") or 0) + 1),
                    "label": label,
                    "money_keys": money_keys,
                    "cells": [c for c in cells if c],
                    "headers": headers,
                })

        summary = _make_summary_atom(
            members[0],
            sheet_name=sheet_name,
            line_count=len(members),
            values=all_values,
            folded_rows=folded_rows,
            is_commercial=is_commercial,
            original_type=_atom_type_str(members[0]),
            min_money_value=_MIN_MONEY_VALUE,
            EvidenceAtom=EvidenceAtom,
            AtomType=AtomType,
            AuthorityClass=AuthorityClass,
            ReviewStatus=ReviewStatus,
            SourceRef=SourceRef,
            stable_id=stable_id,
        )
        fold_summary[key] = summary
        for a in members:
            folded_ids.add(id(a))
        stats["groups_folded"] += 1
        stats["rows_folded"] += len(members)
        stats["summary_atoms"] += 1
        stats["commercial" if is_commercial else "bulk"] += 1

    # 2. Rebuild after the table-row fold (drop folded members; splice each
    #    group's summary in at the position of its first member).
    if fold_summary:
        id_to_key: dict[int, tuple] = {}
        for key in fold_summary:
            for a in groups[key]:
                id_to_key[id(a)] = key
        emitted: set[tuple] = set()
        out: list[Any] = []
        for atom in atoms:
            if id(atom) in folded_ids:
                key = id_to_key.get(id(atom))
                if key is not None and key not in emitted:
                    out.append(fold_summary[key])
                    emitted.add(key)
                continue
            out.append(atom)
    else:
        out = list(atoms)

    # 3. Site cap — TRAINING-ONLY (decoupled from production output).
    #    A huge near-identical site list (e.g. a 2,443-store address sheet) would
    #    dominate the kNN training store with low-diversity rows, so the TRAINING
    #    driver sets SOWSMITH_SITE_ROLLUP_KEEP to fold the tail into one roster
    #    (every site still preserved in value.rows). In PRODUCTION the env var is
    #    UNSET → no cap → EVERY site is emitted as its own atom so the deliverable
    #    (Deal Kit / site list) is complete. The money-table fold above always runs
    #    (it is an enrich-cost necessity); only the site cap is gated here.
    _site_keep_env = os.environ.get("SOWSMITH_SITE_ROLLUP_KEEP")
    if _site_keep_env is not None:
        site_keep = _env_int("SOWSMITH_SITE_ROLLUP_KEEP", _DEFAULT_SITE_KEEP)
        site_min = _env_int("SOWSMITH_SITE_ROLLUP_MIN", _DEFAULT_SITE_MIN)
        site_groups: "OrderedDict[str, list[Any]]" = OrderedDict()
        for a in out:
            if _atom_type_str(a) == "physical_site":
                site_groups.setdefault(getattr(a, "artifact_id", "") or "", []).append(a)

        drop_ids: set[int] = set()
        roster_at_first: dict[int, Any] = {}
        for art, members in site_groups.items():
            if len(members) < site_min:
                continue
            keep, tail = members[:site_keep], members[site_keep:]
            if not tail:
                continue
            roster = _make_site_roster_atom(
                tail, total=len(members), kept=len(keep),
                EvidenceAtom=EvidenceAtom, AtomType=AtomType,
                AuthorityClass=AuthorityClass, ReviewStatus=ReviewStatus,
                SourceRef=SourceRef, stable_id=stable_id,
            )
            for a in tail:
                drop_ids.add(id(a))
            roster_at_first[id(tail[0])] = roster
            stats["site_groups_folded"] += 1
            stats["sites_folded"] += len(tail)
            stats["sites_kept"] += len(keep)

        if drop_ids:
            capped: list[Any] = []
            for a in out:
                if id(a) in drop_ids:
                    r = roster_at_first.get(id(a))
                    if r is not None:
                        capped.append(r)  # splice roster at the first tail position
                    continue
                capped.append(a)
            out = capped

    return out, stats


def _make_summary_atom(
    template: Any,
    *,
    sheet_name: str,
    line_count: int,
    values: list[float],
    folded_rows: list[dict[str, Any]],
    is_commercial: bool,
    original_type: str,
    min_money_value: float,
    EvidenceAtom: Any,
    AtomType: Any,
    AuthorityClass: Any,
    ReviewStatus: Any,
    SourceRef: Any,
    stable_id: Any,
) -> Any:
    """Build the single rollup atom for a folded group.

    Money-bearing tables become a ``pricing_assumption`` summary mirroring
    ``XlsxParser._commercial_summary_atom`` (so the commercial / BOM views
    render unchanged).  Other bulk tables keep their original atom type but
    collapse to one rolled summary carrying ``value.rows`` for drill-down.
    """
    artifact_id = getattr(template, "artifact_id", "") or ""
    project_id = getattr(template, "project_id", "") or ""

    src_refs = getattr(template, "source_refs", None) or []
    tmpl_src = src_refs[0] if src_refs else None
    artifact_type = getattr(tmpl_src, "artifact_type", None)
    filename = getattr(tmpl_src, "filename", "") or ""

    if is_commercial:
        lo = min(values) if values else 0.0
        hi = max(values) if values else 0.0
        total = sum(values)
        money_keys = sorted(
            {f"money:{int(round(v))}" for v in (lo, hi, total) if v >= min_money_value}
        )
        # ASCII hyphen (en-dash mojibanged through non-UTF-8 consumers before).
        label = (
            f"{sheet_name}: {line_count} pricing line"
            f"{'s' if line_count != 1 else ''}, "
            f"${int(round(lo)):,}-${int(round(hi)):,}"
        )
        atom_type = AtomType.pricing_assumption
        review_flags = ["pricing_rollup", "table_rollup_backstop"]
        value_extra = {
            "is_summary": True,
            "money_min": round(lo, 2),
            "money_max": round(hi, 2),
            "money_sum": round(total, 2),
            "money_keys": money_keys,
            "sheet_role": "",  # escaped the classifier; let content drive BOM
        }
        entity_keys = money_keys
    else:
        label = f"{sheet_name}: {line_count} table rows (rolled up)"
        # Keep the original type so downstream typing/packeting is unchanged;
        # the giant table just becomes one atom instead of thousands.
        atom_type = getattr(AtomType, original_type, AtomType.scope_item)
        review_flags = ["table_rollup_backstop"]
        value_extra = {"is_summary": True}
        entity_keys = []

    atom_id = stable_id("atm", artifact_id, atom_type.value, sheet_name, "rollup_backstop")
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        locator={"sheet": sheet_name, "extraction": "table_rollup_backstop", "rollup": True},
        extraction_method="table_rollup_backstop",
        parser_version="table_rollup_v1",
    )
    value = {
        "label": label,
        "line_count": line_count,
        "sheet_name": sheet_name,
        "rows": folded_rows if folded_rows else None,
        "_rolled_up": True,
        "_source": "table_rollup_backstop",
        **value_extra,
    }
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=label,
        normalized_text=label.lower(),
        value=value,
        entity_keys=entity_keys,
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.7,
        confidence_raw=0.7,
        calibrated_confidence=0.7,
        review_status=ReviewStatus.needs_review,
        review_flags=review_flags,
        parser_version="table_rollup_v1",
    )


def _make_site_roster_atom(
    tail: list[Any],
    *,
    total: int,
    kept: int,
    EvidenceAtom: Any,
    AtomType: Any,
    AuthorityClass: Any,
    ReviewStatus: Any,
    SourceRef: Any,
    stable_id: Any,
) -> Any:
    """Fold the tail of a huge per-artifact site list into one roster summary.

    Keeps the type ``physical_site`` so downstream site handling treats it as a
    single roster entry; every folded site is preserved in ``value.rows`` for
    drill-down. ``total`` is the full site count; ``kept`` the number left as
    individual atoms.
    """
    template = tail[0]
    artifact_id = getattr(template, "artifact_id", "") or ""
    project_id = getattr(template, "project_id", "") or ""
    src_refs = getattr(template, "source_refs", None) or []
    tmpl_src = src_refs[0] if src_refs else None
    artifact_type = getattr(tmpl_src, "artifact_type", None)
    filename = getattr(tmpl_src, "filename", "") or ""

    rows: list[dict[str, Any]] = []
    for a in tail:
        v = getattr(a, "value", None)
        rows.append(v if isinstance(v, dict) else {"text": getattr(a, "raw_text", "")})

    label = f"Site roster: {len(tail):,} additional sites (rolled up; {kept} shown individually, {total:,} total)"
    atom_id = stable_id("atm", artifact_id, "physical_site", "site_roster", "rollup_backstop")
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        locator={"extraction": "table_rollup_backstop", "rollup": True, "roster": True},
        extraction_method="table_rollup_backstop",
        parser_version="table_rollup_v1",
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.physical_site,
        raw_text=label,
        normalized_text=label.lower(),
        value={
            "is_summary": True,
            "label": label,
            "line_count": len(tail),
            "rows": rows,
            "_rolled_up": True,
            "_source": "table_rollup_backstop_site_roster",
        },
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.approved_site_roster,
        confidence=0.7,
        confidence_raw=0.7,
        calibrated_confidence=0.7,
        review_status=ReviewStatus.needs_review,
        review_flags=["site_roster_rollup", "table_rollup_backstop"],
        parser_version="table_rollup_v1",
    )


__all__ = ["roll_up_table_rows"]
