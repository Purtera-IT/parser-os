"""Gold-standard comparison (PRODUCTION_GAPS.md P3.3).

Compares a finished ``CompileResult`` JSON against the
``gold_standard.json`` files in ``real_data_cases/STRESS_*/labels/``.
Each metric in the gold file becomes a pass/fail verdict.

Supported metric keys (all optional; only the ones present in gold are
checked):

* ``expected_min_atom_count`` — atoms ≥ this value
* ``expected_min_packet_count`` — packets ≥ this value
* ``expected_min_quantity_atoms`` — atoms with `quantity:*` keys ≥ this
* ``expected_min_distinct_sites`` — distinct ``site:*`` entity_keys ≥ this
* ``expected_min_unique_vendors_referenced`` — distinct ``vendor:*`` keys ≥ this
* ``expected_min_constraint_atoms`` — atoms with type=constraint ≥ this
* ``expected_min_compliance_atoms`` — atoms whose entity_keys include
  any ``requirement:*`` ≥ this
* ``expected_quantity_conflict_edges_within_artifact`` — exact match
  (or ≥ when the value is suffixed ``+`` in source)
* ``expected_min_cross_artifact_edges`` — cross-artifact edges ≥ this
* ``expected_packet_families`` — set of packet families that must
  appear at least once
* ``expected_entity_keys_must_include`` — every key in the list must
  appear in at least one atom
* ``expected_legend_entries_min`` — total ``schematic_legend`` entries
  across all legend atoms ≥ this value
* ``expected_detection_targets_include`` — every target_key in the
  list must appear in at least one ``schematic_detection_target_set``
  atom's targets
* ``expected_symbol_counts`` — ``{target_key: int}`` mapping; each
  target must produce ≥ N detection atoms of that target_key
* ``expected_missing_legend_pages`` — list of page indices that
  must each have a ``missing_legend`` warning
* ``expected_unknown_symbol_count_max`` — at most N
  ``unknown_symbol`` warnings allowed
* ``expected_all_schematic_atoms_have_bbox`` — boolean; when true,
  every schematic atom's source_refs must carry a 4-element bbox
  with ``bbox_units=="pdf_points"``

The gold file may also embed nested ``per_artifact_gold_files`` references;
those are advisory and ignored by the comparator (only the top-level
metrics gate pass/fail).
"""

from __future__ import annotations

from typing import Any


def _atoms_with_entity_prefix(atoms: list[dict[str, Any]], prefix: str) -> int:
    return sum(
        1
        for atom in atoms
        if any(str(k).startswith(prefix) for k in (atom.get("entity_keys") or []))
    )


def _atoms_with_atom_type(atoms: list[dict[str, Any]], atype: str) -> int:
    return sum(
        1
        for atom in atoms
        if str(atom.get("atom_type") or "") == atype
    )


def _distinct_keys_with_prefix(atoms: list[dict[str, Any]], prefix: str) -> set[str]:
    out: set[str] = set()
    for atom in atoms:
        for k in atom.get("entity_keys") or []:
            if isinstance(k, str) and k.startswith(prefix):
                out.add(k)
    return out


def _packet_families(packets: list[dict[str, Any]]) -> set[str]:
    return {str(p.get("family") or "") for p in packets if p.get("family")}


def _cross_artifact_edge_count(edges: list[dict[str, Any]]) -> int:
    return sum(
        1
        for e in edges
        if isinstance(e.get("metadata"), dict) and bool(e["metadata"].get("cross_artifact"))
    )


def _quantity_conflict_count(edges: list[dict[str, Any]]) -> int:
    out = 0
    for e in edges:
        meta = e.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        family = str(meta.get("edge_family") or "")
        if family in {"part_number_quantity_conflict", "quantity_contradiction"}:
            out += 1
    return out


def _schematic_metrics(
    gold: dict[str, Any], atoms: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Schematic-specific gold assertions.

    All metrics in this group are skipped when the corresponding key
    is absent from ``gold``, so non-schematic cases are unaffected.
    """

    out: dict[str, dict[str, Any]] = {}

    legend_atoms = [a for a in atoms if a.get("atom_type") == "schematic_legend"]
    target_atoms = [a for a in atoms if a.get("atom_type") == "schematic_detection_target_set"]
    detection_atoms = [a for a in atoms if a.get("atom_type") == "schematic_symbol_detection"]
    warning_atoms = [a for a in atoms if a.get("atom_type") == "schematic_warning"]

    legend_entry_threshold = gold.get("expected_legend_entries_min")
    if legend_entry_threshold is not None:
        total_entries = sum(
            int(a.get("value", {}).get("entry_count") or 0) for a in legend_atoms
        )
        out["legend_entries_min"] = _check_min(total_entries, legend_entry_threshold)
    else:
        out["legend_entries_min"] = {"verdict": "skipped"}

    expected_targets = gold.get("expected_detection_targets_include") or []
    if expected_targets:
        present_targets: set[str] = set()
        for a in target_atoms:
            for t in a.get("value", {}).get("targets") or []:
                present_targets.add(str(t.get("target_key")))
        missing = [t for t in expected_targets if t not in present_targets]
        out["detection_targets_include"] = {
            "verdict": "pass" if not missing else "fail",
            "actual_present": sorted(present_targets),
            "expected_to_include": list(expected_targets),
            "missing": missing,
        }
    else:
        out["detection_targets_include"] = {"verdict": "skipped"}

    expected_counts = gold.get("expected_symbol_counts") or {}
    if expected_counts:
        actual_counts: dict[str, int] = {}
        for a in detection_atoms:
            key = str(a.get("value", {}).get("target_key") or "")
            if not key:
                continue
            actual_counts[key] = actual_counts.get(key, 0) + 1
        misses = {
            k: {"expected_min": v, "actual": actual_counts.get(k, 0)}
            for k, v in expected_counts.items()
            if actual_counts.get(k, 0) < int(v)
        }
        out["symbol_counts"] = {
            "verdict": "pass" if not misses else "fail",
            "actual": actual_counts,
            "expected_min": dict(expected_counts),
            "misses": misses,
        }
    else:
        out["symbol_counts"] = {"verdict": "skipped"}

    expected_missing_pages = gold.get("expected_missing_legend_pages")
    if expected_missing_pages is not None:
        flagged_pages: set[int] = set()
        for a in warning_atoms:
            value = a.get("value", {})
            if str(value.get("warning_type")) == "missing_legend":
                try:
                    flagged_pages.add(int(value.get("page")))
                except (TypeError, ValueError):
                    continue
        missing = [int(p) for p in expected_missing_pages if int(p) not in flagged_pages]
        out["missing_legend_pages"] = {
            "verdict": "pass" if not missing else "fail",
            "actual_flagged": sorted(flagged_pages),
            "expected_to_flag": [int(p) for p in expected_missing_pages],
            "missing": missing,
        }
    else:
        out["missing_legend_pages"] = {"verdict": "skipped"}

    unknown_max = gold.get("expected_unknown_symbol_count_max")
    if unknown_max is not None:
        unknown_count = sum(
            1
            for a in warning_atoms
            if str(a.get("value", {}).get("warning_type")) == "unknown_symbol"
        )
        out["unknown_symbol_max"] = {
            "verdict": "pass" if unknown_count <= int(unknown_max) else "fail",
            "actual": unknown_count,
            "expected_max": int(unknown_max),
        }
    else:
        out["unknown_symbol_max"] = {"verdict": "skipped"}

    if gold.get("expected_all_schematic_atoms_have_bbox") is True:
        all_schematic = (
            legend_atoms + target_atoms + detection_atoms + warning_atoms
        )
        missing_bbox: list[str] = []
        for a in all_schematic:
            for src in a.get("source_refs") or []:
                loc = src.get("locator") if isinstance(src, dict) else {}
                if not isinstance(loc, dict):
                    continue
                if str(a.get("atom_type")) == "schematic_symbol_detection":
                    bbox = loc.get("bbox")
                    if not (
                        isinstance(bbox, (list, tuple))
                        and len(bbox) == 4
                        and loc.get("bbox_units") == "pdf_points"
                    ):
                        missing_bbox.append(str(a.get("id")))
                # For non-detection schematic atoms, the bbox is optional
                # (legend / target_set may not have one) — we only enforce
                # it on detections via this gate.
        out["all_schematic_atoms_have_bbox"] = {
            "verdict": "pass" if not missing_bbox else "fail",
            "missing_bbox_atom_ids": missing_bbox[:8],
            "missing_count": len(missing_bbox),
        }
    else:
        out["all_schematic_atoms_have_bbox"] = {"verdict": "skipped"}

    return out


def _check_min(actual: int, threshold: Any) -> dict[str, Any]:
    """Return verdict for a "value ≥ threshold" check.

    ``threshold`` may be an int, a string like "60+", or any
    representation that accepts ``int(...)`` after stripping a
    trailing ``+``.
    """
    if threshold is None:
        return {"verdict": "skipped", "reason": "threshold absent"}
    raw = str(threshold).strip()
    if raw.endswith("+"):
        raw = raw[:-1]
    try:
        target = int(float(raw))
    except (TypeError, ValueError):
        return {"verdict": "skipped", "reason": f"unparseable threshold {threshold!r}"}
    return {
        "verdict": "pass" if actual >= target else "fail",
        "actual": actual,
        "expected_min": target,
    }


def compare_to_gold(*, gold: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    """Run the gold-vs-compiled comparison.

    Returns ``{"case_id", "metrics": {name: {verdict, actual, expected_min}}, "overall": {pass_fraction, …}}``.
    """
    atoms = list(compiled.get("atoms") or [])
    edges = list(compiled.get("edges") or [])
    packets = list(compiled.get("packets") or [])

    metrics: dict[str, dict[str, Any]] = {}

    metrics["atom_count"] = _check_min(
        len(atoms), gold.get("expected_min_atom_count")
    )
    metrics["packet_count"] = _check_min(
        len(packets), gold.get("expected_min_packet_count")
    )
    metrics["quantity_atoms"] = _check_min(
        _atoms_with_entity_prefix(atoms, "quantity:"),
        gold.get("expected_min_quantity_atoms"),
    )
    metrics["distinct_sites"] = _check_min(
        len(_distinct_keys_with_prefix(atoms, "site:")),
        gold.get("expected_min_distinct_sites"),
    )
    metrics["unique_vendors"] = _check_min(
        len(_distinct_keys_with_prefix(atoms, "vendor:")),
        gold.get("expected_min_unique_vendors_referenced"),
    )
    metrics["unique_part_numbers"] = _check_min(
        len(_distinct_keys_with_prefix(atoms, "part_number:")),
        gold.get("expected_min_part_number_atoms"),
    )
    metrics["constraint_atoms"] = _check_min(
        _atoms_with_atom_type(atoms, "constraint"),
        gold.get("expected_min_constraint_atoms"),
    )
    metrics["compliance_atoms"] = _check_min(
        _atoms_with_entity_prefix(atoms, "requirement:"),
        gold.get("expected_min_compliance_atoms"),
    )
    metrics["quantity_conflict_edges"] = _check_min(
        _quantity_conflict_count(edges),
        gold.get("expected_quantity_conflict_edges_within_artifact")
        or gold.get("expected_min_quantity_conflict_edges"),
    )
    metrics["cross_artifact_edges"] = _check_min(
        _cross_artifact_edge_count(edges),
        gold.get("expected_min_cross_artifact_edges"),
    )

    expected_families = gold.get("expected_packet_families") or []
    if expected_families:
        present = _packet_families(packets)
        missing = [f for f in expected_families if f not in present]
        metrics["packet_families"] = {
            "verdict": "pass" if not missing else "fail",
            "actual_present": sorted(present),
            "expected_to_include": list(expected_families),
            "missing": missing,
        }
    else:
        metrics["packet_families"] = {"verdict": "skipped"}

    metrics.update(_schematic_metrics(gold, atoms))

    expected_keys = gold.get("expected_entity_keys_must_include") or []
    if expected_keys:
        all_keys: set[str] = set()
        for atom in atoms:
            for k in atom.get("entity_keys") or []:
                if isinstance(k, str):
                    all_keys.add(k)
        missing_keys = [k for k in expected_keys if k not in all_keys]
        metrics["entity_keys_must_include"] = {
            "verdict": "pass" if not missing_keys else "fail",
            "expected_count": len(expected_keys),
            "missing_count": len(missing_keys),
            "missing_sample": missing_keys[:8],
        }
    else:
        metrics["entity_keys_must_include"] = {"verdict": "skipped"}

    # Overall pass fraction across non-skipped metrics
    non_skipped = [m for m in metrics.values() if m.get("verdict") != "skipped"]
    if non_skipped:
        passes = sum(1 for m in non_skipped if m.get("verdict") == "pass")
        pass_fraction = passes / len(non_skipped)
    else:
        passes = 0
        pass_fraction = 0.0

    return {
        "case_id": gold.get("case_id") or compiled.get("project_id") or "unknown",
        "metrics": metrics,
        "overall": {
            "pass": passes,
            "fail": sum(1 for m in non_skipped if m.get("verdict") == "fail"),
            "skipped": sum(1 for m in metrics.values() if m.get("verdict") == "skipped"),
            "total_checked": len(non_skipped),
            "pass_fraction": round(pass_fraction, 4),
        },
    }


__all__ = ["compare_to_gold"]
