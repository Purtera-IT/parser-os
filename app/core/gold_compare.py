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
