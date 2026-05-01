from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class GoldExpectedPacket(BaseModel):
    family: str
    anchor_key_contains: str
    must_contain_quantities: list[float] = Field(default_factory=list)
    expected_status: str | None = None
    forbidden_governing_authority: list[str] = Field(default_factory=list)


class GoldExpectedGoverning(BaseModel):
    family: str
    anchor_key_contains: str
    governing_authority: str


class GoldForbiddenCondition(BaseModel):
    condition: str


class GoldScenario(BaseModel):
    scenario_id: str
    project_dir: str | None = None
    expected_packets: list[GoldExpectedPacket] = Field(default_factory=list)
    expected_governing: list[GoldExpectedGoverning] = Field(default_factory=list)
    forbidden: list[GoldForbiddenCondition] = Field(default_factory=list)


def load_gold(path: Path) -> GoldScenario:
    return GoldScenario.model_validate(json.loads(path.read_text(encoding="utf-8")))


def copper_001_material_gold_checks(compile_payload: dict) -> dict[str, object]:
    """
    PASS/FAIL checks for COPPER_001-style roster aggregate vs vendor primary totals.
    Works on compile_result.json-shaped dicts (atoms, packets, edges, entities).
    """
    atoms: dict[str, dict] = {a["id"]: a for a in compile_payload.get("atoms") or [] if isinstance(a, dict) and a.get("id")}
    packets: list[dict] = [p for p in compile_payload.get("packets") or [] if isinstance(p, dict)]
    edges: list[dict] = [e for e in compile_payload.get("edges") or [] if isinstance(e, dict)]
    entities: list[dict] = [e for e in compile_payload.get("entities") or [] if isinstance(e, dict)]

    def _text_blob(atom_id: str) -> str:
        a = atoms.get(atom_id) or {}
        raw = str(a.get("raw_text", ""))
        val = a.get("value") if isinstance(a.get("value"), dict) else {}
        return f"{raw} {val}".lower()

    def _material_packet(identity: str, family: str) -> dict | None:
        for p in packets:
            if p.get("family") != family:
                continue
            if p.get("anchor_key") == f"material:{identity}":
                return p
        return None

    def _qty_pair_from_packet(p: dict | None) -> tuple[bool, str]:
        if not p:
            return False, "missing_packet"
        cert = p.get("certificate") or {}
        blob = f"{p.get('reason', '')} {cert.get('existence_reason', '')} {cert.get('contradiction_summary', '')}"
        return True, blob

    checks: dict[str, object] = {}

    p_rj = _material_packet("rj45", "quantity_conflict")
    ok_rj, blob_rj = _qty_pair_from_packet(p_rj)
    checks["quantity_conflict_rj45_72_68"] = ok_rj and "72" in blob_rj and "68" in blob_rj

    p_utp = _material_packet("cat6_utp", "vendor_mismatch")
    ok_u, blob_u = _qty_pair_from_packet(p_utp)
    checks["vendor_mismatch_cat6_utp_66_60"] = ok_u and "66" in blob_u and "60" in blob_u

    p_stp = _material_packet("cat6_stp", "vendor_mismatch")
    ok_s, blob_s = _qty_pair_from_packet(p_stp)
    checks["vendor_mismatch_cat6_stp_6_8"] = ok_s and "6" in blob_s and "8" in blob_s

    bad_vendor_governs = False
    for p in packets:
        for gid in p.get("governing_atom_ids") or []:
            a = atoms.get(str(gid))
            if not a:
                continue
            if a.get("authority_class") == "vendor_quote" and a.get("atom_type") in {
                "scope_item",
                "scope_line",
                "scope_line_item",
            }:
                bad_vendor_governs = True
    checks["forbidden_vendor_quote_governs_scope"] = not bad_vendor_governs

    bad_rfp = False
    for p in packets:
        for gid in p.get("governing_atom_ids") or []:
            a = atoms.get(str(gid))
            if not a:
                continue
            ac = str(a.get("authority_class", ""))
            if ac in {"original_rfp", "rfp_original", "issued_rfp"}:
                bad_rfp = True
    checks["forbidden_original_rfp_governs_over_addendum"] = not bad_rfp

    bad_total_entity = any(
        "total" == str(e.get("canonical_key", "")).strip().lower()
        or str(e.get("canonical_name", "")).strip().lower() in {"total", "totals", "grand total"}
        for e in entities
    )
    checks["forbidden_total_row_becomes_entity"] = not bad_total_entity

    bad_power = False
    for p in packets:
        if p.get("family") != "scope_inclusion":
            continue
        for aid in (p.get("governing_atom_ids") or []) + (p.get("supporting_atom_ids") or []):
            blob = _text_blob(str(aid))
            if re.search(r"\bpower\b", blob) and re.search(r"\b(cat6|rj45|structured\s*cabling)\b", blob):
                bad_power = True
                break
        if bad_power:
            break
    checks["forbidden_power_as_in_scope_structured_cabling"] = not bad_power

    mat_edges = [
        e
        for e in edges
        if e.get("edge_type") == "contradicts"
        and (e.get("metadata") or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    checks["material_contradiction_edge_count"] = len(mat_edges)

    checks["overall_pass"] = all(v is True for k, v in checks.items() if k != "material_contradiction_edge_count")
    return checks
