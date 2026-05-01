"""
Gold comparator: labels/gold_packets.json vs outputs/compile_result.json for any real-data case.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GoldExpectedPacketRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    index: int | None = None
    family: str
    anchor_key_contains: str = ""
    expected_status: str | None = None
    must_contain_quantities: list[float] = Field(default_factory=list)
    must_contain_text: list[str] = Field(default_factory=list)
    # OR: at least one substring must appear in the evidence blob.
    must_contain_text_any_of: list[str] = Field(default_factory=list)
    # If set, primary governing atom (first governing_atom_ids) must have this authority_class.
    expected_governing_authority: str | None = None
    # If set, any governing atom must have one of these authority classes (OR).
    expected_governing_authority_any_of: list[str] = Field(default_factory=list)
    # If false, row is still reported but does not affect overall_pass (e.g. aspirational packet not yet emitted).
    required: bool = True


class GoldPacketsFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    scenario_id: str | None = None
    expected_packets: list[GoldExpectedPacketRow] = Field(default_factory=list)
    forbidden_conditions: list[str] = Field(default_factory=list)


def load_gold_packets(path: Path) -> GoldPacketsFile:
    return GoldPacketsFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _atom_blob(atom: dict[str, Any]) -> str:
    raw = str(atom.get("raw_text", "") or "")
    val = atom.get("value")
    blob = raw
    if isinstance(val, dict):
        blob += " " + json.dumps(val, sort_keys=True)
    return _norm(blob)


def _quantity_in_blob(q: float, blob: str) -> bool:
    blob_n = _norm(blob)
    if q == int(q):
        if str(int(q)) in blob_n.replace(".0", ""):
            return True
    s = str(q).lower().rstrip("0").rstrip(".")
    if s in blob_n:
        return True
    alt = f"{q:.4f}".rstrip("0").rstrip(".").lower()
    return alt in blob_n


def _anchor_haystack(packet: dict[str, Any]) -> str:
    parts = [
        str(packet.get("anchor_key", "") or ""),
        str(packet.get("topic", "") or ""),
    ]
    sig = packet.get("anchor_signature")
    if isinstance(sig, dict):
        parts.append(str(sig.get("canonical_key", "") or ""))
        parts.append(str(sig.get("normalized_topic", "") or ""))
    return _norm(" ".join(parts))


def _certificate_blob(cert: dict[str, Any] | None) -> str:
    if not isinstance(cert, dict):
        return ""
    parts = [
        str(cert.get("existence_reason", "") or ""),
        str(cert.get("governing_rationale", "") or ""),
        str(cert.get("contradiction_summary", "") or ""),
        json.dumps(cert.get("authority_path", []) or [], sort_keys=True),
    ]
    return _norm(" ".join(parts))


def _packet_linked_atom_ids(packet: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("governing_atom_ids", "supporting_atom_ids", "contradicting_atom_ids"):
        for x in packet.get(key) or []:
            if x:
                ids.append(str(x))
    return ids


def _collect_evidence_blob(packet: dict[str, Any], atoms: dict[str, dict[str, Any]]) -> str:
    parts = [
        _norm(str(packet.get("reason", "") or "")),
        _certificate_blob(packet.get("certificate")),
    ]
    for aid in _packet_linked_atom_ids(packet):
        a = atoms.get(aid)
        if a:
            parts.append(_atom_blob(a))
    cert = packet.get("certificate")
    if isinstance(cert, dict):
        for mid in cert.get("minimal_sufficient_atom_ids") or []:
            a = atoms.get(str(mid))
            if a:
                parts.append(_atom_blob(a))
    return " ".join(parts)


def _primary_governing_authority(packet: dict[str, Any], atoms: dict[str, dict[str, Any]]) -> str | None:
    gids = packet.get("governing_atom_ids") or []
    if not gids:
        return None
    a = atoms.get(str(gids[0]))
    if not a:
        return None
    return str(a.get("authority_class") or "")


def _match_expected_row(
    row: GoldExpectedPacketRow,
    packets: list[dict[str, Any]],
    atoms: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None, str | None, dict[str, Any]]:
    needle = _norm(row.anchor_key_contains) if row.anchor_key_contains else ""
    candidates = [p for p in packets if str(p.get("family", "")) == row.family]
    anchor_filtered: list[dict[str, Any]] = []
    for p in candidates:
        hay = _anchor_haystack(p)
        if needle and needle not in hay:
            continue
        anchor_filtered.append(p)

    if not anchor_filtered:
        return False, None, "no_packet_matching_family_and_anchor", {"candidates_count": len(candidates)}

    last_reason: str | None = "no_candidate_passed_post_filters"
    last_evidence: dict[str, Any] = {}
    for matched in anchor_filtered:
        if row.expected_status and str(matched.get("status", "")) != row.expected_status:
            last_reason = f"status_want_{row.expected_status}_got_{matched.get('status')}"
            continue

        blob = _collect_evidence_blob(matched, atoms)
        qty_ok = True
        for q in row.must_contain_quantities:
            if not _quantity_in_blob(q, blob):
                qty_ok = False
                last_reason = f"missing_quantity_{q}"
                last_evidence = {"blob_excerpt": blob[:400]}
                break
        if not qty_ok:
            continue

        text_ok = True
        for t in row.must_contain_text:
            if _norm(t) not in blob:
                text_ok = False
                last_reason = f"missing_text:{t!r}"
                last_evidence = {"blob_excerpt": blob[:400]}
                break
        if not text_ok:
            continue

        if row.must_contain_text_any_of:
            if not any(_norm(t) in blob for t in row.must_contain_text_any_of):
                last_reason = f"missing_any_of_text:{row.must_contain_text_any_of!r}"
                last_evidence = {"blob_excerpt": blob[:400]}
                continue

        if row.expected_governing_authority:
            got = _primary_governing_authority(matched, atoms)
            if got != row.expected_governing_authority:
                last_reason = f"governing_authority_want_{row.expected_governing_authority}_got_{got}"
                continue

        if row.expected_governing_authority_any_of:
            gov_ids = matched.get("governing_atom_ids") or []
            got_classes = {str(atoms.get(gid, {}).get("authority_class")) for gid in gov_ids if gid in atoms}
            want = set(row.expected_governing_authority_any_of)
            if not (got_classes & want):
                last_reason = f"governing_authority_none_of_{want}_got_{got_classes}"
                continue

        return True, matched, None, {"blob_excerpt": blob[:500]}

    return False, anchor_filtered[0], last_reason, last_evidence


def _forbidden_vendor_quote_governs_scope(packets: list[dict], atoms: dict[str, dict]) -> tuple[bool, str]:
    """Vendor quote must not govern active scope inclusion (vendor may govern pollution scope_exclusion)."""
    for p in packets:
        fam = str(p.get("family", ""))
        if fam != "scope_inclusion":
            continue
        for gid in p.get("governing_atom_ids") or []:
            a = atoms.get(str(gid))
            if not a:
                continue
            if a.get("authority_class") == "vendor_quote":
                return False, f"packet {p.get('id')} scope_inclusion governed by vendor_quote atom {gid}"
    return True, ""


def _forbidden_original_rfp_over_addendum(packets: list[dict], atoms: dict[str, dict]) -> tuple[bool, str]:
    bad_classes = {"original_rfp", "rfp_original", "issued_rfp", "formal_rfp"}
    for p in packets:
        for gid in p.get("governing_atom_ids") or []:
            a = atoms.get(str(gid))
            if not a:
                continue
            ac = str(a.get("authority_class", ""))
            if ac in bad_classes:
                return False, f"packet {p.get('id')} governed by {ac} atom {gid}"
    return True, ""


def _forbidden_total_row_entity(entities: list[dict]) -> tuple[bool, str]:
    for e in entities:
        ck = _norm(str(e.get("canonical_key", "") or ""))
        cn = _norm(str(e.get("canonical_name", "") or ""))
        if ck in {"total", "totals", "grand total"} or cn in {"total", "totals", "grand total"}:
            return False, f"entity {e.get('id')} looks like spreadsheet total row: {ck!r} / {cn!r}"
        if ck.endswith(":total") or ck.endswith(":totals"):
            return False, f"entity {e.get('id')} canonical_key ends with total-like token: {ck!r}"
    return True, ""


def _forbidden_power_in_scope_structured(packets: list[dict], atoms: dict[str, dict]) -> tuple[bool, str]:
    for p in packets:
        if str(p.get("family")) != "scope_inclusion":
            continue
        for aid in _packet_linked_atom_ids(p):
            b = _atom_blob(atoms.get(aid, {}))
            if re.search(r"\bpower\b", b) and re.search(r"\b(cat6|rj45|structured\s*cabling)\b", b):
                return False, f"scope_inclusion {p.get('id')} links power with structured cabling context"
    return True, ""


def _forbidden_deleted_text_governs(packets: list[dict], atoms: dict[str, dict]) -> tuple[bool, str]:
    for p in packets:
        for gid in p.get("governing_atom_ids") or []:
            a = atoms.get(str(gid))
            if a and a.get("authority_class") == "deleted_text":
                return False, f"packet {p.get('id')} governed by deleted_text {gid}"
    return True, ""


def _forbidden_quoted_old_email_governs_current(packets: list[dict], atoms: dict[str, dict]) -> tuple[bool, str]:
    for p in packets:
        gids = p.get("governing_atom_ids") or []
        if not gids:
            continue
        primary = atoms.get(str(gids[0]))
        if not primary or primary.get("authority_class") != "quoted_old_email":
            continue
        pool = set(_packet_linked_atom_ids(p))
        for aid in pool:
            a = atoms.get(aid)
            if a and a.get("authority_class") == "customer_current_authored":
                return False, f"packet {p.get('id')} governed by quoted_old_email while customer_current_authored present"
    return True, ""


_FORBIDDEN_DISPATCH: dict[str, Any] = {
    "vendor_quote_governs_scope": lambda payload: _forbidden_vendor_quote_governs_scope(
        payload.get("packets") or [], {a["id"]: a for a in payload.get("atoms") or [] if a.get("id")}
    ),
    "original_rfp_governs_over_current_addendum": lambda payload: _forbidden_original_rfp_over_addendum(
        payload.get("packets") or [], {a["id"]: a for a in payload.get("atoms") or [] if a.get("id")}
    ),
    "total_row_becomes_entity": lambda payload: _forbidden_total_row_entity(payload.get("entities") or []),
    "power_treated_as_in_scope_structured_cabling": lambda payload: _forbidden_power_in_scope_structured(
        payload.get("packets") or [], {a["id"]: a for a in payload.get("atoms") or [] if a.get("id")}
    ),
    "deleted_text_governs": lambda payload: _forbidden_deleted_text_governs(
        payload.get("packets") or [], {a["id"]: a for a in payload.get("atoms") or [] if a.get("id")}
    ),
    "quoted_old_email_governs_current_conflict": lambda payload: _forbidden_quoted_old_email_governs_current(
        payload.get("packets") or [], {a["id"]: a for a in payload.get("atoms") or [] if a.get("id")}
    ),
}


def _likely_layer(missing_reason: str | None, family: str) -> str:
    if not missing_reason:
        return "none"
    if missing_reason == "no_packet_matching_family_and_anchor":
        if family in {"quantity_conflict", "vendor_mismatch"}:
            return "graph_builder_or_packetizer"
        return "packetizer_or_labels"
    if missing_reason.startswith("missing_quantity") or missing_reason.startswith("missing_text"):
        return "packetizer_or_graph_builder"
    if "governing" in (missing_reason or ""):
        return "authority_or_packetizer"
    if missing_reason.startswith("status_"):
        return "packetizer_or_risk"
    return "unknown"


def _severity_for_row(found: bool, missing_reason: str | None) -> str:
    if found:
        return "none"
    if missing_reason == "no_packet_matching_family_and_anchor":
        return "critical"
    return "high"


@dataclass
class GoldCompareResult:
    case_dir: Path
    overall_pass: bool
    expected_results: list[dict[str, Any]] = field(default_factory=list)
    forbidden_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "case_dir": str(self.case_dir),
            "overall_pass": self.overall_pass,
            "expected_results": self.expected_results,
            "forbidden_results": self.forbidden_results,
            "errors": self.errors,
        }
        return d


def compare_gold_to_compile(
    *,
    gold: GoldPacketsFile,
    compile_payload: dict[str, Any],
    case_dir: Path | None = None,
) -> GoldCompareResult:
    atoms_list = compile_payload.get("atoms") or []
    atoms = {str(a["id"]): a for a in atoms_list if isinstance(a, dict) and a.get("id")}
    packets = [p for p in compile_payload.get("packets") or [] if isinstance(p, dict)]

    expected_results: list[dict[str, Any]] = []
    all_expected_ok = True
    for i, row in enumerate(gold.expected_packets):
        expected_id = row.id if row.id is not None else f"row_{i}"
        if not row.required:
            ok, matched, reason, evidence = _match_expected_row(row, packets, atoms)
            expected_results.append(
                {
                    "expected_id": expected_id,
                    "index": row.index if row.index is not None else i,
                    "family": row.family,
                    "found": ok,
                    "matched_packet_id": matched.get("id") if matched else None,
                    "evidence": evidence,
                    "missing_reason": reason,
                    "severity": "none" if ok else _severity_for_row(ok, reason),
                    "likely_layer_to_fix": "none" if ok else _likely_layer(reason, row.family),
                    "overall_pass": True,
                    "required": False,
                }
            )
            continue
        ok, matched, reason, evidence = _match_expected_row(row, packets, atoms)
        all_expected_ok = all_expected_ok and ok
        expected_results.append(
            {
                "expected_id": expected_id,
                "index": row.index if row.index is not None else i,
                "family": row.family,
                "found": ok,
                "matched_packet_id": matched.get("id") if matched else None,
                "evidence": evidence,
                "missing_reason": reason,
                "severity": _severity_for_row(ok, reason),
                "likely_layer_to_fix": _likely_layer(reason, row.family),
                "overall_pass": ok,
                "required": True,
            }
        )

    forbidden_results: list[dict[str, Any]] = []
    all_forbidden_ok = True
    for cond in gold.forbidden_conditions:
        fn = _FORBIDDEN_DISPATCH.get(cond)
        if fn is None:
            forbidden_results.append(
                {
                    "condition": cond,
                    "passed": False,
                    "details": f"unknown_forbidden_condition:{cond}",
                    "forbidden_pass": False,
                }
            )
            all_forbidden_ok = False
            continue
        passed, details = fn(compile_payload)
        all_forbidden_ok = all_forbidden_ok and passed
        forbidden_results.append(
            {
                "condition": cond,
                "passed": passed,
                "details": details,
                "forbidden_pass": passed,
            }
        )

    overall = all_forbidden_ok
    if gold.expected_packets:
        overall = overall and all_expected_ok

    return GoldCompareResult(
        case_dir=case_dir or Path("."),
        overall_pass=overall,
        expected_results=expected_results,
        forbidden_results=forbidden_results,
        errors=[],
    )


def compare_case_directory(case_dir: Path) -> GoldCompareResult:
    case_dir = case_dir.resolve()
    gold_path = case_dir / "labels" / "gold_packets.json"
    compile_path = case_dir / "outputs" / "compile_result.json"
    errors: list[str] = []
    if not gold_path.is_file():
        errors.append(f"missing_gold_file:{gold_path}")
    if not compile_path.is_file():
        errors.append(f"missing_compile_result:{compile_path}")
    if errors:
        return GoldCompareResult(case_dir=case_dir, overall_pass=False, errors=errors)

    gold = load_gold_packets(gold_path)
    compile_payload = json.loads(compile_path.read_text(encoding="utf-8"))
    return compare_gold_to_compile(gold=gold, compile_payload=compile_payload, case_dir=case_dir)


def render_markdown(report: GoldCompareResult) -> str:
    lines = [
        f"# Gold comparison — `{report.case_dir}`",
        "",
        f"**overall_pass:** `{report.overall_pass}`",
        "",
    ]
    if report.errors:
        lines.append("## Errors")
        for e in report.errors:
            lines.append(f"- {e}")
        lines.append("")
    lines.extend(["## Expected packets", "", "| id | family | found | packet | missing_reason | layer |", "|---|--------|-------|--------|----------------|-------|"])
    for r in report.expected_results:
        lines.append(
            f"| {r.get('expected_id')} | {r.get('family')} | {r.get('found')} | "
            f"{r.get('matched_packet_id') or ''} | {r.get('missing_reason') or ''} | {r.get('likely_layer_to_fix')} |"
        )
    lines.extend(["", "## Forbidden conditions", "", "| condition | pass | details |", "|-----------|------|---------|"])
    for r in report.forbidden_results:
        lines.append(f"| {r.get('condition')} | {r.get('passed')} | {str(r.get('details', ''))[:200]} |")
    lines.append("")
    failed_layers = sorted({r["likely_layer_to_fix"] for r in report.expected_results if not r.get("found")})
    lines.extend(["## Remaining failure layers", "", ", ".join(failed_layers) if failed_layers else "_none_", ""])
    return "\n".join(lines)


def write_comparison_outputs(case_dir: Path, result: GoldCompareResult) -> tuple[Path, Path]:
    out_dir = case_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gold_comparison.json"
    md_path = out_dir / "gold_comparison.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path
