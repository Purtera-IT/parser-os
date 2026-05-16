"""Persist a :class:`TakeoffDocument` as JSON / Markdown / EvidenceAtoms."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.takeoff.schemas import (
    DeviceInstance,
    LegendRule,
    QuoteLine,
    SheetRecord,
    TakeoffDocument,
)

TAKEOFF_FILENAME = "takeoff.json"
TAKEOFF_MARKDOWN_FILENAME = "takeoff.md"


# ──────────────────────────── JSON ────────────────────────────────────


def write_takeoff_doc(pdf_path: Path, takeoff: TakeoffDocument) -> Path:
    """Write the takeoff document JSON next to the derived directory."""
    derived = _derived_dir_for(pdf_path)
    derived.mkdir(parents=True, exist_ok=True)
    out = derived / TAKEOFF_FILENAME
    out.write_text(
        json.dumps(takeoff.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def write_takeoff_markdown(pdf_path: Path, takeoff: TakeoffDocument) -> Path:
    """Write the human-readable markdown projection next to the JSON."""
    derived = _derived_dir_for(pdf_path)
    derived.mkdir(parents=True, exist_ok=True)
    out = derived / TAKEOFF_MARKDOWN_FILENAME
    out.write_text(takeoff_doc_to_markdown(takeoff), encoding="utf-8")
    return out


def _derived_dir_for(pdf_path: Path) -> Path:
    """Match ``orbitbrief_pdf.derived_dir_for`` so both layers share dir."""
    pdf_path = Path(pdf_path)
    return pdf_path.with_name(f"{pdf_path.stem}.derived")


# ──────────────────────────── Markdown ────────────────────────────────


def takeoff_doc_to_markdown(takeoff: TakeoffDocument) -> str:
    """Render a human-readable markdown view of the takeoff document.

    The output is organized into the six sections required by the spec:
    Sheet Classification, Legend Rules, Device Counts by Sheet, WN
    Rollup, Quote Lines, Warnings / Open Questions.
    """
    lines: list[str] = []
    lines.append(f"# Low-Voltage Takeoff — {Path(takeoff.source_pdf).name}")
    lines.append("")
    lines.append(f"schema: {takeoff.schema_version}")
    lines.append("")

    # 1) Sheet Classification
    lines.append("## Sheet Classification")
    lines.append("")
    lines.append("| Page | Sheet | Name | Type | In Scope | Mult | Levels |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in takeoff.sheets:
        lines.append(
            "| {p} | {sn} | {nm} | {pt} | {inscope} | {m} | {lv} |".format(
                p=s.page_index,
                sn=s.sheet_number or "-",
                nm=s.sheet_name or "-",
                pt=s.page_type,
                inscope="yes" if s.in_scope else "NO",
                m=s.multiplier,
                lv=", ".join(s.levels_represented) or "-",
            )
        )
    lines.append("")

    # 2) Legend Rules
    lines.append("## Legend Rules")
    lines.append("")
    lines.append("| Symbol | Class | System | Quote unit | Cable |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in takeoff.legend_rules:
        lines.append(
            "| {s} | {c} | {sy} | {q} | {ct} |".format(
                s=r.raw_symbol,
                c=r.normalized_class,
                sy=r.system,
                q=r.quote_unit or "-",
                ct=r.cable_type or "-",
            )
        )
    lines.append("")

    # 3) Device Counts by Sheet
    lines.append("## Device Counts by Sheet")
    lines.append("")
    counts_by_sheet = _device_counts_by_sheet(takeoff.devices)
    lines.append("| Sheet | Class | Base | Multiplier | Extended |")
    lines.append("| --- | --- | --- | --- | --- |")
    for (sheet_number, normalized_class), payload in sorted(counts_by_sheet.items()):
        base = payload["base"]
        mult = payload["multiplier"]
        ext = payload["extended"]
        lines.append(
            f"| {sheet_number or '-'} | {normalized_class} | {base} | {mult} | {ext} |"
        )
    lines.append("")

    # 4) WN Rollup
    lines.append("## WN Rollup")
    lines.append("")
    wn = (takeoff.summary or {}).get("wireless_node_outlet", {})
    if wn:
        lines.append(
            f"- base_floor_plan_count: {wn.get('base_floor_plan_count', 0)}"
        )
        lines.append(f"- extended_count: {wn.get('extended_count', 0)}")
        lines.append(
            f"- excluded_not_in_scope_count: {wn.get('excluded_not_in_scope_count', 0)}"
        )
        lines.append(
            f"- rejected_non_plan_count: {wn.get('rejected_non_plan_count', 0)}"
        )
    else:
        lines.append("- no WN devices detected")
    lines.append("")

    # 5) Quote Lines
    lines.append("## Quote Lines")
    lines.append("")
    lines.append("| Item | Description | Qty | Unit | Floor | Home-run to |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for q in takeoff.quote_lines:
        lines.append(
            "| {ik} | {desc} | {qty} | {u} | {fl} | {hr} |".format(
                ik=q.item_key,
                desc=q.description,
                qty=q.quantity,
                u=q.unit,
                fl=q.floor_label or "-",
                hr=q.home_run_to or "-",
            )
        )
    lines.append("")

    # 6) Warnings / Open Questions
    lines.append("## Warnings / Open Questions")
    lines.append("")
    if takeoff.warnings:
        for w in takeoff.warnings:
            lines.append(f"- WARNING: {w}")
    if takeoff.open_questions:
        for q in takeoff.open_questions:
            lines.append(f"- OPEN: {q}")
    if not takeoff.warnings and not takeoff.open_questions:
        lines.append("- (none)")
    lines.append("")

    return "\n".join(lines)


def _device_counts_by_sheet(
    devices: list[DeviceInstance],
) -> dict[tuple[str | None, str], dict[str, int]]:
    counts: dict[tuple[str | None, str], dict[str, int]] = {}
    for d in devices:
        key = (d.sheet_number, d.normalized_class)
        slot = counts.setdefault(
            key, {"base": 0, "multiplier": d.multiplier, "extended": 0}
        )
        slot["base"] += 1
        slot["multiplier"] = d.multiplier
        slot["extended"] += d.multiplier
    return counts


# ──────────────────────────── Atoms ───────────────────────────────────


def takeoff_to_atoms(
    *,
    takeoff: TakeoffDocument,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    """Emit one EvidenceAtom per sheet+class rollup, plus globals.

    The spec is explicit: do NOT emit one atom per WN. Each sheet+class
    becomes one quantity atom; one extra atom for the overall WN
    extended total; one open_question per ambiguous-zone warning; one
    assumption for the Wi-Fi vendor survey caveat.
    """
    # 1) Per-sheet, per-class rollup atoms.
    counts = _device_counts_by_sheet(takeoff.devices)
    devices_by_key: dict[tuple[str | None, str], list[DeviceInstance]] = {}
    for d in takeoff.devices:
        devices_by_key.setdefault((d.sheet_number, d.normalized_class), []).append(d)

    for (sheet_number, normalized_class), payload in sorted(counts.items()):
        device_ids = [d.id for d in devices_by_key.get((sheet_number, normalized_class), [])]
        page_index = devices_by_key[(sheet_number, normalized_class)][0].page_index
        source = _build_source_ref(
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            locator={
                "page": page_index,
                "sheet_number": sheet_number,
                "takeoff_kind": "sheet_class_rollup",
                "normalized_class": normalized_class,
                "device_ids": device_ids,
            },
        )
        raw = (
            f"{normalized_class}: {payload['base']} base x{payload['multiplier']} "
            f"= {payload['extended']} on sheet {sheet_number or '?'}"
        )
        atom_id = stable_id(
            "atom_takeoff",
            sheet_number or "?",
            normalized_class,
            "rollup",
        )
        yield EvidenceAtom(
            id=atom_id,
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.quantity,
            raw_text=raw,
            normalized_text=raw,
            value={
                "normalized_class": normalized_class,
                "sheet_number": sheet_number,
                "base_count": payload["base"],
                "multiplier": payload["multiplier"],
                "extended_count": payload["extended"],
            },
            entity_keys=[normalized_class, sheet_number or "?"],
            source_refs=[source],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.94,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version=parser_version,
        )

    # 2) WN-total atom.
    wn_summary = (takeoff.summary or {}).get("wireless_node_outlet")
    if wn_summary:
        device_ids = [d.id for d in takeoff.devices if d.normalized_class == "wireless_node_outlet"]
        source = _build_source_ref(
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            locator={
                "page": None,
                "sheet_number": None,
                "takeoff_kind": "wireless_node_outlet_total",
                "normalized_class": "wireless_node_outlet",
                "device_ids": device_ids,
            },
        )
        raw = (
            f"wireless_node_outlet: {wn_summary.get('extended_count', 0)} drops "
            f"(extended) across {wn_summary.get('base_floor_plan_count', 0)} base devices"
        )
        yield EvidenceAtom(
            id=stable_id("atom_takeoff", "wireless_node_outlet", "total"),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.quantity,
            raw_text=raw,
            normalized_text=raw,
            value=dict(wn_summary),
            entity_keys=["wireless_node_outlet"],
            source_refs=[source],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.94,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version=parser_version,
        )

    # 3) Open-question atom per warning/open question.
    for idx, q in enumerate(takeoff.open_questions or []):
        source = _build_source_ref(
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            locator={
                "page": None,
                "sheet_number": None,
                "takeoff_kind": "open_question",
                "normalized_class": None,
                "candidate_ids": [],
            },
        )
        yield EvidenceAtom(
            id=stable_id("atom_takeoff", "open_question", idx, q[:60]),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.open_question,
            raw_text=q,
            normalized_text=q,
            value={"question": q},
            entity_keys=["takeoff_open_question"],
            source_refs=[source],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.6,
            review_status=ReviewStatus.needs_review,
            review_flags=["takeoff_open_question"],
            parser_version=parser_version,
        )

    # 4) Wi-Fi vendor survey assumption (always emitted for WN takeoffs).
    if any(d.normalized_class == "wireless_node_outlet" for d in takeoff.devices):
        source = _build_source_ref(
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            locator={
                "page": None,
                "sheet_number": None,
                "takeoff_kind": "assumption_wifi_survey",
                "normalized_class": "wireless_node_outlet",
            },
        )
        text = (
            "Final WAP locations are subject to a Wi-Fi vendor predictive / "
            "post-installation survey. Counts reflect the drawing set; vendor "
            "may relocate access points."
        )
        yield EvidenceAtom(
            id=stable_id("atom_takeoff", "assumption", "wifi_survey"),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.assumption,
            raw_text=text,
            normalized_text=text,
            value={"assumption": "wifi_vendor_survey"},
            entity_keys=["wireless_node_outlet", "wifi_vendor_survey"],
            source_refs=[source],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.9,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version=parser_version,
        )


def _build_source_ref(
    *,
    artifact_id: str,
    filename: str,
    parser_version: str,
    locator: dict[str, Any],
) -> SourceRef:
    return SourceRef(
        id=stable_id("src", artifact_id, "takeoff", json.dumps(locator, sort_keys=True, default=str)),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator=locator,
        extraction_method="takeoff_low_voltage_v1",
        parser_version=parser_version,
    )


# ──────────────────────────── Summary ─────────────────────────────────


def takeoff_summary(
    sheets: list[SheetRecord],
    devices: list[DeviceInstance],
    candidates_by_class: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Compute the ``summary`` block on a :class:`TakeoffDocument`.

    The returned dict is keyed by ``normalized_class`` so the WN rollup
    sits at ``summary["wireless_node_outlet"]`` per the spec.
    """
    summary: dict[str, Any] = {}
    classes: set[str] = {d.normalized_class for d in devices}
    classes.update((candidates_by_class or {}).keys())

    floor_plan_pages = {s.page_index for s in sheets if s.page_type in {"floor_plan", "typical_plan"}}

    for cls in sorted(classes):
        cls_devices = [d for d in devices if d.normalized_class == cls]
        base = len(cls_devices)
        extended = sum(d.multiplier for d in cls_devices)
        cand_payload = (candidates_by_class or {}).get(cls, {})
        summary[cls] = {
            "base_floor_plan_count": base,
            "extended_count": extended,
            "excluded_not_in_scope_count": int(cand_payload.get("not_in_scope", 0)),
            "rejected_non_plan_count": int(cand_payload.get("non_plan_page", 0)),
            "rejected_outside_viewport_count": int(cand_payload.get("outside_viewport", 0)),
        }

    summary["sheet_counts"] = {
        "total": len(sheets),
        "floor_plan_pages": len(floor_plan_pages),
    }
    return summary


__all__ = [
    "TAKEOFF_FILENAME",
    "TAKEOFF_MARKDOWN_FILENAME",
    "write_takeoff_doc",
    "write_takeoff_markdown",
    "takeoff_doc_to_markdown",
    "takeoff_to_atoms",
    "takeoff_summary",
]
