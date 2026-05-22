"""OrbitBrief project envelope — the "perfect compressible" LLM input.

A single ``orbitbrief.input.v1`` envelope packages every artifact in a
project into one self-contained payload an open-source LLM (Llama-3.1
70B, Qwen-2.5 72B, Mistral-Large, etc.) can consume in a single prompt.

Two output formats are produced from the same in-memory envelope:

* ``orbitbrief.input.json`` — strict, machine-consumable, deterministic.
  Used by code-side consumers and as the source-of-truth for replay.
* ``orbitbrief.input.md`` — token-efficient markdown projection.  Same
  hierarchy, with stable ``<a id="..."></a>`` anchors so an LLM can
  cite a region by anchor and a UI can scroll to the same place.

Envelope shape (JSON)::

    {
      "schema_version": "orbitbrief.input.v1",
      "project_id": "...",
      "compile_id": "...",
      "generated_at": "ISO-8601 UTC",
      "summary": {
          "artifact_count": int,
          "page_count": int,
          "atom_count": int,
          "packet_count": int,
          "by_atom_type": {AtomType.value: int, ...},
          "by_authority_class": {AuthorityClass.value: int, ...},
      },
      "documents": [
          {
              "artifact_id": "...",
              "filename": "...",
              "artifact_type": "pdf|docx|xlsx|csv|email|transcript|txt",
              "parser_name": "...",
              "parser_version": "...",
              "structured": <full PDF structured doc | projected envelope>,
              "atom_ids": ["atm_...", ...],
          },
          ...
      ],
      "atoms": [<compact atom rows>, ...],
      "packets": [<compact packet rows>, ...],
      "indexes": {
          "atoms_by_section_path": {"a > b": ["atm_..."]},
          "atoms_by_atom_type":    {"scope_item": ["atm_..."]},
          "atoms_by_authority":    {"contractual_scope": ["atm_..."]},
      },
    }

A compact atom row is intentionally small — OrbitBrief can fetch the
full ``EvidenceAtom`` from the compile result if needed.  The envelope
is the "swallow it whole" view; the compile result is the audit log.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.orbitbrief_core import (
    build_change_order_timeline,
    build_pm_dashboard,
    build_project_vitals,
    build_scope_truth,
    build_site_readiness,
    build_sow_readiness_scorecard,
    build_srl_missing_checklist,
    build_stakeholder_load,
)
from app.core.schemas import (
    ArtifactType,
    CompileResult,
    EntityRecord,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    SourceRef,
)
from app.parsers.structured_projection import (
    DERIVED_DIR_SUFFIX,
    STRUCTURED_FILENAME,
    structured_doc_to_markdown,
)

ENVELOPE_SCHEMA_VERSION = "orbitbrief.input.v2"
ENVELOPE_FILENAME = "orbitbrief.input.json"
ENVELOPE_MARKDOWN_FILENAME = "orbitbrief.input.md"


# ────────────────────────── public API ───────────────────────────────────


def build_orbitbrief_envelope(
    *,
    project_dir: Path,
    compile_result: CompileResult,
) -> dict[str, Any]:
    """Build the in-memory envelope from a compile result.

    The compile result already carries every parsed atom, packet, and the
    manifest with parser routing — we just need to fuse those together
    with each artifact's structured projection (PDFs use their persisted
    ``structured.json``; non-PDF parsers get a synthesized projection
    from their atoms grouped by section path).
    """
    project_dir = Path(project_dir).resolve()
    manifest = compile_result.manifest
    atoms = list(compile_result.atoms or [])
    packets = list(compile_result.packets or [])
    entities = list(compile_result.entities or [])
    edges = list(compile_result.edges or [])

    atoms_by_artifact: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_artifact[atom.artifact_id].append(atom)

    # A6 graceful degradation: build a per-file outcome index from the
    # manifest's parser_routing so each document carries its own
    # status (ok / ok_empty / skipped_no_parser / failed_parse).
    # PM_HANDOFF builders read this to render a "Files processed"
    # table and avoid the silent failure where a parse error left the
    # file count looking normal but produced 0 evidence.
    outcome_by_artifact: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        for routing_entry in (manifest.parser_routing or []):
            aid = routing_entry.get("artifact_id")
            outcome = routing_entry.get("outcome")
            if aid and isinstance(outcome, dict):
                outcome_by_artifact[aid] = outcome

    documents: list[dict[str, Any]] = []
    artifact_iter = manifest.artifact_fingerprints if manifest is not None else []
    for fp in artifact_iter:
        artifact_atoms = atoms_by_artifact.get(fp.artifact_id, [])
        artifact_path = _resolve_artifact_path(project_dir, fp.filename)
        structured_projection = _structured_projection_for(
            artifact_path=artifact_path,
            artifact_type=fp.artifact_type,
            artifact_atoms=artifact_atoms,
            filename=fp.filename,
        )
        documents.append(
            {
                "artifact_id": fp.artifact_id,
                "filename": fp.filename,
                "artifact_type": fp.artifact_type.value,
                "sha256": fp.sha256,
                "size_bytes": fp.size_bytes,
                "parser_name": fp.parser_name,
                "parser_version": fp.parser_version,
                "structured": structured_projection,
                "atom_ids": sorted(a.id for a in artifact_atoms),
                # A6 graceful degradation: per-file parse outcome.
                # ``status`` is one of ok / ok_empty / skipped_no_parser
                # / failed_parse. PM_HANDOFF reads this to surface
                # files that the engineer should manually inspect.
                "parse_outcome": outcome_by_artifact.get(
                    fp.artifact_id,
                    {"status": "unknown", "atom_count": len(artifact_atoms), "warning_count": 0},
                ),
            }
        )

    summary = _build_summary(
        atoms=atoms,
        packets=packets,
        documents=documents,
        entities=entities,
        edges=edges,
    )
    indexes = _build_indexes(atoms=atoms, entities=entities, edges=edges)
    drawings = _build_drawings_section(
        atoms=atoms,
        packets=packets,
        edges=edges,
        atoms_by_artifact=atoms_by_artifact,
        documents=documents,
    )

    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "project_id": compile_result.project_id,
        "compile_id": compile_result.compile_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "documents": documents,
        "atoms": [_compact_atom(a) for a in atoms],
        "packets": [_compact_packet(p) for p in packets],
        "entities": [_compact_entity(e, atoms_by_artifact, atoms) for e in entities],
        "edges": [_compact_edge(edge) for edge in edges],
        "indexes": indexes,
    }
    # OrbitBrief-Core deliverables — deterministic pre-aggregations so
    # the downstream LLM synthesis layer (and the PM cockpit) can render
    # the Monday-morning view, the SOW-readiness scorecard, and the
    # required-fields checklist directly without re-scanning atoms.
    envelope["pm_dashboard"] = build_pm_dashboard(
        atoms=atoms, packets=packets, edges=edges, entities=entities,
    )
    envelope["sow_readiness_scorecard"] = build_sow_readiness_scorecard(
        atoms=atoms, packets=packets, edges=edges, entities=entities,
    )
    envelope["srl_missing_checklist"] = build_srl_missing_checklist(
        atoms=atoms, documents=documents,
    )
    # S+++++ cockpit surfaces — authority-weighted scope truth,
    # chronological change-order audit, per-site readiness rollup,
    # per-stakeholder workload matrix, and a single 0-100 project
    # vitals number that blends every signal above into one
    # auditable cockpit-header score.
    envelope["scope_truth"] = build_scope_truth(atoms=atoms, edges=edges)
    envelope["change_order_timeline"] = build_change_order_timeline(atoms=atoms)
    envelope["site_readiness"] = build_site_readiness(atoms=atoms, edges=edges)
    envelope["stakeholder_load"] = build_stakeholder_load(atoms=atoms)
    envelope["project_vitals"] = build_project_vitals(
        atoms=atoms,
        edges=edges,
        packets=packets,
        scorecard=envelope["sow_readiness_scorecard"],
        checklist=envelope["srl_missing_checklist"],
        site_readiness=envelope["site_readiness"],
        stakeholder_load=envelope["stakeholder_load"],
        scope_truth=envelope["scope_truth"],
    )
    # Drawings section is omitted entirely on non-schematic projects so
    # the envelope shape stays byte-identical for the existing test grid.
    if drawings["artifacts"]:
        envelope["drawings"] = drawings
    return envelope


def write_orbitbrief_envelope(
    *,
    project_dir: Path,
    envelope: dict[str, Any],
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the envelope as both JSON and markdown.

    Returns ``(json_path, markdown_path)``.  Defaults to writing under
    ``<project_dir>/.orbitbrief/`` so consumers know exactly where to
    look.  Pass ``out_dir`` to override.
    """
    out_dir = Path(out_dir) if out_dir is not None else (Path(project_dir) / ".orbitbrief")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ENVELOPE_FILENAME
    md_path = out_dir / ENVELOPE_MARKDOWN_FILENAME
    json_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(envelope_to_markdown(envelope), encoding="utf-8")
    return json_path, md_path


def envelope_to_markdown(envelope: dict[str, Any]) -> str:
    """Render the envelope as a single LLM-ready markdown document.

    The output is a concatenation of every document's structured
    markdown, separated by clear horizontal rules and tagged with the
    artifact id so anchors stay unique even across documents.
    """
    lines: list[str] = []
    lines.append("---")
    lines.append(f"schema: {envelope.get('schema_version', ENVELOPE_SCHEMA_VERSION)}")
    lines.append(f"project_id: {envelope.get('project_id', '')}")
    lines.append(f"compile_id: {envelope.get('compile_id', '')}")
    lines.append(f"generated_at: {envelope.get('generated_at', '')}")
    lines.append("---")
    lines.append("")

    summary = envelope.get("summary") or {}
    if summary:
        lines.append("# OrbitBrief Project Envelope")
        lines.append("")
        lines.append(
            f"_{summary.get('artifact_count', 0)} artifacts • "
            f"{summary.get('page_count', 0)} pages • "
            f"{summary.get('atom_count', 0)} atoms • "
            f"{summary.get('packet_count', 0)} packets_"
        )
        lines.append("")
        by_type = summary.get("by_atom_type") or {}
        if by_type:
            lines.append("**Atoms by type**")
            for atom_type, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {atom_type}: {count}")
            lines.append("")

    for doc in envelope.get("documents", []) or []:
        artifact_id = doc.get("artifact_id", "")
        filename = doc.get("filename", "")
        artifact_type = doc.get("artifact_type", "")
        lines.append("---")
        lines.append("")
        lines.append(f'<!-- artifact id="{artifact_id}" type="{artifact_type}" -->')
        lines.append(f"## File: {filename}")
        lines.append("")
        structured = doc.get("structured") or {}
        schema = (structured.get("schema_version") if isinstance(structured, dict) else "") or ""
        if schema and schema != "orbitbrief.atom_projection.v1":
            # Every "real" structured doc — PDF, XLSX, DOCX, transcript,
            # email, quote — uses the unified renderer.
            lines.append(structured_doc_to_markdown(structured))
        else:
            lines.append(_render_generic_structured_md(structured))
        lines.append("")

    entities = envelope.get("entities") or []
    if entities:
        lines.append("---")
        lines.append("")
        lines.append("## Entities (cross-artifact)")
        lines.append("")
        lines.append("| Type | Canonical | Aliases | Artifacts | Atoms |")
        lines.append("|---|---|---|---|---|")
        for entity in entities:
            lines.append(
                "| {type} | {name} | {aliases} | {arts} | {atoms} |".format(
                    type=entity.get("entity_type", ""),
                    name=entity.get("canonical_name", ""),
                    aliases=", ".join((entity.get("aliases") or [])[:6]),
                    arts=len(entity.get("artifact_ids") or []),
                    atoms=len(entity.get("source_atom_ids") or []),
                )
            )
        lines.append("")

    edges = envelope.get("edges") or []
    if edges:
        lines.append("---")
        lines.append("")
        lines.append("## Cross-references and contradictions")
        lines.append("")
        lines.append("| Edge type | From | To | Cross-artifact | Reason |")
        lines.append("|---|---|---|---|---|")
        for edge in edges:
            lines.append(
                "| {type} | {fa} | {ta} | {ca} | {reason} |".format(
                    type=edge.get("edge_type", ""),
                    fa=edge.get("from_atom_id", ""),
                    ta=edge.get("to_atom_id", ""),
                    ca="yes" if edge.get("cross_artifact") else "no",
                    reason=(edge.get("reason") or "").replace("|", "\\|"),
                )
            )
        lines.append("")

    drawings = envelope.get("drawings") or {}
    artifacts = drawings.get("artifacts") or []
    if artifacts:
        lines.append("---")
        lines.append("")
        lines.append("## Drawings")
        lines.append("")
        idx = drawings.get("indexes") or {}
        det_counts = idx.get("detections_by_target_key") or {}
        if det_counts:
            lines.append("**Detection counts across all drawings**")
            for target_key, count in sorted(det_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {target_key}: {count}")
            lines.append("")
        warn_counts = idx.get("warnings_by_type") or {}
        if warn_counts:
            lines.append("**Warnings across all drawings**")
            for wt, count in sorted(warn_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {wt}: {count}")
            lines.append("")
        for art in artifacts:
            lines.append(f"### {art.get('filename') or art.get('artifact_id') or 'drawing'}")
            lines.append("")
            qc_ids = art.get("quantity_conflict_packet_ids") or []
            if qc_ids:
                lines.append(
                    f"_{len(qc_ids)} quantity_conflict packet(s): "
                    + ", ".join(qc_ids)
                    + "_"
                )
                lines.append("")
            for page in art.get("pages", []) or []:
                p = page.get("page")
                sn = page.get("sheet_number") or "?"
                lines.append(f"#### Page {p} — Sheet {sn}")
                meta = page.get("sheet_metadata") or {}
                if meta:
                    parts: list[str] = []
                    for k in ("sheet_title", "project_name", "scale", "issue_date", "revision"):
                        v = meta.get(k)
                        if v:
                            parts.append(f"{k}={v}")
                    if parts:
                        lines.append("- " + " • ".join(parts))
                target_counts = page.get("target_counts") or {}
                if target_counts:
                    lines.append("- Target counts: " + ", ".join(
                        f"{k}={v}" for k, v in sorted(target_counts.items())
                    ))
                rooms = page.get("rooms") or []
                if rooms:
                    lines.append(
                        "- Rooms: "
                        + ", ".join(
                            f"{r.get('label')}{(' ' + r['number']) if r.get('number') else ''}"
                            for r in rooms
                        )
                    )
                notes = page.get("keyed_notes") or []
                if notes:
                    lines.append(f"- Keyed notes: {len(notes)}")
                schedules = page.get("schedule_rows") or []
                if schedules:
                    lines.append(f"- Schedule rows: {len(schedules)}")
                warnings = page.get("warnings") or []
                if warnings:
                    types = sorted({w.get("warning_type") for w in warnings if w.get("warning_type")})
                    lines.append("- Warnings: " + ", ".join(types))
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ────────────────────────── internals ────────────────────────────────────


def _resolve_artifact_path(project_dir: Path, filename: str) -> Path:
    """Resolve a stored artifact file inside the project dir.

    Manifest filenames are project-relative (forward-slash normalized).
    """
    rel = filename.replace("\\", "/")
    return (project_dir / rel).resolve()


def _structured_projection_for(
    *,
    artifact_path: Path,
    artifact_type: ArtifactType,
    artifact_atoms: list[EvidenceAtom],
    filename: str,
) -> dict[str, Any]:
    """Build the per-artifact ``structured`` payload for the envelope.

    Every Parser OS parser that opts in writes a structured doc to
    ``<stem>.derived/structured.json`` — load it directly so the
    envelope's markdown projection has the same fidelity for PDFs,
    XLSX/CSV workbooks, DOCX documents, email threads, transcripts,
    and vendor quotes.  When a parser hasn't produced one yet (legacy
    or unsupported artifacts), synthesize a flat atom projection so
    the artifact still shows up in the envelope.
    """
    derived = artifact_path.with_name(f"{artifact_path.stem}{DERIVED_DIR_SUFFIX}") / STRUCTURED_FILENAME
    if derived.is_file():
        try:
            return json.loads(derived.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return _project_atoms_to_structured(
        atoms=artifact_atoms,
        filename=filename,
        artifact_type=artifact_type,
    )


def _project_atoms_to_structured(
    *,
    atoms: list[EvidenceAtom],
    filename: str,
    artifact_type: ArtifactType,
) -> dict[str, Any]:
    """Synthesize a structured doc from a flat list of atoms.

    Group atoms by their ``SourceRef.locator['section_path']`` (or
    ``location`` / ``sheet`` / ``speaker`` for legacy locators) so the
    markdown projection still has section structure.
    """
    sections_by_path: dict[tuple[str, ...], dict[str, Any]] = {}
    for atom in atoms:
        path = _atom_section_path(atom)
        key = tuple(path)
        section = sections_by_path.get(key)
        if section is None:
            section = {
                "id": f"sec_{abs(hash(key)) % (10**12):012d}",
                "level": max(len(path), 1),
                "heading": path[-1] if path else "",
                "blocks": [],
                "subsections": [],
            }
            sections_by_path[key] = section
        section["blocks"].append(
            {
                "id": atom.id.replace("atm_", "blk_"),
                "kind": _atom_to_block_kind(atom),
                "text": atom.raw_text,
            }
        )

    sections = [sections_by_path[k] for k in sections_by_path]
    return {
        "schema_version": "orbitbrief.atom_projection.v1",
        "source": {"filename": filename, "artifact_type": artifact_type.value},
        "document": {"title": filename, "metadata": []},
        "pages": [
            {
                "page": 0,
                "title": filename,
                "metadata": [],
                "outline": [
                    {
                        "level": s["level"],
                        "heading": s["heading"],
                        "block_count": len(s["blocks"]),
                    }
                    for s in sections
                ],
                "sections": sections,
            }
        ],
    }


def _atom_section_path(atom: EvidenceAtom) -> list[str]:
    if not atom.source_refs:
        return []
    ref: SourceRef = atom.source_refs[0]
    locator = ref.locator or {}
    section_path = locator.get("section_path")
    if isinstance(section_path, list) and section_path:
        return [str(x) for x in section_path]
    # Fall back to whatever locator field gives us section-ish context.
    fallback_keys = ("section", "sheet", "speaker", "channel", "location")
    for key in fallback_keys:
        value = locator.get(key)
        if value:
            return [str(value)]
    return []


def _atom_to_block_kind(atom: EvidenceAtom) -> str:
    if atom.source_refs:
        locator_kind = atom.source_refs[0].locator.get("block_kind")
        if isinstance(locator_kind, str) and locator_kind:
            return locator_kind
    return "text"


def _build_drawings_section(
    *,
    atoms: list[EvidenceAtom],
    packets: list[Any],
    edges: list[Any],
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the schematic ``drawings`` envelope section.

    Groups every schematic_* atom by (artifact, page), surfaces the
    parsed legend, the per-page target counts, the resolved schedule
    rows, the keyed notes, the rooms, the warnings, and any
    quantity_conflict packets that came out of schematic atoms.

    Empty by design when the project has no schematic atoms — the
    envelope's caller drops the section in that case so non-schematic
    projects produce byte-identical output.
    """
    schematic_atom_types = {
        "schematic_sheet_metadata",
        "schematic_legend",
        "schematic_room",
        "schematic_keyed_note",
        "schematic_note_callout",
        "schematic_schedule_row",
        "schematic_detection_target_set",
        "schematic_symbol_detection",
        "schematic_warning",
    }
    schematic_atoms = [a for a in atoms if a.atom_type.value in schematic_atom_types]
    if not schematic_atoms:
        return {"artifacts": [], "indexes": {}}

    artifact_filenames = {d["artifact_id"]: d.get("filename") for d in documents}

    by_art: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for a in schematic_atoms:
        by_art[a.artifact_id].append(a)

    artifacts_out: list[dict[str, Any]] = []
    drawings_by_sheet: dict[str, list[str]] = defaultdict(list)
    detections_by_target: dict[str, int] = defaultdict(int)
    warnings_by_type: dict[str, int] = defaultdict(int)

    for artifact_id in sorted(by_art):
        art_atoms = by_art[artifact_id]
        per_page: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "sheet_number": None,
                "sheet_metadata": None,
                "legend_id": None,
                "target_counts": defaultdict(int),
                "warnings": [],
                "rooms": [],
                "keyed_notes": [],
                "schedule_rows": [],
                "atom_ids": [],
            }
        )
        legends_out: list[dict[str, Any]] = []
        for atom in art_atoms:
            value = atom.value if isinstance(atom.value, dict) else {}
            page = value.get("page")
            if isinstance(page, int):
                per_page[page]["atom_ids"].append(atom.id)
            atom_kind = atom.atom_type.value
            if atom_kind == "schematic_sheet_metadata":
                if isinstance(page, int):
                    per_page[page]["sheet_metadata"] = {
                        k: v for k, v in value.items() if k != "page"
                    }
                    per_page[page]["sheet_number"] = value.get("sheet_number")
                    sn = value.get("sheet_number")
                    if isinstance(sn, str) and sn:
                        drawings_by_sheet[sn].append(atom.id)
            elif atom_kind == "schematic_legend":
                legends_out.append(
                    {
                        "legend_id": value.get("legend_id"),
                        "page": value.get("page"),
                        "sheet_number": value.get("sheet_number"),
                        "scope": value.get("scope"),
                        "entry_count": value.get("entry_count"),
                    }
                )
            elif atom_kind == "schematic_detection_target_set":
                if isinstance(page, int):
                    per_page[page]["legend_id"] = value.get("legend_id")
            elif atom_kind == "schematic_symbol_detection":
                tk = value.get("target_key")
                if isinstance(page, int) and isinstance(tk, str):
                    per_page[page]["target_counts"][tk] += 1
                    detections_by_target[tk] += 1
            elif atom_kind == "schematic_warning":
                wt = value.get("warning_type")
                if isinstance(page, int):
                    per_page[page]["warnings"].append(
                        {
                            "warning_type": wt,
                            "detail": value.get("detail"),
                            "target_key": value.get("target_key"),
                        }
                    )
                if isinstance(wt, str):
                    warnings_by_type[wt] += 1
            elif atom_kind == "schematic_room":
                if isinstance(page, int):
                    per_page[page]["rooms"].append(
                        {
                            "room_id": value.get("room_id"),
                            "label": value.get("label"),
                            "number": value.get("number"),
                        }
                    )
            elif atom_kind == "schematic_keyed_note":
                if isinstance(page, int):
                    per_page[page]["keyed_notes"].append(
                        {
                            "number": value.get("number"),
                            "text": value.get("text"),
                            "callout_count": value.get("callout_count", 0),
                        }
                    )
            elif atom_kind == "schematic_schedule_row":
                if isinstance(page, int):
                    per_page[page]["schedule_rows"].append(
                        {
                            "row_id": value.get("row_id"),
                            "schedule_kind": value.get("schedule_kind"),
                            "tag": value.get("tag"),
                            "fields": value.get("fields", {}),
                        }
                    )

        # Schematic quantity conflicts on this artifact.
        artifact_packet_ids: list[str] = []
        artifact_atom_ids = {a.id for a in art_atoms}
        for p in packets:
            if p.family.value != "quantity_conflict":
                continue
            packet_atom_ids = set(
                (p.contradicting_atom_ids or []) + (p.governing_atom_ids or [])
            )
            if packet_atom_ids & artifact_atom_ids:
                artifact_packet_ids.append(p.id)

        # Stabilize per_page payloads (dict -> dict).
        pages_out = []
        for page_index in sorted(per_page):
            entry = per_page[page_index]
            pages_out.append(
                {
                    "page": page_index,
                    "sheet_number": entry["sheet_number"],
                    "legend_id": entry["legend_id"],
                    "sheet_metadata": entry["sheet_metadata"],
                    "target_counts": dict(sorted(entry["target_counts"].items())),
                    "warnings": entry["warnings"],
                    "rooms": entry["rooms"],
                    "keyed_notes": entry["keyed_notes"],
                    "schedule_rows": entry["schedule_rows"],
                    "atom_ids": sorted(entry["atom_ids"]),
                }
            )

        artifacts_out.append(
            {
                "artifact_id": artifact_id,
                "filename": artifact_filenames.get(artifact_id),
                "pages": pages_out,
                "legends": sorted(legends_out, key=lambda l: (l.get("page") or 0, l.get("legend_id") or "")),
                "quantity_conflict_packet_ids": sorted(artifact_packet_ids),
            }
        )

    return {
        "artifacts": artifacts_out,
        "indexes": {
            "drawings_by_sheet_number": {
                k: sorted(v) for k, v in sorted(drawings_by_sheet.items())
            },
            "detections_by_target_key": dict(sorted(detections_by_target.items())),
            "warnings_by_type": dict(sorted(warnings_by_type.items())),
        },
    }


def _build_summary(
    *,
    atoms: list[EvidenceAtom],
    packets: list[EvidencePacket],
    documents: list[dict[str, Any]],
    entities: list[EntityRecord] | None = None,
    edges: list[EvidenceEdge] | None = None,
) -> dict[str, Any]:
    entities = entities or []
    edges = edges or []
    by_atom_type: Counter[str] = Counter(a.atom_type.value for a in atoms)
    by_authority: Counter[str] = Counter(a.authority_class.value for a in atoms)
    by_artifact_type: Counter[str] = Counter(d.get("artifact_type", "") for d in documents)
    by_edge_type: Counter[str] = Counter(e.edge_type.value for e in edges)
    by_entity_type: Counter[str] = Counter(e.entity_type for e in entities)
    cross_artifact_edges = sum(1 for e in edges if e.metadata.get("cross_artifact"))
    page_count = 0
    for doc in documents:
        structured = doc.get("structured") or {}
        if isinstance(structured, dict):
            page_count += len(structured.get("pages") or [])
    # A6 graceful degradation: roll up per-file parse_outcome into a
    # summary counter + an explicit degraded-files list. PM_HANDOFF
    # uses this to render a "Files requiring manual review" callout.
    parse_outcomes_counter: Counter[str] = Counter()
    degraded_files: list[dict[str, str]] = []
    for doc in documents:
        outcome = doc.get("parse_outcome") or {}
        status = outcome.get("status") or "unknown"
        parse_outcomes_counter[status] += 1
        if status in {"failed_parse", "skipped_no_parser", "ok_empty"}:
            degraded_files.append({
                "filename": str(doc.get("filename", "")),
                "status": status,
                "reason": str(outcome.get("reason", ""))[:300],
            })
    return {
        "artifact_count": len(documents),
        "page_count": page_count,
        "atom_count": len(atoms),
        "packet_count": len(packets),
        "entity_count": len(entities),
        "edge_count": len(edges),
        "cross_artifact_edge_count": cross_artifact_edges,
        "by_artifact_type": dict(by_artifact_type),
        "by_atom_type": dict(by_atom_type),
        "by_authority_class": dict(by_authority),
        "by_edge_type": dict(by_edge_type),
        "by_entity_type": dict(by_entity_type),
        "parse_outcomes": dict(parse_outcomes_counter),
        "degraded_files": degraded_files,
    }


def _build_indexes(
    *,
    atoms: list[EvidenceAtom],
    entities: list[EntityRecord] | None = None,
    edges: list[EvidenceEdge] | None = None,
) -> dict[str, dict[str, list[str]]]:
    entities = entities or []
    edges = edges or []
    by_section: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    by_authority: dict[str, list[str]] = defaultdict(list)
    by_artifact: dict[str, list[str]] = defaultdict(list)
    by_entity_key: dict[str, list[str]] = defaultdict(list)
    for atom in atoms:
        section_key = " > ".join(_atom_section_path(atom)) or "(root)"
        by_section[section_key].append(atom.id)
        by_type[atom.atom_type.value].append(atom.id)
        by_authority[atom.authority_class.value].append(atom.id)
        by_artifact[atom.artifact_id].append(atom.id)
        for key in atom.entity_keys:
            by_entity_key[key].append(atom.id)
    edges_by_atom: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        edges_by_atom[edge.from_atom_id].append(edge.id)
        edges_by_atom[edge.to_atom_id].append(edge.id)
    entities_by_key: dict[str, str] = {entity.canonical_key: entity.id for entity in entities}
    return {
        "atoms_by_section_path": {k: sorted(v) for k, v in sorted(by_section.items())},
        "atoms_by_atom_type": {k: sorted(v) for k, v in sorted(by_type.items())},
        "atoms_by_authority": {k: sorted(v) for k, v in sorted(by_authority.items())},
        "atoms_by_artifact": {k: sorted(v) for k, v in sorted(by_artifact.items())},
        "atoms_by_entity_key": {k: sorted(v) for k, v in sorted(by_entity_key.items())},
        "edges_by_atom": {k: sorted(v) for k, v in sorted(edges_by_atom.items())},
        "entity_id_by_canonical_key": dict(sorted(entities_by_key.items())),
    }


def _compact_atom(atom: EvidenceAtom) -> dict[str, Any]:
    primary_ref = atom.source_refs[0] if atom.source_refs else None
    return {
        "id": atom.id,
        "artifact_id": atom.artifact_id,
        "atom_type": atom.atom_type.value,
        "authority_class": atom.authority_class.value,
        "confidence": atom.confidence,
        "text": atom.raw_text,
        "section_path": _atom_section_path(atom),
        "locator": dict(primary_ref.locator) if primary_ref is not None else {},
        "verified": _atom_verification_state(atom),
        # A5 cross-doc reconciliation needs entity_keys + structured
        # values on every atom so consumers can group atoms touching
        # the same logical entity (e.g. total_contract_value) and
        # flag value contradictions across documents. Previously the
        # compact projection dropped both, forcing PM_HANDOFF to
        # regex over raw_text. Same data unlocks B2 (risk register),
        # B6 (per-site pricing rollup), etc.
        "entity_keys": list(atom.entity_keys),
        "structured": dict(atom.value) if atom.value else {},
    }


def _atom_verification_state(atom: EvidenceAtom) -> str:
    if not atom.receipts:
        return "unverified"
    statuses = {r.replay_status for r in atom.receipts}
    if "failed" in statuses:
        return "failed"
    if statuses == {"verified"}:
        return "verified"
    if "verified" in statuses:
        return "partial"
    return "unsupported"


def _compact_entity(
    entity: EntityRecord,
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    all_atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    """Add ``artifact_ids`` provenance so consumers can see which files
    mention this entity."""
    artifact_ids: set[str] = set()
    atom_ids = set(entity.source_atom_ids)
    for atom in all_atoms:
        if atom.id in atom_ids:
            artifact_ids.add(atom.artifact_id)
    return {
        "id": entity.id,
        "entity_type": entity.entity_type,
        "canonical_key": entity.canonical_key,
        "canonical_name": entity.canonical_name,
        "aliases": list(entity.aliases),
        "artifact_ids": sorted(artifact_ids),
        "source_atom_ids": list(entity.source_atom_ids),
        "review_status": entity.review_status.value,
        "confidence": entity.confidence,
    }


def _compact_edge(edge: EvidenceEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "edge_type": edge.edge_type.value,
        "from_atom_id": edge.from_atom_id,
        "to_atom_id": edge.to_atom_id,
        "reason": edge.reason,
        "confidence": edge.confidence,
        "cross_artifact": bool(edge.metadata.get("cross_artifact")),
        "metadata": dict(edge.metadata or {}),
    }


def _compact_packet(packet: EvidencePacket) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": packet.id,
        "family": packet.family.value,
        "anchor_type": packet.anchor_type,
        "anchor_key": packet.anchor_key,
        "status": packet.status.value,
        "confidence": packet.confidence,
        "governing_atom_ids": list(packet.governing_atom_ids),
        "supporting_atom_ids": list(packet.supporting_atom_ids),
        "contradicting_atom_ids": list(packet.contradicting_atom_ids),
        "reason": packet.reason,
    }
    # Preserve the PacketCertificate so downstream consumers
    # (SOWSmith.scope_clause, OrbitBrief.scope_truth, RunbookGen.site_steps,
    # AtlasDispatch.site_readiness, VisionQC.photo_requirements) see the
    # cert's blast_radius declaration through the envelope.
    cert = getattr(packet, "certificate", None)
    if cert is not None:
        try:
            out["certificate"] = cert.model_dump()
        except Exception:  # pragma: no cover
            try:
                out["certificate"] = dict(cert)
            except Exception:
                out["certificate"] = None
        if out.get("certificate") and isinstance(out["certificate"], dict):
            br = out["certificate"].get("blast_radius") or []
            if br:
                out["blast_radius"] = list(br)
    return out


def _render_generic_structured_md(structured: dict[str, Any]) -> str:
    """Render a non-PDF projection (the lighter ``atom_projection.v1``)."""
    lines: list[str] = []
    document = structured.get("document") or {}
    title = document.get("title")
    if title:
        lines.append(f"### {title}")
        lines.append("")
    for page in structured.get("pages", []) or []:
        for section in page.get("sections", []) or []:
            heading = section.get("heading") or "(uncategorized)"
            sec_id = section.get("id")
            anchor = f'  <a id="{sec_id}"></a>' if sec_id else ""
            lines.append(f"#### {heading}{anchor}")
            lines.append("")
            for block in section.get("blocks", []) or []:
                block_id = block.get("id")
                if block_id:
                    lines.append(f'<a id="{block_id}"></a>')
                text = (block.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "ENVELOPE_FILENAME",
    "ENVELOPE_MARKDOWN_FILENAME",
    "build_orbitbrief_envelope",
    "write_orbitbrief_envelope",
    "envelope_to_markdown",
]
