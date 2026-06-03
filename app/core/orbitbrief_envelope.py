"""OrbitBrief project envelope — the "perfect compressible" LLM input.

DEPRECATED LOCATION — this module has moved to Orbitbrief-Core at
``orbitbrief_core.envelope``. The copy in parser-os remains for
back-compat with existing callers (app/cli.py, app/core/production_report.py,
several scripts, several tests) and stays functionally identical.

The brief-layer surfaces (pm_dashboard, scope_truth, project_vitals,
etc.) have always been Orbitbrief concerns; parser-os will keep this
shim during a deprecation window, then drop it. New code should
import from ``orbitbrief_core.envelope`` directly.

See: https://github.com/Purtera-IT/Orbitbrief-Core
PR:  feat/envelope-migration-from-parser-os



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
    build_bill_of_materials,
    build_change_order_timeline,
    build_deal_financials,
    build_deal_header,
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
PARSER_MANIFEST_SIDECAR = ".parser_manifest.json"


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
    crm = _load_manifest_crm(project_dir)
    if crm:
        summary["crm"] = crm
    # v57.3.5: filter site:* entities + redirect ghost atom keys
    # BEFORE building the indexes — because orbitbrief-core's cluster
    # builder reads from ``envelope.indexes.atoms_by_entity_key`` to
    # decide how many atoms each canonical site has. If we filter
    # entities AFTER the index is built, the index still maps ghost
    # keys to lots of atom_ids while the canonical keys only have the
    # 1 physical_site atom each — so canonical clusters fail the >2
    # atoms gate in orbitbrief-core and get dropped from the dossier.
    # Move filter ABOVE _build_indexes so the redirect propagates.
    entities = _filter_site_entities_against_physical_atoms(entities, atoms)
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

    # v49 FIX 6: enrich site_readiness rows with structured attributes
    # (address, mdf_idf, access_window, escort, users, rooms, notes,
    # aliases) from physical_site atoms. These come from the v48 site
    # roster extractor and v49 docx schema registry and are the single
    # source of truth for site metadata.
    try:
        import re as _re_v49
        def _atom_type_str(_a) -> str:
            _at = getattr(_a, "atom_type", None)
            return _at.value if hasattr(_at, "value") else str(_at or "")
        _sr = envelope.get("site_readiness") or {}
        _sites_list = _sr.get("sites") or []
        # site_readiness.sites is a LIST of dicts keyed by "site" field
        _by_slug: dict[str, dict] = {}
        for _entry in _sites_list:
            if isinstance(_entry, dict):
                _k = _entry.get("site") or _entry.get("site_key") or ""
                if _k:
                    _by_slug[_k] = _entry
        for _atom in atoms:
            if _atom_type_str(_atom) != "physical_site":
                continue
            _val = getattr(_atom, "value", None) or {}
            if not isinstance(_val, dict):
                continue
            _sid = _val.get("id") or _val.get("site_id") or ""
            if not _sid:
                continue
            _slug = f"site:{_re_v49.sub(r'[^a-z0-9]+', '_', _sid.lower()).strip('_')}"
            _entry = _by_slug.get(_slug)
            if _entry is None:
                continue
            for _attr in ("address", "mdf_idf", "access_window", "escort", "users", "rooms", "notes",
                          "facility_name", "street_address", "escort_owner", "contact", "phone", "email"):
                _v = _val.get(_attr)
                if _v and not _entry.get(_attr):
                    _entry[_attr] = _v
            _names = _val.get("names") or []
            if _names:
                _aliases = _entry.setdefault("aliases", [])
                for _n in _names:
                    if _n and _n not in _aliases:
                        _aliases.append(_n)
    except Exception as _v49_exc:
        import logging as _lg_v49
        _lg_v49.getLogger(__name__).warning("v49 site attribute passthrough failed: %s", _v49_exc)

    envelope["stakeholder_load"] = build_stakeholder_load(atoms=atoms)

    # Deal header / financials / BOM — PM-facing assembly of the
    # structured commercial atoms the xlsx parser emits. Each is omitted
    # when the deal carries no such data, so the envelope shape stays
    # stable for non-commercial projects. Never fatal.
    try:
        _deal_header = build_deal_header(atoms=atoms)
        if _deal_header.get("present"):
            envelope["deal_header"] = _deal_header
        _deal_financials = build_deal_financials(atoms=atoms)
        if _deal_financials.get("present"):
            envelope["deal_financials"] = _deal_financials
        _bom = build_bill_of_materials(atoms=atoms)
        if _bom.get("present"):
            envelope["bill_of_materials"] = _bom
    except Exception as _deal_exc:
        import logging as _lg_deal
        _lg_deal.getLogger(__name__).warning("deal section build failed: %s", _deal_exc)

    # Gap F — Truth Gate: grade every entity by independent-source
    # corroboration so single-sourced facts are visibly distinct from
    # facts three documents agree on. Deterministic, never fatal.
    try:
        from app.core.truth_gate import build_truth_gate
        envelope["truth_gate"] = build_truth_gate(
            atoms=atoms, entities=entities, edges=edges,
        )
    except Exception as _tg_exc:
        import logging as _lg_tg
        _lg_tg.getLogger(__name__).warning("truth_gate build failed: %s", _tg_exc)
        envelope["truth_gate"] = {}

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
    # CRM context (when the parser-manifest sidecar carries it) is
    # exposed at the top of the envelope so downstream consumers can
    # render deal name / opportunity ID / amount without re-reading
    # the manifest blob.
    if crm:
        envelope["crm"] = crm
    return envelope


def _load_manifest_crm(project_dir: Path) -> dict[str, Any] | None:
    """Read ``context.crm`` from the parser manifest sidecar when present."""
    path = Path(project_dir) / PARSER_MANIFEST_SIDECAR
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ctx = manifest.get("context")
        if not isinstance(ctx, dict):
            return None
        crm = ctx.get("crm")
        return dict(crm) if isinstance(crm, dict) else None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def write_orbitbrief_envelope(
    *,
    project_dir: Path,
    envelope: dict[str, Any],
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Write the envelope JSON, markdown, and (if SowSmith is installed) the SOW.

    Returns ``(json_path, markdown_path, sow_path_or_None)``. Defaults
    to writing under ``<project_dir>/.orbitbrief/``. Pass ``out_dir``
    to override.

    The ``sow.md`` file is rendered by the standalone ``sowsmith``
    package (https://github.com/Purtera-IT/SowSmith) if it's
    installed. If SowSmith isn't on the path, ``sow_path`` is
    returned as ``None`` and only the envelope JSON + markdown are
    written. This keeps parser-os usable with or without the
    downstream SOW generator on the same machine.

    Install SowSmith to enable in-process SOW rendering::

        pip install -e path/to/SowSmith

    Or render after the fact::

        sowsmith render <project>/.orbitbrief/orbitbrief.input.json
    """
    out_dir = Path(out_dir) if out_dir is not None else (Path(project_dir) / ".orbitbrief")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ENVELOPE_FILENAME
    md_path = out_dir / ENVELOPE_MARKDOWN_FILENAME
    json_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(envelope_to_markdown(envelope), encoding="utf-8")

    sow_path: Path | None = None
    try:
        from sowsmith import build_sow_markdown  # type: ignore[import-not-found]
    except ImportError:
        build_sow_markdown = None  # type: ignore[assignment]
    if build_sow_markdown is not None:
        sow_path = out_dir / "sow.md"
        sow_path.write_text(build_sow_markdown(envelope), encoding="utf-8")
    return json_path, md_path, sow_path


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

    # OrbitBrief-Core cockpit surfaces — rendered as markdown sections
    # so an LLM consuming the markdown form sees the pre-aggregated
    # signals (not just raw atoms / packets).
    lines.extend(_render_cockpit_surfaces_md(envelope))

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


def _render_cockpit_surfaces_md(envelope: dict[str, Any]) -> list[str]:
    """Render the OrbitBrief-Core cockpit fields as markdown sections.

    Surfaces rendered:
      * project_vitals          — header score with component breakdown
      * pm_dashboard            — blockers / contradictions / open Qs
      * sow_readiness_scorecard — dimension table + grade
      * srl_missing_checklist   — coverage + missing field list
      * scope_truth             — canonical scope table + contested
      * change_order_timeline   — chronological change order audit
      * site_readiness          — per-site rollup table
      * stakeholder_load        — workload + bottleneck matrix

    Missing surfaces are silently skipped — older compile results
    without OrbitBrief-Core fields still render their atoms section.
    """
    lines: list[str] = []

    vitals = envelope.get("project_vitals") or {}
    if vitals:
        lines.append("---")
        lines.append("")
        lines.append("## Project Vitals")
        lines.append("")
        lines.append(
            f"**Score:** {vitals.get('score_100', '—')} / 100  ·  "
            f"**Band:** `{vitals.get('band', '—')}`  ·  "
            f"**Top drivers:** {', '.join(vitals.get('top_drivers') or []) or '—'}  ·  "
            f"**Top detractors:** {', '.join(vitals.get('top_detractors') or []) or '—'}"
        )
        lines.append("")
        components = vitals.get("components") or []
        if components:
            lines.append("| Component | Weight | Score | Contribution |")
            lines.append("|---|---|---|---|")
            for c in components:
                lines.append(
                    f"| {c.get('name', '—')} | {c.get('weight', 0):.2f} | "
                    f"{c.get('raw_score', 0):.2f} | {c.get('contribution', 0):.3f} |"
                )
            lines.append("")

    dash = envelope.get("pm_dashboard") or {}
    if dash:
        lines.append("---")
        lines.append("")
        lines.append("## PM Dashboard")
        lines.append("")
        bl = dash.get("blockers") or []
        if bl:
            lines.append(f"**Blockers ({len(bl)})**")
            for b in bl[:15]:
                lines.append(f"- [{b.get('kind', '—')}] {(b.get('summary', '') or '')[:200]}")
            lines.append("")
        cd = dash.get("cross_doc_contradictions") or []
        if cd:
            lines.append(f"**Cross-doc contradictions ({len(cd)})**")
            for c in cd[:10]:
                lines.append(f"- {(c.get('reason') or '')[:200]}")
            lines.append("")
        co = dash.get("change_orders") or []
        if co:
            lines.append(f"**Change orders ({len(co)})**")
            for c in co[:10]:
                delta = c.get("change_delta")
                delta_str = f" ({delta.get('from')}→{delta.get('to')}, Δ{delta.get('delta'):+d})" if delta else ""
                lines.append(f"- {(c.get('text') or '')[:200]}{delta_str}")
            lines.append("")
        oq = dash.get("open_questions") or []
        if oq:
            lines.append(f"**Open questions ({len(oq)})**")
            for q in oq[:10]:
                lines.append(f"- {(q.get('text') or '')[:200]}")
            lines.append("")
        sla = dash.get("sla_summary") or []
        if sla:
            lines.append(f"**SLA targets ({len(sla)})**")
            for s in sla[:10]:
                targets = s.get("sla") or {}
                target_str = ", ".join(f"{k}={v}" for k, v in targets.items())
                lines.append(f"- {target_str}  ({(s.get('text') or '')[:80]})")
            lines.append("")
        money = dash.get("money_summary") or {}
        if money.get("total"):
            lines.append(f"**Commercial total:** ${money['total']:,.2f} ({len(money.get('atoms', []))} atoms)")
            lines.append("")

    sc = envelope.get("sow_readiness_scorecard") or {}
    if sc:
        lines.append("---")
        lines.append("")
        lines.append("## SOW Readiness Scorecard")
        lines.append("")
        lines.append(
            f"**Overall:** {sc.get('readiness_score', 0):.2f} / 1.00  ·  "
            f"**Grade:** `{sc.get('grade', '—')}`"
        )
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|---|---|")
        for dim, d in (sc.get("dimensions") or {}).items():
            lines.append(f"| {dim} | {d.get('score', 0):.2f} |")
        lines.append("")

    ck = envelope.get("srl_missing_checklist") or {}
    if ck:
        lines.append("---")
        lines.append("")
        lines.append("## SRL Coverage")
        lines.append("")
        lines.append(
            f"**Coverage:** {ck.get('present_count', 0)} / {ck.get('field_count', 0)} fields "
            f"({(ck.get('coverage', 0) or 0) * 100:.0f}%)"
        )
        lines.append("")
        by_cat = ck.get("by_category") or {}
        if by_cat:
            lines.append("| Category | Present / Total | Coverage |")
            lines.append("|---|---|---|")
            for cat, stats in sorted(by_cat.items()):
                lines.append(
                    f"| {cat} | {stats.get('present', 0)} / {stats.get('total', 0)} | "
                    f"{(stats.get('coverage', 0) or 0) * 100:.0f}% |"
                )
            lines.append("")
        missing = ck.get("missing") or []
        if missing:
            lines.append(f"**Missing fields ({len(missing)})**")
            for m in missing:
                lines.append(f"- `{m.get('field_id')}` — {m.get('label')}")
            lines.append("")

    st = envelope.get("scope_truth") or {}
    if st.get("devices"):
        lines.append("---")
        lines.append("")
        lines.append("## Scope Truth")
        lines.append("")
        lines.append(
            f"**{st.get('device_count', 0)}** devices across **{st.get('site_count', 0)}** sites  ·  "
            f"**{st.get('contested_count', 0)}** contested"
        )
        lines.append("")
        lines.append("| Device | Site | Quantity | Governing | Status |")
        lines.append("|---|---|---|---|---|")
        for d in st["devices"]:
            status = "⚠ contested" if d.get("is_contested") else "✓"
            lines.append(
                f"| {d.get('device', '—')} | {d.get('site', '—')} | "
                f"**{d.get('canonical_quantity', '—')}** | "
                f"`{d.get('governing_authority', '—')}` | {status} |"
            )
        lines.append("")

    ct = envelope.get("change_order_timeline") or {}
    if ct.get("entries"):
        lines.append("---")
        lines.append("")
        lines.append(f"## Change Order Timeline ({ct.get('entry_count', 0)} entries)")
        lines.append("")
        lines.append("| Kind | Delta | Approved | Text |")
        lines.append("|---|---|---|---|")
        for e in ct["entries"][:20]:
            delta = e.get("change_delta") or {}
            if delta:
                delta_str = f"{delta.get('from')}→{delta.get('to')} ({delta.get('delta'):+d})"
            else:
                delta_str = "—"
            approval = "✓" if e.get("approval_signal") else "—"
            text = (e.get("text") or "").replace("|", "\\|").replace("\n", " ")[:160]
            lines.append(f"| {e.get('kind', '—')} | {delta_str} | {approval} | {text} |")
        lines.append("")

    sr = envelope.get("site_readiness") or {}
    if sr.get("sites"):
        lines.append("---")
        lines.append("")
        lines.append(
            f"## Site Readiness ({sr.get('site_count', 0)} sites, avg {sr.get('avg_readiness', 0):.2f})"
        )
        lines.append("")
        lines.append("| Site | Readiness | Devices | Stakeholders | Constraints | Contradictions |")
        lines.append("|---|---|---|---|---|---|")
        for s in sr["sites"]:
            lines.append(
                f"| `{s.get('site', '—')}` | {s.get('readiness', 0):.2f} | "
                f"{s.get('device_count', 0)} | {s.get('stakeholder_count', 0)} | "
                f"{s.get('constraint_count', 0)} | {s.get('contradiction_count', 0)} |"
            )
        lines.append("")

    sl = envelope.get("stakeholder_load") or {}
    if sl.get("stakeholders"):
        lines.append("---")
        lines.append("")
        lines.append(f"## Stakeholder Load ({sl.get('stakeholder_count', 0)} stakeholders)")
        lines.append("")
        if sl.get("bottlenecks"):
            lines.append(f"⚠ **Bottlenecks:** {', '.join(sl['bottlenecks'])}")
            lines.append("")
        lines.append("| Stakeholder | Risks | Critical | High | Actions | Severity Load |")
        lines.append("|---|---|---|---|---|---|")
        for s in sl["stakeholders"]:
            lines.append(
                f"| {s.get('slug', '—')} | {s.get('risk_count', 0)} | "
                f"{s.get('critical_risk_count', 0)} | {s.get('high_risk_count', 0)} | "
                f"{s.get('action_item_count', 0)} | {s.get('risk_severity_load', 0)} |"
            )
        lines.append("")

    return lines


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
    by_stakeholder: dict[str, list[str]] = defaultdict(list)
    by_device: dict[str, list[str]] = defaultdict(list)
    by_site: dict[str, list[str]] = defaultdict(list)
    for atom in atoms:
        section_key = " > ".join(_atom_section_path(atom)) or "(root)"
        by_section[section_key].append(atom.id)
        by_type[atom.atom_type.value].append(atom.id)
        by_authority[atom.authority_class.value].append(atom.id)
        by_artifact[atom.artifact_id].append(atom.id)
        for key in atom.entity_keys:
            by_entity_key[key].append(atom.id)
            # Per-entity-prefix specialized indexes: O(1) lookup of
            # "every fact about this stakeholder / device / site"
            # without re-scanning atoms_by_entity_key. Downstream
            # consumers (SOWSmith.scope_clause, PM cockpit) hit these
            # constantly.
            if key.startswith("stakeholder:"):
                by_stakeholder[key[len("stakeholder:"):]].append(atom.id)
            elif key.startswith("device:"):
                by_device[key[len("device:"):]].append(atom.id)
            elif key.startswith("site:"):
                by_site[key[len("site:"):]].append(atom.id)
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
        "atoms_by_stakeholder_slug": {k: sorted(v) for k, v in sorted(by_stakeholder.items())},
        "atoms_by_device_slug": {k: sorted(v) for k, v in sorted(by_device.items())},
        "atoms_by_site_slug": {k: sorted(v) for k, v in sorted(by_site.items())},
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


def _filter_site_entities_against_physical_atoms(
    entities: list[EntityRecord],
    atoms: list[EvidenceAtom],
) -> list[EntityRecord]:
    """v57.3 — drop ``site:*`` entity records that don't trace to a
    physical_site atom; v57.3.1 — also REWRITE atom.entity_keys so
    atoms previously tagged with a ghost site key get reassigned to
    the canonical site they actually describe.

    Why: ``orbitbrief-core/world_model/site_reality/cluster.py`` walks
    every ``site:*`` entity in the envelope and builds one cluster per
    entity. The dossier renders one row per cluster. Without this
    filter, every LLM-extracted ghost (``atlanta_west_office``,
    ``optbot_atlanta_office``, ``atl_hq_2026``, ``site:site``,
    ``atlanta_headquarters_innovation_tower``, ...) becomes a dossier
    site even though the canonical roster only has the 5 ATL-XX-XX rows.

    The rule: a ``site:*`` entity is real iff its canonical_key,
    canonical_name, or any alias matches the slugified site_id, name,
    or facility_name of a physical_site atom. Everything else is a
    ghost LLM cluster.

    v57.3.1 follow-up: when we DROP a ghost site:* entity, atoms that
    previously had ``entity_keys=[..., site:ghost_name]`` get orphaned
    (their cluster disappears) — that's why the OPTBOT dossier showed
    ATL-WEST-02 missing. Fix: for each dropped ghost, find the best
    canonical match by token overlap against the physical_site facility
    names, then walk all atoms and rewrite their entity_keys so
    ``site:atlanta_west_office`` becomes ``site:atl_west_02``. The
    canonical cluster then absorbs those atoms and passes
    orbitbrief-core's >2-evidence gate.

    Non-``site:*`` entities (vendor, device, money, etc.) pass through
    untouched.
    """
    import re as _re

    def _slug(s: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

    def _tokens(slug: str) -> set[str]:
        return {t for t in slug.split("_") if t and len(t) >= 2}

    # ── 1. Build canonical site catalog from physical_site atoms.
    # ``canonical_site_id_slugs`` = the *strict* truth (only the
    # ``slug(site_id)`` of each canonical roster atom). An entity is
    # accepted as canonical ONLY when its key matches one of these
    # exactly. Everything else — even legitimate aliases like
    # ``site:atl_047`` (truncation of ``atl_047_04``) — is treated as
    # ghost and routed through token-overlap redirect so atoms collapse
    # into the canonical key. This guarantees the dossier sees one
    # cluster per real roster row.
    canonical_site_id_slugs: set[str] = set()
    canonical_to_token_sets: dict[str, list[set[str]]] = {}
    # site_id slug → primary facility_name for entity injection.
    primary_to_facility_name: dict[str, str] = {}

    for a in atoms:
        atype = getattr(a, "atom_type", None)
        atype_s = atype.value if hasattr(atype, "value") else str(atype or "")
        if atype_s != "physical_site":
            continue
        val = getattr(a, "value", None) or {}
        if not isinstance(val, dict):
            continue
        sid = val.get("site_id") or val.get("id") or ""
        sid_slug = _slug(str(sid))
        if not sid_slug:
            continue
        primary = sid_slug
        canonical_site_id_slugs.add(primary)
        canonical_to_token_sets.setdefault(primary, [])
        facility = val.get("facility_name") or val.get("name") or sid
        if isinstance(facility, str) and facility.strip():
            primary_to_facility_name.setdefault(primary, facility.strip())
        identity_strings: list[str] = []
        for field in ("site_id", "id", "name", "facility_name", "address", "street_address"):
            v = val.get(field)
            if isinstance(v, str) and v.strip():
                identity_strings.append(v)
        names_field = val.get("names") or val.get("aliases") or ()
        if isinstance(names_field, (list, tuple)):
            for n in names_field:
                if isinstance(n, str) and n.strip():
                    identity_strings.append(n)
        for s in identity_strings:
            slug = _slug(s)
            if slug:
                canonical_to_token_sets[primary].append(_tokens(slug))

    if not canonical_site_id_slugs:
        return entities

    # ── 2. Pass 1 — classify each entity as canonical-or-ghost. For
    # ghosts, also pick the best canonical to redirect them to.
    kept: list[EntityRecord] = []
    ghost_to_canonical: dict[str, str] = {}  # ghost site_key → canonical site_key

    def _best_canonical_for(candidate_slugs: set[str]) -> str | None:
        """Return the primary canonical slug whose token sets best
        overlap with any candidate slug, or None if no confident match.

        Scoring is two-tier: ``(total_overlap, site_id_overlap)``.
        Total overlap counts any identity-string token shared; site_id
        overlap counts only tokens shared with the canonical's site_id
        slug itself (the distinguishing piece).

        Acceptance rule: REQUIRE at least one shared site_id token. This
        catches ``atlanta_west_office`` -> ``atl_west_02`` (shares ``west``
        with the site_id ``atl_west_02``) but rejects ``headquarters`` /
        ``site:site`` (no site_id contains ``headquarters`` or ``site``).
        Pure facility-name matches without a site_id token are too risky
        — ``office`` alone is in every canonical and would over-collapse.
        """
        if not candidate_slugs:
            return None
        cand_tokens: set[str] = set()
        for cs in candidate_slugs:
            cand_tokens |= _tokens(cs)
        if not cand_tokens:
            return None
        best_primary: str | None = None
        best_score: tuple[int, int] = (0, 0)
        for primary, token_sets in canonical_to_token_sets.items():
            site_id_tokens = _tokens(primary)
            # Exclude tokens that are pure digits or 1-char — they're
            # too generic to be discriminative (``01``, ``02``, ``a``).
            site_id_tokens = {t for t in site_id_tokens if len(t) >= 2 and not t.isdigit()}
            site_id_overlap = len(cand_tokens & site_id_tokens)
            if site_id_overlap < 1:
                continue
            for ts in token_sets:
                total_overlap = len(cand_tokens & ts)
                score = (total_overlap, site_id_overlap)
                if score > best_score:
                    best_score = score
                    best_primary = primary
        return best_primary

    canonical_seen_in_entities: set[str] = set()
    for ent in entities:
        ck = getattr(ent, "canonical_key", "") or ""
        if not ck.startswith("site:"):
            kept.append(ent)
            continue
        ck_slug = ck[len("site:"):]
        # STRICT canonical: entity is real iff its slug == one of the
        # canonical site_id slugs exactly. Everything else (truncated
        # aliases, LLM names, year-suffix hallucinations) is a ghost and
        # gets routed through the token-overlap redirect below.
        if ck_slug in canonical_site_id_slugs:
            kept.append(ent)
            canonical_seen_in_entities.add(ck_slug)
            continue
        # Ghost — try to redirect to a canonical via token overlap.
        candidate_slugs: set[str] = {ck_slug}
        cname = getattr(ent, "canonical_name", "") or ""
        if cname:
            candidate_slugs.add(_slug(cname))
        for alias in (getattr(ent, "aliases", None) or ()):
            if isinstance(alias, str) and alias:
                candidate_slugs.add(_slug(alias))
        candidate_slugs.discard("")
        best_primary = _best_canonical_for(candidate_slugs)
        if best_primary:
            ghost_to_canonical[ck] = f"site:{best_primary}"

    # ── 3. Pass 2 — rewrite atom.entity_keys: replace each ghost site
    # key with its canonical-mapped site key. This redirects orphaned
    # atoms into the canonical cluster so the >2-evidence gate in
    # orbitbrief-core promotes them.
    if ghost_to_canonical:
        for atom in atoms:
            keys = getattr(atom, "entity_keys", None)
            if not keys:
                continue
            try:
                new_keys: list[str] = []
                changed = False
                seen: set[str] = set()
                for k in keys:
                    if isinstance(k, str) and k.startswith("site:") and k in ghost_to_canonical:
                        canon = ghost_to_canonical[k]
                        changed = True
                        if canon not in seen:
                            new_keys.append(canon)
                            seen.add(canon)
                    else:
                        if isinstance(k, str) and k not in seen:
                            new_keys.append(k)
                            seen.add(k)
                if changed:
                    try:
                        atom.entity_keys = new_keys
                    except (AttributeError, TypeError):
                        pass
            except TypeError:
                continue

    # ── 4. Inject canonical site entities for any physical_site atom
    # whose canonical key isn't already represented. Without this, the
    # ``site:atl_047_04`` key has atoms tagged to it (from the
    # physical_site atom + redirected ghosts) but no entity record →
    # orbitbrief-core's cluster builder never seeds a cluster for it
    # and the dossier shows the alias name instead of the facility name.
    # We inject a minimal EntityRecord so the cluster gets built with
    # the canonical key + facility-name display.
    missing = canonical_site_id_slugs - canonical_seen_in_entities
    if missing:
        try:
            from app.core.schemas import EntityRecord, ReviewStatus  # local import to avoid cycles
            import uuid as _uuid
            for slug in sorted(missing):
                # Skip if we couldn't extract a facility name (shouldn't
                # happen for real physical_site atoms but defensive).
                facility = primary_to_facility_name.get(slug, "")
                if not facility:
                    continue
                # Find one source atom_id to anchor provenance.
                anchor_atom_id = ""
                for a in atoms:
                    atype = getattr(a, "atom_type", None)
                    atype_s = atype.value if hasattr(atype, "value") else str(atype or "")
                    if atype_s != "physical_site":
                        continue
                    val = getattr(a, "value", None) or {}
                    if not isinstance(val, dict):
                        continue
                    sid = val.get("site_id") or val.get("id") or ""
                    if _slug(str(sid)) == slug:
                        anchor_atom_id = getattr(a, "id", "") or ""
                        break
                # Find the original project_id from any atom (all share one).
                proj_id = ""
                for a in atoms:
                    pid = getattr(a, "project_id", None)
                    if pid:
                        proj_id = str(pid)
                        break
                injected = EntityRecord(
                    id=f"ent_canon_{slug}_{_uuid.uuid4().hex[:8]}",
                    project_id=proj_id,
                    entity_type="site",
                    canonical_key=f"site:{slug}",
                    canonical_name=facility,
                    aliases=[],
                    source_atom_ids=[anchor_atom_id] if anchor_atom_id else [],
                    confidence=0.99,
                    review_status=ReviewStatus.auto_accepted,
                )
                kept.append(injected)
        except Exception:
            # If injection fails for any reason, fall through silently
            # — the dossier might miss a cluster but at least won't crash.
            pass

    return kept


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
