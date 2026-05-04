"""Review folder writer — per-compile human-review dossier.

Pass ``--review-out PATH`` to ``app.cli compile`` and Parser OS will drop a
self-contained review folder beside the compile result.  The intent is to
make every run *teach* the operator something:

* **Original artifacts** are copied verbatim so the reviewer can open the
  source side-by-side with our parsed view.
* **Per-artifact dossier** (``REVIEW.md`` plus a JSON sidecar) shows the
  routing decision, every atom we extracted, the receipt status, and a
  side-by-side of raw text vs normalized claim.
* **Ontology gaps** lists candidate aliases / patterns the active pack is
  missing — the file is a checklist a reviewer can tick to grow the pack.
* **Cross-artifact graph** edges and contradictions get flattened into a
  scannable markdown table.
* **Top-level REVIEW.md** is the entrypoint: a checklist that walks the
  reviewer through (1) any errors, (2) ontology gaps, (3) each artifact,
  (4) every needs_review packet, (5) cross-artifact contradictions.

The folder is opt-in (no flag = nothing written) and is laid out so it can
be committed (or zipped) and shared with a human reviewer who has never
opened the project before.

Folder shape (one per compile_id)::

    <out>/
      REVIEW.md                  # entrypoint walkthrough + checkboxes
      manifest.json              # the CompileManifest verbatim
      trace.json                 # CompileTrace verbatim (when present)
      warnings.md                # categorized warnings (errors, then warnings)
      ontology_gaps.md           # gap detector output, human-checklist style
      ontology_gaps.json         # same data, machine-readable
      graph/
        edges.md                 # all cross-artifact edges, sortable
        contradictions.md        # contradicts edges + the packets they fed
      packets/
        REVIEW.md                # per-packet dossier with citations
      artifacts/
        <stem>/
          original.<ext>         # exact copy of the input file
          REVIEW.md              # routing + atoms + receipts side-by-side
          atoms.json             # all atoms from this artifact (full schema)
          routing.json           # parser decision + confidence + reasons
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.ontology_gaps import detect_ontology_gaps, render_gaps_markdown
from app.core.schemas import (
    CompileResult,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    PacketStatus,
)
from app.domain.schemas import DomainPack


def write_review_folder(
    *,
    project_dir: Path,
    compile_result: CompileResult,
    out_dir: Path,
    pack: DomainPack,
    artifact_paths: dict[str, Path],
) -> Path:
    """Write the full review folder for a single compile and return its root path.

    ``out_dir`` is the parent — the actual folder created is
    ``<out_dir>/<compile_id>/`` so multiple runs accumulate side-by-side.
    """
    out_dir = Path(out_dir).resolve()
    compile_id = compile_result.compile_id or "unknown_compile"
    root = out_dir / compile_id
    root.mkdir(parents=True, exist_ok=True)

    atoms_by_artifact: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in compile_result.atoms:
        atoms_by_artifact[atom.artifact_id].append(atom)

    edges_by_atom: dict[str, list[EvidenceEdge]] = defaultdict(list)
    for edge in compile_result.edges:
        edges_by_atom[edge.from_atom_id].append(edge)
        edges_by_atom[edge.to_atom_id].append(edge)

    # Manifest + trace sidecars (verbatim copy) so the reviewer doesn't have
    # to dig back into the original compile.json.
    if compile_result.manifest is not None:
        (root / "manifest.json").write_text(
            compile_result.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if compile_result.trace is not None:
        (root / "trace.json").write_text(
            compile_result.trace.model_dump_json(indent=2),
            encoding="utf-8",
        )

    _write_warnings_markdown(root, compile_result)
    gap_report = detect_ontology_gaps(atoms=list(compile_result.atoms), pack=pack)
    (root / "ontology_gaps.json").write_text(
        json.dumps(gap_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "ontology_gaps.md").write_text(render_gaps_markdown(gap_report), encoding="utf-8")

    # Pack auto-suggest YAML — turns gap candidates into a copy/pasteable
    # YAML snippet operators can drop into the active pack file.  See
    # PRODUCTION_GAPS / Week 5.  We only write the file when there are
    # ≥ 1 candidate; empty pack files are confusing.
    suggest_yaml = _render_pack_suggestions(gap_report, pack)
    if suggest_yaml:
        (root / "pack_suggestions.yaml").write_text(suggest_yaml, encoding="utf-8")

    _write_graph_markdown(root, compile_result)
    _write_packets_markdown(root, compile_result)
    _write_artifact_dossiers(
        root,
        compile_result=compile_result,
        atoms_by_artifact=atoms_by_artifact,
        artifact_paths=artifact_paths,
        edges_by_atom=edges_by_atom,
    )
    _write_top_level_review(
        root,
        project_dir=project_dir,
        compile_result=compile_result,
        atoms_by_artifact=atoms_by_artifact,
        gap_report=gap_report,
    )
    return root


# ── Top-level REVIEW.md walkthrough ──────────────────────────────────────


def _write_top_level_review(
    root: Path,
    *,
    project_dir: Path,
    compile_result: CompileResult,
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    gap_report: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f"# Compile review — `{compile_result.compile_id}`")
    lines.append("")
    lines.append(f"_project: `{project_dir.name}` • generated: "
                 f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_")
    lines.append("")
    lines.append("This folder is the per-run dossier.  Walk it in order:")
    lines.append("")
    lines.append("1. **Errors first** — see `warnings.md` for any `ERROR:` lines.  Don't ship a build with errors.")
    lines.append("2. **Ontology gaps** — `ontology_gaps.md` is a checklist of phrases the pack didn't recognize.  Tick to add to the active pack for next run.")
    lines.append("3. **Per-artifact dossiers** — under `artifacts/<stem>/REVIEW.md`, see what each parser extracted vs the original file.")
    lines.append("4. **Packets that need review** — `packets/REVIEW.md` lists every packet with a checkbox.")
    lines.append("5. **Cross-artifact contradictions** — `graph/contradictions.md` is the highest-stakes review surface.")
    lines.append("")

    # Stats block
    manifest = compile_result.manifest
    atom_count = len(compile_result.atoms)
    packet_count = len(compile_result.packets)
    edge_count = len(compile_result.edges)
    needs_review_packets = sum(1 for p in compile_result.packets if p.status == PacketStatus.needs_review)
    contradiction_packets = sum(1 for p in compile_result.packets if p.contradicting_atom_ids)
    cross_artifact_edges = sum(1 for e in compile_result.edges if (e.metadata or {}).get("cross_artifact"))
    error_count = sum(1 for w in compile_result.warnings if w.startswith("ERROR:"))
    warning_count = sum(1 for w in compile_result.warnings if w.startswith("WARNING:"))

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- artifacts parsed: **{len(atoms_by_artifact)}**")
    lines.append(f"- atoms extracted: **{atom_count}**")
    lines.append(f"- entities resolved: **{len(compile_result.entities)}**")
    lines.append(f"- edges built: **{edge_count}** ({cross_artifact_edges} cross-artifact)")
    lines.append(f"- packets: **{packet_count}** ({needs_review_packets} needs_review, {contradiction_packets} with contradictions)")
    lines.append(f"- validation: **{error_count} errors / {warning_count} warnings** (see `warnings.md`)")
    lines.append(f"- ontology gaps: **{gap_report['summary']['vocab_gap_count']} vocab + "
                 f"{gap_report['summary']['entity_gap_count']} entity** (see `ontology_gaps.md`)")
    if manifest is not None:
        lines.append(f"- input signature: `{manifest.input_signature}`")
        lines.append(f"- output signature: `{manifest.output_signature}`")
        lines.append(f"- domain pack: **{manifest.domain_pack_id}** v{manifest.domain_pack_version}")
        lines.append(f"- cache: {manifest.cache_hits} hit / {manifest.cache_misses} miss")
    lines.append("")

    # Routing decisions
    if manifest is not None and manifest.parser_routing:
        lines.append("## Parser routing")
        lines.append("")
        lines.append("| Artifact | Parser | Confidence | Cache | Reasons |")
        lines.append("|---|---|---|---|---|")
        for row in manifest.parser_routing:
            reasons = ", ".join(row.get("reasons", [])[:4])
            lines.append(
                f"| `{row.get('filename', '')}` | `{row.get('chosen_parser', '')}` "
                f"| {row.get('confidence', 0):.2f} | {'hit' if row.get('cache_hit') else 'miss'} "
                f"| {reasons} |"
            )
        lines.append("")

    # Per-artifact links
    lines.append("## Per-artifact dossiers")
    lines.append("")
    for fp in (manifest.artifact_fingerprints if manifest else []):
        atoms = atoms_by_artifact.get(fp.artifact_id) or []
        lines.append(f"- [ ] `{fp.filename}` → "
                     f"[`artifacts/{Path(fp.filename).stem}/REVIEW.md`](artifacts/{Path(fp.filename).stem}/REVIEW.md) "
                     f"({fp.parser_name}, {len(atoms)} atoms)")
    lines.append("")

    # Quick "things to look at first"
    lines.append("## Top fix-it list (auto-prioritized)")
    lines.append("")
    if error_count:
        lines.append(f"- [ ] resolve {error_count} validation errors in `warnings.md`")
    if needs_review_packets:
        lines.append(f"- [ ] review {needs_review_packets} `needs_review` packets in `packets/REVIEW.md`")
    if contradiction_packets:
        lines.append(f"- [ ] reconcile {contradiction_packets} packets carrying contradictions (`graph/contradictions.md`)")
    if gap_report["summary"]["vocab_gap_count"]:
        lines.append(f"- [ ] decide on {gap_report['summary']['vocab_gap_count']} ontology vocab candidates (`ontology_gaps.md`)")
    if gap_report["summary"]["entity_gap_count"]:
        lines.append(f"- [ ] resolve {gap_report['summary']['entity_gap_count']} unknown-entity buckets so packets stop anchoring on `*:unknown`")
    if not (error_count or needs_review_packets or contradiction_packets):
        lines.append("- looks clean ✓")
    lines.append("")

    (root / "REVIEW.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ── warnings.md ──────────────────────────────────────────────────────────


def _write_warnings_markdown(root: Path, compile_result: CompileResult) -> None:
    errors = sorted(w for w in compile_result.warnings if w.startswith("ERROR:"))
    warnings = sorted(w for w in compile_result.warnings if w.startswith("WARNING:"))
    other = sorted(w for w in compile_result.warnings if not w.startswith(("ERROR:", "WARNING:")))

    # Group warnings by their stable prefix for skim-ability.
    by_topic: dict[str, list[str]] = defaultdict(list)
    for w in warnings:
        head = w.split(":", 1)[1].strip().split(" ", 1)[0] if ":" in w else "other"
        by_topic[head].append(w)

    lines: list[str] = []
    lines.append("# Compile warnings")
    lines.append("")
    lines.append(f"_{len(errors)} errors • {len(warnings)} warnings • {len(other)} info_")
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        for w in errors:
            lines.append(f"- [ ] {w}")
        lines.append("")
    if by_topic:
        lines.append("## Warnings (grouped)")
        lines.append("")
        for topic in sorted(by_topic):
            rows = by_topic[topic]
            lines.append(f"### {topic} ({len(rows)})")
            lines.append("")
            for w in rows[:200]:
                lines.append(f"- {w}")
            lines.append("")
    if other:
        lines.append("## Other")
        lines.append("")
        for w in other:
            lines.append(f"- {w}")
        lines.append("")
    (root / "warnings.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ── graph/edges.md + contradictions.md ────────────────────────────────────


def _write_graph_markdown(root: Path, compile_result: CompileResult) -> None:
    graph_dir = root / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    edges = list(compile_result.edges)
    families = Counter((e.metadata or {}).get("edge_family", "unspecified") for e in edges)

    lines: list[str] = []
    lines.append("# Cross-artifact and intra-artifact edges")
    lines.append("")
    lines.append(f"_{len(edges)} total • by family: " + ", ".join(f"{k}={v}" for k, v in families.most_common()) + "_")
    lines.append("")
    lines.append("| Type | Family | Cross-artifact | Confidence | From | To | Reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in sorted(edges, key=lambda x: (x.edge_type.value, x.id)):
        meta = e.metadata or {}
        lines.append(
            "| {t} | {f} | {ca} | {c:.2f} | `{fa}` | `{ta}` | {r} |".format(
                t=e.edge_type.value,
                f=meta.get("edge_family", "—"),
                ca="yes" if meta.get("cross_artifact") else "no",
                c=e.confidence,
                fa=e.from_atom_id,
                ta=e.to_atom_id,
                r=(e.reason or "").replace("|", "\\|").replace("\n", " ")[:160],
            )
        )
    lines.append("")
    (graph_dir / "edges.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # Contradictions only — the highest-stakes view
    contradictions = [e for e in edges if e.edge_type.value == "contradicts"]
    atom_by_id = {a.id: a for a in compile_result.atoms}
    lines = ["# Contradictions",
             "",
             f"_{len(contradictions)} contradicts edges • each one is a manual review item_",
             ""]
    for e in sorted(contradictions, key=lambda x: x.id):
        a = atom_by_id.get(e.from_atom_id)
        b = atom_by_id.get(e.to_atom_id)
        lines.append(f"## `{e.id}` — {(e.metadata or {}).get('edge_family', '—')}")
        lines.append("")
        lines.append(f"- reason: {e.reason}")
        lines.append(f"- confidence: {e.confidence:.2f}")
        lines.append(f"- cross-artifact: {'yes' if (e.metadata or {}).get('cross_artifact') else 'no'}")
        if a is not None:
            lines.append(f"- **from** `{a.id}` ({a.atom_type.value}, {a.authority_class.value}): {a.raw_text.strip()[:240]}")
        if b is not None:
            lines.append(f"- **to** `{b.id}` ({b.atom_type.value}, {b.authority_class.value}): {b.raw_text.strip()[:240]}")
        lines.append(f"- [ ] reviewed and resolved")
        lines.append("")
    (graph_dir / "contradictions.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ── packets/REVIEW.md ────────────────────────────────────────────────────


def _write_packets_markdown(root: Path, compile_result: CompileResult) -> None:
    pkt_dir = root / "packets"
    pkt_dir.mkdir(parents=True, exist_ok=True)
    atom_by_id = {a.id: a for a in compile_result.atoms}

    lines: list[str] = []
    lines.append("# Packets — needs_review / contradiction first")
    lines.append("")
    lines.append(f"_{len(compile_result.packets)} packets total_")
    lines.append("")

    def packet_sort(p: EvidencePacket) -> tuple:
        # needs_review first, then contradictions, then by anchor key
        return (
            0 if p.status == PacketStatus.needs_review else 1,
            0 if p.contradicting_atom_ids else 1,
            p.family.value,
            p.anchor_key,
            p.id,
        )

    for p in sorted(compile_result.packets, key=packet_sort):
        lines.append(f"## `{p.id}` — {p.family.value} ({p.status.value})")
        lines.append("")
        lines.append(f"- anchor: `{p.anchor_key}`")
        lines.append(f"- confidence: {p.confidence:.2f}")
        if p.review_flags:
            lines.append(f"- flags: {', '.join(p.review_flags)}")
        lines.append(f"- reason: {p.reason}")
        if p.governing_atom_ids:
            lines.append("- governing atoms:")
            for aid in p.governing_atom_ids:
                a = atom_by_id.get(aid)
                if a:
                    lines.append(f"  - `{aid}` ({a.authority_class.value}): {a.raw_text.strip()[:200]}")
        if p.contradicting_atom_ids:
            lines.append("- contradicting atoms:")
            for aid in p.contradicting_atom_ids:
                a = atom_by_id.get(aid)
                if a:
                    lines.append(f"  - `{aid}` ({a.authority_class.value}): {a.raw_text.strip()[:200]}")
        if p.certificate is not None:
            lines.append(f"- certificate: {p.certificate.existence_reason}")
        lines.append(f"- [ ] reviewed")
        lines.append("")
    (pkt_dir / "REVIEW.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ── artifacts/<stem>/ ────────────────────────────────────────────────────


def _write_artifact_dossiers(
    root: Path,
    *,
    compile_result: CompileResult,
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    artifact_paths: dict[str, Path],
    edges_by_atom: dict[str, list[EvidenceEdge]],
) -> None:
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = compile_result.manifest
    routing_by_artifact: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        for row in manifest.parser_routing:
            routing_by_artifact[row.get("artifact_id", "")] = row

    for fp in (manifest.artifact_fingerprints if manifest else []):
        artifact_id = fp.artifact_id
        stem = Path(fp.filename).stem
        adir = artifacts_dir / stem
        adir.mkdir(parents=True, exist_ok=True)

        # Copy original (if available locally).
        original = artifact_paths.get(artifact_id)
        if original is not None and original.is_file():
            try:
                shutil.copy2(original, adir / f"original{Path(fp.filename).suffix}")
            except OSError:
                pass

        atoms = sorted(atoms_by_artifact.get(artifact_id) or [], key=lambda a: a.id)
        (adir / "atoms.json").write_text(
            json.dumps([a.model_dump(mode="json") for a in atoms], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        routing = routing_by_artifact.get(artifact_id, {})
        (adir / "routing.json").write_text(
            json.dumps(routing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Per-artifact REVIEW.md
        lines: list[str] = []
        lines.append(f"# `{fp.filename}` review")
        lines.append("")
        lines.append(f"- artifact_id: `{artifact_id}`")
        lines.append(f"- type: `{fp.artifact_type.value}`")
        lines.append(f"- sha256: `{fp.sha256}`")
        lines.append(f"- size: {fp.size_bytes:,} bytes")
        lines.append(f"- parser: `{fp.parser_name}` v{fp.parser_version}")
        if routing:
            lines.append(
                f"- routing confidence: {routing.get('confidence', 0):.2f} "
                f"({'cache hit' if routing.get('cache_hit') else 'cache miss'})"
            )
            if routing.get("reasons"):
                lines.append(f"- routing reasons: {', '.join(routing['reasons'][:6])}")
        lines.append("")
        lines.append(f"_{len(atoms)} atoms extracted_")
        lines.append("")
        lines.append("## Atoms (raw text → normalized claim → entity keys)")
        lines.append("")
        lines.append("| Atom | Type | Authority | Conf | Receipt | Entity keys | Raw text |")
        lines.append("|---|---|---|---|---|---|---|")
        for a in atoms:
            receipt_state = _atom_receipt_state(a)
            keys = ", ".join(a.entity_keys[:4]) or "—"
            text = (a.raw_text or "").replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(
                f"| `{a.id}` | {a.atom_type.value} | {a.authority_class.value} "
                f"| {a.confidence:.2f} | {receipt_state} | {keys} | {text} |"
            )
        lines.append("")

        # Per-atom checklist with cross-references
        lines.append("## Per-atom review checklist")
        lines.append("")
        for a in atoms:
            connected = edges_by_atom.get(a.id) or []
            edge_summary = "—"
            if connected:
                fams = Counter((e.metadata or {}).get("edge_family", "unspecified") for e in connected)
                edge_summary = f"{len(connected)} edges ({', '.join(f'{k}×{v}' for k, v in fams.most_common(3))})"
            lines.append(f"- [ ] `{a.id}` ({a.atom_type.value}) — {edge_summary}")
            lines.append(f"  - raw: _{(a.raw_text or '').strip()[:240]}_")
            if a.normalized_text and a.normalized_text != a.raw_text:
                lines.append(f"  - normalized: _{(a.normalized_text or '').strip()[:240]}_")
            if a.review_flags:
                lines.append(f"  - flags: {', '.join(a.review_flags)}")
            for receipt in a.receipts:
                if receipt.replay_status != "verified":
                    lines.append(f"  - receipt {receipt.replay_status}: {receipt.reason}")
        lines.append("")
        (adir / "REVIEW.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _render_pack_suggestions(gap_report: dict, pack: object) -> str | None:
    """Render a copy/pasteable YAML snippet from a gap report.

    Operators can drop the snippet into ``app/domain/<pack>.yaml`` to
    teach parser-os about the new vocabulary.  Returns ``None`` when
    there's nothing to suggest (no gaps).  See PRODUCTION_GAPS /
    Week 5.
    """
    vocab_gaps = gap_report.get("vocab_gaps") or []
    if not vocab_gaps:
        return None
    by_kind: dict[str, list[dict]] = {}
    for gap in vocab_gaps:
        kind = gap.get("kind") or "unknown"
        by_kind.setdefault(kind, []).append(gap)
    if not by_kind:
        return None

    pack_id = getattr(pack, "pack_id", "default_pack")
    lines: list[str] = []
    lines.append(f"# pack_suggestions.yaml — drop these into app/domain/{pack_id}.yaml")
    lines.append("# Auto-generated from the ontology_gaps detector.  Each candidate is")
    lines.append("# tagged with its occurrence count so you can prioritize.  Review")
    lines.append("# every entry before merging — false positives DO show up.")
    lines.append("")
    if "device" in by_kind:
        lines.append("# Suggested additions to device_aliases:")
        lines.append("device_aliases:")
        for gap in sorted(by_kind["device"], key=lambda g: -g["occurrences"])[:30]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"  # {count}x — sample: {gap['sample_text'][:60]}")
            slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "device"
            lines.append(f"  {slug}:")
            lines.append(f"    - {phrase!r}")
        lines.append("")
    if "vendor" in by_kind:
        lines.append("# Suggested vendor canonical keys (extend the entity_extraction")
        lines.append("# cross-pack vendor catalog or pack.entity_types for vendor):")
        lines.append("vendor_candidates:")
        for gap in sorted(by_kind["vendor"], key=lambda g: -g["occurrences"])[:30]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"  - {phrase!r}  # {count}x")
        lines.append("")
    if "part_number" in by_kind:
        lines.append("# Suggested part-number canonical keys (mostly for documentation —")
        lines.append("# part numbers are extracted automatically by the universal SKU regex):")
        lines.append("part_number_candidates:")
        for gap in sorted(by_kind["part_number"], key=lambda g: -g["occurrences"])[:30]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"  - {phrase!r}  # {count}x")
        lines.append("")
    if "site" in by_kind:
        lines.append("# Suggested site_alias_patterns:")
        lines.append("site_alias_patterns:")
        for gap in sorted(by_kind["site"], key=lambda g: -g["occurrences"])[:20]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"  - {phrase!r}  # {count}x")
        lines.append("")
    if "constraint" in by_kind:
        lines.append("# Suggested constraint_patterns:")
        lines.append("constraint_patterns:")
        lines.append("  custom:")
        for gap in sorted(by_kind["constraint"], key=lambda g: -g["occurrences"])[:20]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"    - {phrase!r}  # {count}x")
        lines.append("")
    if "exclusion" in by_kind:
        lines.append("# Suggested exclusion_patterns:")
        lines.append("exclusion_patterns:")
        for gap in sorted(by_kind["exclusion"], key=lambda g: -g["occurrences"])[:20]:
            phrase = gap["phrase"].replace("'", "")
            count = gap["occurrences"]
            lines.append(f"  - {phrase!r}  # {count}x")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _atom_receipt_state(atom: EvidenceAtom) -> str:
    if not atom.receipts:
        return "none"
    statuses = {r.replay_status for r in atom.receipts}
    if "failed" in statuses:
        return "FAILED"
    if statuses == {"verified"}:
        return "verified"
    if "verified" in statuses:
        return "partial"
    return "unsupported"


__all__ = ["write_review_folder"]
