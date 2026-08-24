"""Production hand-off report builder.

Bundles a compile result + compare verdict + extraction summary + plain-
English failure analysis into one operator-friendly package.  The output
is a directory (and optionally a ZIP) that a non-developer can read
top-down without having to know the internal data shapes.

The report is the answer to "I gave parser-os to my tester, what do I
get back?" — every produced atom, every parser routing decision, every
warning, every ontology gap, every cross-artifact contradiction, plus a
1-page executive summary that surfaces the top failure modes in plain
English.

Layout produced::

    <out_dir>/
    ├── REPORT.md                       # 1-page executive summary
    ├── result.json                     # full CompileResult JSON
    ├── compare.json                    # gold compare verdict (when gold present)
    ├── reviews/cmp_<id>/               # per-compile review folder (atoms,
    │                                   # packets, ontology gaps, warnings,
    │                                   # contradictions, pack suggestions)
    └── envelope/                       # OrbitBrief envelope (JSON + Markdown)
        ├── orbitbrief.input.json
        └── orbitbrief.input.md

When ``--zip`` is passed an additional ``<out_dir>.zip`` is written.

The ``REPORT.md`` content is the load-bearing piece — see
``_render_executive_report`` for the structure.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{100 * numerator / denominator:.0f}%"


def _fmt_ms(ms: float | int | None) -> str:
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def _summarize_atoms(atoms: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(a.get("atom_type") or "?" for a in atoms)
    by_authority = Counter(a.get("authority_class") or "?" for a in atoms)
    confidences = [a.get("confidence") for a in atoms if isinstance(a.get("confidence"), (int, float))]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < 0.7)

    all_keys: set[str] = set()
    for a in atoms:
        all_keys.update(a.get("entity_keys") or [])
    by_prefix: dict[str, int] = Counter()
    for k in all_keys:
        prefix = k.split(":", 1)[0] if ":" in k else "no_prefix"
        by_prefix[prefix] += 1

    receipts: Counter[str] = Counter()
    for a in atoms:
        for r in a.get("receipts") or []:
            receipts[r.get("replay_status") or "?"] += 1

    return {
        "count": len(atoms),
        "by_type": dict(by_type),
        "by_authority": dict(by_authority),
        "avg_confidence": round(avg_conf, 3),
        "low_confidence_count": low_conf_count,
        "distinct_entity_keys": len(all_keys),
        "entity_keys_by_prefix": dict(by_prefix),
        "receipts": dict(receipts),
    }


def _summarize_packets(packets: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = Counter(p.get("family") or "?" for p in packets)
    by_status = Counter(p.get("status") or "?" for p in packets)
    contradiction_count = sum(
        1 for p in packets if p.get("contradicting_atom_ids")
    )
    return {
        "count": len(packets),
        "by_family": dict(by_family),
        "by_status": dict(by_status),
        "with_contradictions": contradiction_count,
    }


def _summarize_coverage(
    result_dict: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Run the existing coverage report against the compile result.

    Surfaces *evidence-coverage* (what fraction of source-text segments
    became atoms vs. were ignored or unsupported) per artifact.  The
    raw ``ProjectCoverageReport`` JSON is written to
    ``<out_dir>/coverage.json`` so reviewers can drill in.

    Returns a compact dict for the executive summary; absent or empty
    when coverage info isn't computable for this project.
    """
    try:
        from app.eval.coverage import build_coverage_report
    except Exception:
        return {}
    try:
        report = build_coverage_report(result_dict)
    except Exception as exc:
        return {"error": f"coverage report failed: {exc}"}
    coverage_path = out_dir / "coverage.json"
    coverage_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {
        "overall_coverage_rate": float(report.overall_coverage_rate or 0.0),
        "artifact_count": len(report.artifact_reports),
        "warnings": list(report.warnings or []),
        "per_artifact": [
            {
                "filename": a.filename,
                "segment_count": a.segment_count,
                "covered_count": a.covered_count,
                "ignored_count": a.ignored_count,
                "unsupported_count": a.unsupported_count,
                "coverage_rate": float(a.coverage_rate or 0.0),
            }
            for a in report.artifact_reports
        ],
        "recommended_parser_improvements": list(
            report.recommended_parser_improvements or []
        ),
    }


def _summarize_artifacts(
    manifest: dict[str, Any], atoms: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-artifact summary table.

    Pulls from ``manifest.artifact_fingerprints`` (file metadata) and
    ``manifest.parser_routing`` (which parser was chosen + confidence)
    and counts atoms per artifact_id from the atoms list.
    """
    routing_by_id: dict[str, dict[str, Any]] = {
        r.get("artifact_id"): r for r in (manifest.get("parser_routing") or [])
    }
    atoms_by_id: Counter[str] = Counter()
    for a in atoms:
        aid = a.get("artifact_id")
        if aid:
            atoms_by_id[aid] += 1

    out = []
    for fp in manifest.get("artifact_fingerprints") or []:
        aid = fp.get("artifact_id") or ""
        routing = routing_by_id.get(aid) or {}
        out.append(
            {
                "filename": fp.get("filename"),
                "artifact_type": fp.get("artifact_type"),
                "size_bytes": fp.get("size_bytes"),
                "parser_name": fp.get("parser_name") or routing.get("chosen_parser"),
                "parser_version": fp.get("parser_version") or routing.get("parser_version"),
                "atom_count": atoms_by_id.get(aid, 0),
                "routing_confidence": routing.get("confidence"),
                "routing_reasons": routing.get("reasons") or [],
                "cache_hit": routing.get("cache_hit"),
            }
        )
    return out


def _failure_analysis(
    *,
    result: dict[str, Any],
    compare_report: dict[str, Any] | None,
    atom_summary: dict[str, Any],
    packet_summary: dict[str, Any],
) -> list[str]:
    """Plain-English bullet list of what went wrong / needs review."""
    findings: list[str] = []

    receipts = atom_summary["receipts"]
    failed = receipts.get("failed", 0)
    unsupported = receipts.get("unsupported", 0)
    if failed:
        findings.append(
            f"❌ **{failed} atoms failed source-replay** — the parser couldn't "
            "re-extract the atom text from the source artifact.  This usually "
            "means the locator went stale (page renumbered, row reshuffled).  "
            "Investigate `warnings.md` for the specific atom IDs."
        )
    if unsupported:
        findings.append(
            f"⚠️ {unsupported} atoms have `replay_status=unsupported` — the "
            "parser format doesn't carry replay info.  Output is still usable "
            "but you can't auto-verify each atom against its source."
        )

    if atom_summary["low_confidence_count"]:
        findings.append(
            f"⚠️ {atom_summary['low_confidence_count']} atoms below 0.70 "
            "confidence.  Look at `reviews/<compile>/packets/REVIEW.md` for "
            "anything worth promoting / demoting."
        )

    quality = result.get("quality") or {}
    parsers_zero = quality.get("parsers_with_zero_atoms") or []
    if parsers_zero:
        findings.append(
            f"❌ **{len(parsers_zero)} parser(s) ran but produced 0 atoms** — "
            f"{parsers_zero}.  Either the artifact was empty, or the parser "
            "matched but couldn't extract anything (a content/format mismatch)."
        )

    parsers_low = quality.get("parsers_with_low_confidence") or []
    if parsers_low:
        findings.append(
            f"⚠️ {len(parsers_low)} parser routing decision(s) had low "
            f"confidence — re-check `reviews/<compile>/REVIEW.md` parser "
            "routing table."
        )

    pack_routing_source = quality.get("pack_routing_source") or "unknown"
    if pack_routing_source == "default":
        findings.append(
            "⚠️ Domain pack auto-routing fell through to `default_pack`.  "
            "Add a `project.yaml::service_line` or pack a `SOURCE_NOTES.md` "
            "with vertical keywords for sharper extraction next run."
        )

    er_rate = quality.get("entity_resolution_rate")
    if isinstance(er_rate, (int, float)) and er_rate < 0.50:
        findings.append(
            f"⚠️ Entity-resolution rate is **{er_rate:.0%}** (under 50%) — "
            "many atoms have no canonical entity_keys.  Check `ontology_gaps.md` "
            "for vendor / device candidates the pack didn't recognize."
        )

    pkt_spec = quality.get("packet_specificity")
    if isinstance(pkt_spec, (int, float)) and pkt_spec < 0.85:
        findings.append(
            f"⚠️ Packet specificity is **{pkt_spec:.0%}** — many packets have "
            "generic `*:unknown` anchors.  Domain pack or parser may need "
            "more aliases."
        )

    if packet_summary["with_contradictions"]:
        findings.append(
            f"📋 {packet_summary['with_contradictions']} packet(s) carry "
            "contradicting atoms — see `reviews/<compile>/graph/contradictions.md`.  "
            "Each one is a manual review item."
        )

    if compare_report is not None:
        overall = compare_report.get("overall") or {}
        passed = overall.get("pass", 0)
        total = overall.get("total_checked", 0)
        if total and passed < total:
            metrics = compare_report.get("metrics") or {}
            failing: list[str] = []
            for name, m in metrics.items():
                if m.get("verdict") != "fail":
                    continue
                if "missing" in m and isinstance(m["missing"], list):
                    failing.append(f"`{name}` (missing {m['missing']})")
                elif "missing_count" in m:
                    failing.append(
                        f"`{name}` ({m['missing_count']} keys missing of "
                        f"{m.get('expected_count', '?')})"
                    )
                elif "actual" in m and "expected_min" in m:
                    failing.append(
                        f"`{name}` (actual={m['actual']}, "
                        f"expected_min={m['expected_min']})"
                    )
                else:
                    failing.append(f"`{name}`")
            findings.append(
                f"📋 Gold compare: **{passed}/{total} pass** ("
                f"{100*overall.get('pass_fraction', 0):.0f}%).  "
                f"Failing metrics: {', '.join(failing)}."
            )

    if not findings:
        findings.append("✅ No anomalies detected.  Compile is clean.")

    return findings


def _render_executive_report(
    *,
    result: dict[str, Any],
    compare_report: dict[str, Any] | None,
    atom_summary: dict[str, Any],
    packet_summary: dict[str, Any],
    artifact_summary: list[dict[str, Any]],
    coverage_summary: dict[str, Any] | None = None,
    project_dir: Path,
    out_dir: Path,
) -> str:
    compile_id = result.get("compile_id", "unknown")
    project_id = result.get("project_id", "unknown")
    manifest = result.get("manifest") or {}
    input_sig = manifest.get("input_signature") or result.get("input_signature") or ""
    output_sig = manifest.get("output_signature") or result.get("output_signature") or ""
    trace = result.get("trace") or {}
    quality = result.get("quality") or {}

    lines: list[str] = []
    lines.append(f"# Parser-OS production report — `{compile_id}`")
    lines.append("")
    lines.append(
        f"_project_: `{project_id}`  •  _generated_: "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    # ── Verdict header ──────────────────────────────────────────
    if compare_report is not None:
        overall = compare_report.get("overall") or {}
        pf = overall.get("pass_fraction", 0)
        passed = overall.get("pass", 0)
        total = overall.get("total_checked", 0)
        verdict_emoji = "🟢" if pf >= 0.9 else ("🟡" if pf >= 0.5 else "🔴")
        lines.append(
            f"## {verdict_emoji} Gold-compare verdict: **{passed}/{total} "
            f"pass** ({100*pf:.0f}%)"
        )
    else:
        lines.append("## ✅ Compile completed (no gold standard to compare against)")
    lines.append("")

    # ── Executive summary ──────────────────────────────────────
    findings = _failure_analysis(
        result=result,
        compare_report=compare_report,
        atom_summary=atom_summary,
        packet_summary=packet_summary,
    )
    lines.append("## What went well / what didn't")
    lines.append("")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    # ── Inputs ─────────────────────────────────────────────────
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"_Source:_ `{project_dir}`")
    lines.append("")
    lines.append("| Artifact | Type | Size | Parser | Routing | Atoms |")
    lines.append("|---|---|---:|---|---:|---:|")
    for a in artifact_summary:
        size_kb = (a["size_bytes"] or 0) / 1024
        size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        routing = (
            f"{a['routing_confidence']:.2f}"
            if isinstance(a.get("routing_confidence"), (int, float))
            else "—"
        )
        lines.append(
            f"| `{a['filename']}` | {a['artifact_type']} | {size_str} | "
            f"`{a['parser_name']}` v{a['parser_version']} | "
            f"{routing} | {a['atom_count']} |"
        )
    lines.append("")

    # ── What was extracted ─────────────────────────────────────
    lines.append("## What was extracted")
    lines.append("")
    lines.append(f"**{atom_summary['count']} atoms** across the artifacts:")
    lines.append("")
    for atype, count in sorted(
        atom_summary["by_type"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{atype}`: {count}")
    lines.append("")
    avg = atom_summary["avg_confidence"]
    lines.append(
        f"Average atom confidence: **{avg:.2f}**.  "
        f"Source-replay receipts: {atom_summary['receipts']}"
    )
    lines.append("")
    lines.append(
        f"**{atom_summary['distinct_entity_keys']} distinct entity keys** by prefix:"
    )
    lines.append("")
    for prefix, count in sorted(
        atom_summary["entity_keys_by_prefix"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{prefix}:`*  →  {count}")
    lines.append("")

    # ── Packets ────────────────────────────────────────────────
    lines.append("## Packets produced")
    lines.append("")
    lines.append(f"**{packet_summary['count']} packets** by family:")
    lines.append("")
    for fam, count in sorted(
        packet_summary["by_family"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{fam}`: {count}")
    lines.append("")
    lines.append(
        f"Status distribution: {packet_summary['by_status']} • "
        f"{packet_summary['with_contradictions']} carry contradictions"
    )
    lines.append("")

    # ── Source-text coverage ────────────────────────────────────
    if coverage_summary and coverage_summary.get("per_artifact"):
        # The coverage segmenter doesn't have data for every parser
        # format yet — skip the section when none of the artifacts
        # have segments (rather than report a misleading 0%).
        with_segments = [
            a for a in coverage_summary["per_artifact"] if a.get("segment_count")
        ]
        if with_segments:
            overall = coverage_summary.get("overall_coverage_rate", 0.0)
            emoji = "🟢" if overall >= 0.8 else ("🟡" if overall >= 0.5 else "🔴")
            lines.append(f"## {emoji} Source-text coverage")
            lines.append("")
            lines.append(
                f"_What fraction of source-text segments produced an atom?_  "
                f"**Overall: {overall:.0%}**"
            )
            lines.append("")
            lines.append(
                "| Artifact | Segments | Covered | Ignored | Unsupported | Coverage |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for a in with_segments:
                lines.append(
                    f"| `{a['filename']}` | {a['segment_count']} | "
                    f"{a['covered_count']} | {a['ignored_count']} | "
                    f"{a['unsupported_count']} | "
                    f"{a['coverage_rate']:.0%} |"
                )
            lines.append("")
            recs = coverage_summary.get("recommended_parser_improvements") or []
            if recs:
                lines.append("**Parser improvement suggestions:**")
                lines.append("")
                for r in recs[:8]:
                    lines.append(f"- {r}")
                lines.append("")
            lines.append(
                "_Drill in: `coverage.json` has every segment with its "
                "covered/ignored verdict + reason._"
            )
            lines.append("")

    # ── Quality metrics ────────────────────────────────────────
    if quality:
        lines.append("## Quality metrics")
        lines.append("")
        lines.append(f"- entity_resolution_rate: **{quality.get('entity_resolution_rate', 0):.0%}**")
        lines.append(f"- packet_specificity: **{quality.get('packet_specificity', 0):.0%}**")
        lines.append(
            f"- parser_atom_yield_rate: "
            f"**{quality.get('parser_atom_yield_rate', 0):.0%}**"
        )
        lines.append(
            f"- parser_routing_confidence_avg: "
            f"**{quality.get('parser_routing_confidence_avg', 0):.2f}**"
        )
        lines.append(
            f"- atoms_per_artifact: **{quality.get('atoms_per_artifact', 0):.1f}**"
        )
        lines.append(
            f"- pack: **{quality.get('pack_id', '?')}** "
            f"(routing source: `{quality.get('pack_routing_source', '?')}`, "
            f"confidence: {quality.get('pack_routing_confidence', 0):.2f})"
        )
        lines.append("")

    # ── Stage timing ───────────────────────────────────────────
    if trace.get("stages"):
        lines.append("## Stage timing")
        lines.append("")
        lines.append("| Stage | Duration | In | Out | Errors |")
        lines.append("|---|---:|---:|---:|---:|")
        for stage in trace["stages"]:
            lines.append(
                f"| `{stage.get('stage_name')}` | "
                f"{_fmt_ms(stage.get('duration_ms'))} | "
                f"{stage.get('input_count') or '—'} | "
                f"{stage.get('output_count') or '—'} | "
                f"{len(stage.get('errors') or [])} |"
            )
        lines.append("")
        total_ms = trace.get("total_duration_ms", 0)
        lines.append(f"_Total compile time: **{_fmt_ms(total_ms)}**_")
        lines.append("")

    # ── Determinism receipts ───────────────────────────────────
    lines.append("## Determinism receipts")
    lines.append("")
    lines.append(f"- input_signature: `{input_sig}`")
    lines.append(f"- output_signature: `{output_sig}`")
    lines.append(
        "_Re-running this compile on identical artifacts produces the "
        "same `output_signature`.  Use this for change-detection in CI._"
    )
    lines.append("")

    # ── Where to dig deeper ────────────────────────────────────
    lines.append("## Where to dig deeper")
    lines.append("")
    lines.append("- `result.json` — full compile result (every atom, edge, packet)")
    if compare_report is not None:
        lines.append("- `compare.json` — per-metric gold-compare verdicts")
    lines.append(
        "- `reviews/cmp_<id>/REVIEW.md` — per-compile review folder "
        "(walk it top-down; per-artifact dossiers under `artifacts/<file>/REVIEW.md`)"
    )
    lines.append(
        "- `reviews/cmp_<id>/ontology_gaps.md` — phrases the pack didn't "
        "recognize (vendors / sites / part numbers / device aliases)"
    )
    lines.append(
        "- `reviews/cmp_<id>/pack_suggestions.yaml` — copy/paste-ready "
        "YAML you can merge into the active pack"
    )
    lines.append(
        "- `reviews/cmp_<id>/warnings.md` — every warning emitted during "
        "the compile, grouped by source"
    )
    lines.append(
        "- `envelope/orbitbrief.input.json` — the `orbitbrief.input.v2` "
        "envelope downstream consumers (OrbitBrief, dashboards, review "
        "tools) read"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def build_production_report(
    *,
    project_dir: Path,
    out_dir: Path,
    domain_pack: str | None = None,
    no_cache: bool = False,
    skip_orbitbrief: bool = False,
    zip_bundle: bool = True,
    abstain_threshold: float = 0.7,
    allow_errors: bool = False,
    allow_unverified_receipts: bool = False,
) -> dict[str, Any]:
    """Run a full production compile + report bundle.

    Imports are local so importing this module doesn't drag in the whole
    pipeline (keeps CLI startup snappy).
    """
    from app.core.compiler import compile_project
    from app.core.gold_compare import compare_to_gold
    from app.core.orbitbrief_envelope import (
        build_orbitbrief_envelope,
        write_orbitbrief_envelope,
    )
    from app.core.review_folder import write_review_folder
    from app.domain import load_domain_pack

    project_dir = Path(project_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews_root = out_dir / "reviews"
    envelope_dir = out_dir / "envelope"
    result_path = out_dir / "result.json"
    trace_path = out_dir / "trace.json"

    # 1) Compile (the same call shape used by `parser-os compile`).
    # Lazy import: calibration pulls in sklearn; keep it off the module import path.
    from app.learning.calibration import default_calibrator_path as _default_calibrator_path
    result = compile_project(
        project_dir=project_dir,
        domain_pack=domain_pack,
        # Use the trained calibrator when SOWSMITH_CALIBRATOR_PATH points at a
        # real artifact; None (default) keeps the prod path a no-op until then.
        calibrator_path=_default_calibrator_path(),
        abstain_threshold=abstain_threshold,
        use_cache=not no_cache,
        allow_errors=allow_errors,
        allow_unverified_receipts=allow_unverified_receipts,
    )

    # 2) Persist result + trace.
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if result.trace is not None:
        trace_path.write_text(result.trace.model_dump_json(indent=2), encoding="utf-8")

    # 3) Optional OrbitBrief envelope.
    if not skip_orbitbrief:
        envelope = build_orbitbrief_envelope(
            project_dir=project_dir,
            compile_result=result,
        )
        write_orbitbrief_envelope(
            project_dir=project_dir,
            envelope=envelope,
            out_dir=envelope_dir,
        )

    # 4) Per-compile review folder (opt-out via skip_orbitbrief mirror).
    if not skip_orbitbrief:
        manifest = result.manifest
        artifact_paths: dict[str, Path] = {}
        if manifest is not None:
            for fp in manifest.artifact_fingerprints:
                candidate = (project_dir / fp.filename).resolve()
                if candidate.is_file():
                    artifact_paths[fp.artifact_id] = candidate
        active_pack = load_domain_pack(domain_pack)
        write_review_folder(
            project_dir=project_dir,
            compile_result=result,
            out_dir=reviews_root,
            pack=active_pack,
            artifact_paths=artifact_paths,
        )

    # 5) Re-load as plain dict for summary computation (avoids relying on
    #    Pydantic model attribute names downstream).
    result_dict = json.loads(result_path.read_text(encoding="utf-8"))

    # 6) Gold compare (auto-detect labels/gold_standard.json).
    compare_report: dict[str, Any] | None = None
    gold_path = project_dir / "labels" / "gold_standard.json"
    if gold_path.is_file():
        gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
        compare_report = compare_to_gold(gold=gold_payload, compiled=result_dict)
        (out_dir / "compare.json").write_text(
            json.dumps(compare_report, indent=2), encoding="utf-8"
        )

    # 7) Summaries for the executive report.
    atom_summary = _summarize_atoms(result_dict.get("atoms") or [])
    packet_summary = _summarize_packets(result_dict.get("packets") or [])
    artifact_summary = _summarize_artifacts(
        result_dict.get("manifest") or {},
        result_dict.get("atoms") or [],
    )
    coverage_summary = _summarize_coverage(result_dict, out_dir)

    # 8) Render REPORT.md.
    report_md = _render_executive_report(
        result=result_dict,
        compare_report=compare_report,
        atom_summary=atom_summary,
        packet_summary=packet_summary,
        artifact_summary=artifact_summary,
        coverage_summary=coverage_summary,
        project_dir=project_dir,
        out_dir=out_dir,
    )
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    # 9) Optional ZIP bundle.
    zip_path: Path | None = None
    if zip_bundle:
        zip_path = Path(str(out_dir) + ".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(out_dir.parent)
                    zf.write(path, arcname=str(arcname))

    manifest_dict = result_dict.get("manifest") or {}
    return {
        "out_dir": str(out_dir),
        "report_md": str(out_dir / "REPORT.md"),
        "result_json": str(result_path),
        "compare_json": str(out_dir / "compare.json") if compare_report else None,
        "envelope_json": (
            str(envelope_dir / "orbitbrief.input.json") if not skip_orbitbrief else None
        ),
        "review_folder": (
            str(reviews_root / result.compile_id) if not skip_orbitbrief else None
        ),
        "zip_path": str(zip_path) if zip_path else None,
        "compile_id": result_dict.get("compile_id"),
        "input_signature": manifest_dict.get("input_signature"),
        "output_signature": manifest_dict.get("output_signature"),
        "atom_count": atom_summary["count"],
        "packet_count": packet_summary["count"],
        "gold_pass_fraction": (
            (compare_report or {}).get("overall", {}).get("pass_fraction")
        ),
    }


__all__ = ["build_production_report"]
