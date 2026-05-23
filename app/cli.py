from __future__ import annotations

import json
from pathlib import Path

import typer

from app.core.compiler import compile_project
from app.core.orbitbrief_envelope import (
    build_orbitbrief_envelope,
    write_orbitbrief_envelope,
)
from app.core.review_folder import write_review_folder
from app.domain import load_domain_pack

app = typer.Typer(help="Purtera Evidence Compiler MVP CLI")


@app.command()
def compile(
    project_dir: Path,
    out: Path = typer.Option(..., "--out"),
    trace_out: Path | None = typer.Option(None, "--trace-out"),
    domain_pack: str | None = typer.Option(None, "--domain-pack"),
    calibrator_path: Path | None = typer.Option(None, "--calibrator-path"),
    abstain_threshold: float = typer.Option(0.70, "--abstain-threshold"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable incremental artifact cache reuse"),
    allow_errors: bool = typer.Option(False, "--allow-errors"),
    allow_unverified_receipts: bool = typer.Option(False, "--allow-unverified-receipts"),
    orbitbrief_out: Path | None = typer.Option(
        None,
        "--orbitbrief-out",
        help="Directory for the orbitbrief.input.{json,md} envelope (defaults to <project>/.orbitbrief).",
    ),
    skip_orbitbrief: bool = typer.Option(
        False,
        "--skip-orbitbrief",
        help="Skip writing the OrbitBrief project envelope.",
    ),
    review_out: Path | None = typer.Option(
        None,
        "--review-out",
        help=(
            "Write a per-compile human review folder under the given directory "
            "(``<review-out>/<compile_id>/``).  Includes per-artifact dossiers, "
            "ontology gaps, contradictions, and a checklist REVIEW.md."
        ),
    ),
) -> None:
    """Compile a project directory into structured evidence JSON."""
    result = compile_project(
        project_dir=project_dir,
        domain_pack=domain_pack,
        calibrator_path=calibrator_path,
        abstain_threshold=abstain_threshold,
        use_cache=not no_cache,
        allow_errors=allow_errors,
        allow_unverified_receipts=allow_unverified_receipts,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if trace_out is not None and result.trace is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(result.trace.model_dump_json(indent=2), encoding="utf-8")
    envelope_paths: tuple[Path, Path] | None = None
    if not skip_orbitbrief:
        envelope = build_orbitbrief_envelope(
            project_dir=project_dir,
            compile_result=result,
        )
        envelope_paths = write_orbitbrief_envelope(
            project_dir=project_dir,
            envelope=envelope,
            out_dir=orbitbrief_out,
        )

    # Per-run review dossier — opt-in via --review-out so default compile output
    # stays small.  Reuses the same artifact discovery the compiler did so the
    # original files are copied verbatim into the dossier.
    review_root: Path | None = None
    if review_out is not None:
        manifest = result.manifest
        artifact_paths: dict[str, Path] = {}
        if manifest is not None:
            for fp in manifest.artifact_fingerprints:
                candidate = (project_dir / fp.filename).resolve()
                if candidate.is_file():
                    artifact_paths[fp.artifact_id] = candidate
        active_pack = load_domain_pack(domain_pack)
        review_root = write_review_folder(
            project_dir=project_dir.resolve(),
            compile_result=result,
            out_dir=review_out,
            pack=active_pack,
            artifact_paths=artifact_paths,
        )
    warning_count = len(result.warnings)
    error_count = len([w for w in result.warnings if str(w).startswith("ERROR:")])
    manifest = result.manifest
    verified_count = sum(1 for atom in result.atoms for receipt in atom.receipts if receipt.replay_status == "verified")
    unsupported_count = sum(
        1 for atom in result.atoms for receipt in atom.receipts if receipt.replay_status == "unsupported"
    )
    failed_count = sum(1 for atom in result.atoms for receipt in atom.receipts if receipt.replay_status == "failed")
    typer.echo(
        json.dumps(
            {
                "compile_id": result.compile_id,
                "input_signature": manifest.input_signature if manifest else None,
                "output_signature": manifest.output_signature if manifest else None,
                "atoms": len(result.atoms),
                "edges": len(result.edges),
                "packets": len(result.packets),
                "warnings": warning_count,
                "errors": error_count,
                "receipt_verified": verified_count,
                "receipt_unsupported": unsupported_count,
                "receipt_failed": failed_count,
                "trace_total_duration_ms": result.trace.total_duration_ms if result.trace is not None else None,
                "trace_stage_count": len(result.trace.stages) if result.trace is not None else 0,
                "cache_hits": manifest.cache_hits if manifest is not None else 0,
                "cache_misses": manifest.cache_misses if manifest is not None else 0,
                "reused_artifact_ids": manifest.reused_artifact_ids if manifest is not None else [],
                "orbitbrief_envelope_json": str(envelope_paths[0]) if envelope_paths else None,
                "orbitbrief_envelope_md": str(envelope_paths[1]) if envelope_paths else None,
                "review_folder": str(review_root) if review_root else None,
            }
        )
    )
    if result.trace is not None:
        stage_summary = ", ".join(
            f"{stage.stage_name}:{stage.duration_ms:.1f}ms"
            for stage in result.trace.stages
        )
        typer.echo(
            json.dumps(
                {
                    "compile_id": result.compile_id,
                    "trace_summary": stage_summary,
                }
            )
        )


@app.command("orbitbrief-envelope")
def orbitbrief_envelope(
    project_dir: Path = typer.Argument(..., help="Project directory containing the source artifacts."),
    compile_result: Path = typer.Option(..., "--compile-result", help="Path to a CompileResult JSON file."),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Where to write the envelope (defaults to <project>/.orbitbrief).",
    ),
) -> None:
    """Render an OrbitBrief envelope (JSON + markdown) from a saved compile result."""
    from app.core.schemas import CompileResult  # local import keeps CLI startup snappy

    payload = json.loads(compile_result.read_text(encoding="utf-8"))
    result = CompileResult.model_validate(payload)
    envelope = build_orbitbrief_envelope(
        project_dir=project_dir,
        compile_result=result,
    )
    json_path, md_path, sow_path = write_orbitbrief_envelope(
        project_dir=project_dir,
        envelope=envelope,
        out_dir=out_dir,
    )
    typer.echo(
        json.dumps(
            {
                "envelope_json": str(json_path),
                "envelope_md": str(md_path),
                "sow_md": str(sow_path),
                "documents": len(envelope.get("documents", [])),
                "atoms": len(envelope.get("atoms", [])),
                "packets": len(envelope.get("packets", [])),
            }
        )
    )


@app.command("batch-compile")
def batch_compile(
    projects_root: Path = typer.Argument(
        ...,
        help=(
            "Either a directory containing project subdirectories "
            "(e.g. real_data_cases/) or a glob pattern like "
            "'real_data_cases/STRESS_*'."
        ),
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        help="Directory to write per-project <project_id>.json + <project_id>.quality.json.",
    ),
    review_out: Path | None = typer.Option(
        None,
        "--review-out",
        help="Optional review-folder root (one subdir per compile).",
    ),
    domain_pack: str | None = typer.Option(
        None, "--domain-pack",
        help="Override domain pack for every project (skip auto-routing).",
    ),
    no_cache: bool = typer.Option(False, "--no-cache"),
    skip_orbitbrief: bool = typer.Option(True, "--skip-orbitbrief/--orbitbrief"),
    glob: str = typer.Option(
        "STRESS_*",
        "--glob",
        help="Glob pattern (relative to projects_root) for which subdirectories to compile.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop after the first project that raises (otherwise log + continue).",
    ),
) -> None:
    """Compile a batch of project directories.

    See PRODUCTION_GAPS.md P3.2.  Re-uses the same compile pipeline as
    ``compile`` but iterates directories so operators can re-run an
    entire stress corpus with a single command.
    """
    if not projects_root.exists():
        raise typer.BadParameter(f"projects_root does not exist: {projects_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve project list — accept either an existing directory we
    # walk with ``glob`` or an explicit glob pattern.
    if projects_root.is_dir():
        candidates = sorted(p for p in projects_root.glob(glob) if p.is_dir())
    else:
        candidates = sorted(Path(p) for p in projects_root.parent.glob(projects_root.name) if Path(p).is_dir())

    if not candidates:
        typer.echo(json.dumps({"event": "batch_no_projects_found", "projects_root": str(projects_root), "glob": glob}))
        raise typer.Exit(code=1)

    summaries: list[dict[str, object]] = []
    for project_dir in candidates:
        project_id = project_dir.name
        try:
            result = compile_project(
                project_dir=project_dir,
                domain_pack=domain_pack,
                use_cache=not no_cache,
            )
        except Exception as exc:  # pragma: no cover — production-grade
            typer.echo(json.dumps({
                "event": "batch_project_failed",
                "project_id": project_id,
                "error": str(exc),
            }))
            if fail_fast:
                raise
            continue
        out_path = out_dir / f"{project_id}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if result.quality is not None:
            quality_path = out_dir / f"{project_id}.quality.json"
            quality_path.write_text(
                result.quality.model_dump_json(indent=2),
                encoding="utf-8",
            )
        if review_out is not None:
            try:
                manifest = result.manifest
                artifact_paths: dict[str, Path] = {}
                if manifest is not None:
                    for fp in manifest.artifact_fingerprints:
                        candidate = (project_dir / fp.filename).resolve()
                        if candidate.is_file():
                            artifact_paths[fp.artifact_id] = candidate
                active_pack = load_domain_pack(domain_pack)
                write_review_folder(
                    project_dir=project_dir.resolve(),
                    compile_result=result,
                    out_dir=review_out,
                    pack=active_pack,
                    artifact_paths=artifact_paths,
                )
            except Exception as exc:  # pragma: no cover
                typer.echo(json.dumps({
                    "event": "batch_review_folder_failed",
                    "project_id": project_id,
                    "error": str(exc),
                }))

        if not skip_orbitbrief:
            try:
                envelope = build_orbitbrief_envelope(
                    project_dir=project_dir, compile_result=result
                )
                write_orbitbrief_envelope(project_dir=project_dir, envelope=envelope)
            except Exception as exc:  # pragma: no cover
                typer.echo(json.dumps({
                    "event": "batch_envelope_failed",
                    "project_id": project_id,
                    "error": str(exc),
                }))

        q = result.quality
        summaries.append({
            "project_id": project_id,
            "atoms": len(result.atoms),
            "edges": len(result.edges),
            "packets": len(result.packets),
            "entities": len(result.entities),
            "qty_conflicts": q.quantity_conflict_edge_count if q else 0,
            "entity_resolution_rate": q.entity_resolution_rate if q else 0.0,
            "packet_specificity": q.packet_specificity if q else 0.0,
            "pack_id": q.pack_id if q else "unknown",
            "warnings": len(result.warnings),
            "out": str(out_path),
        })

    typer.echo(json.dumps({"event": "batch_complete", "projects": summaries}, indent=2))


@app.command("compare")
def compare(
    gold: Path = typer.Option(..., "--gold", help="Gold-standard JSON file (e.g. real_data_cases/STRESS_X/labels/gold_standard.json)."),
    compiled: Path = typer.Option(..., "--compiled", help="CompileResult JSON file produced by `compile`."),
    fail_below: float = typer.Option(
        0.0,
        "--fail-below",
        help="Exit non-zero when overall pass-fraction is below this threshold (0.0 = informational mode).",
    ),
) -> None:
    """Compare a CompileResult against a gold-standard JSON.

    See PRODUCTION_GAPS.md P3.3.  Computes per-metric pass / fail
    against thresholds declared in the gold file (e.g.
    ``expected_min_atom_count: 60``).  Output is a JSON envelope with
    overall pass-fraction + per-metric verdicts so it can be wired
    into CI.
    """
    if not gold.is_file():
        raise typer.BadParameter(f"gold file not found: {gold}")
    if not compiled.is_file():
        raise typer.BadParameter(f"compiled file not found: {compiled}")
    from app.core.gold_compare import compare_to_gold  # local import keeps startup snappy

    gold_payload = json.loads(gold.read_text(encoding="utf-8"))
    compiled_payload = json.loads(compiled.read_text(encoding="utf-8"))
    report = compare_to_gold(gold=gold_payload, compiled=compiled_payload)
    typer.echo(json.dumps(report, indent=2))
    if fail_below > 0 and report["overall"]["pass_fraction"] < fail_below:
        raise typer.Exit(code=2)


@app.command("init")
def init_project(
    project_dir: Path = typer.Argument(..., help="Project root (will be created if missing)."),
    service_line: str | None = typer.Option(
        None, "--service-line",
        help="Pre-fill project.yaml service_line (e.g. security_camera, av, wireless).",
    ),
    customer: str | None = typer.Option(None, "--customer"),
    project_name: str | None = typer.Option(None, "--project-name"),
) -> None:
    """Scaffold a new parser-os project directory.

    Creates ``<project>/artifacts/``, ``<project>/labels/``,
    ``<project>/project.yaml`` (template), and ``<project>/.parserignore``
    (template).  Existing files are left in place.
    """
    from app.domain.project_config import write_default_project_yaml

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    (project_dir / "labels").mkdir(exist_ok=True)

    project_yaml = write_default_project_yaml(project_dir)
    if service_line or customer or project_name:
        # If the caller passed flags, replace the templated commented-out
        # entries with the real values.
        text = project_yaml.read_text(encoding="utf-8")
        if service_line:
            text += f"\nservice_line: {service_line}\n"
        if customer:
            text += f"customer: {customer}\n"
        if project_name:
            text += f"project_name: {project_name}\n"
        project_yaml.write_text(text, encoding="utf-8")

    parserignore = project_dir / ".parserignore"
    if not parserignore.is_file():
        parserignore.write_text(
            "# parser-os artifact ignore patterns (one glob per line)\n"
            "# Built-in ignores (labels/, .orbitbrief/, gold_standard.*,\n"
            "# SOURCE_NOTES.md, derived dirs) are always applied.\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps({
        "project_dir": str(project_dir.resolve()),
        "project_yaml": str(project_yaml),
        "artifacts_dir": str((project_dir / "artifacts").resolve()),
        "labels_dir": str((project_dir / "labels").resolve()),
    }))


@app.command("matrix")
def matrix(
    cases_dir: Path = typer.Option(
        ...,
        "--cases-dir",
        help="Directory containing one project subfolder per stress case (e.g. real_data_cases).",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output path for the JSON matrix report.",
    ),
    markdown_out: Path | None = typer.Option(
        None,
        "--markdown-out",
        help="Optional path for a human-readable Markdown table version of the report.",
    ),
    timeout_seconds: int = typer.Option(
        300,
        "--timeout-seconds",
        help="Per-case wall-clock budget (skip cases that exceed this).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable incremental artifact cache reuse.",
    ),
) -> None:
    """Run every stress case under ``--cases-dir`` and emit a green/red grid.

    Loops over every subdirectory of ``--cases-dir`` that contains a
    ``labels/gold_standard.json``, runs ``compile`` then ``compare``,
    and aggregates the per-case verdicts into a single JSON report
    plus an optional Markdown table.

    Use this to spot which cases regressed after a change, and which
    metrics consistently fail across the corpus.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import time

    from app.core.gold_compare import compare_to_gold

    if not cases_dir.is_dir():
        raise typer.BadParameter(f"cases_dir not found: {cases_dir}")

    case_dirs = sorted(p for p in cases_dir.iterdir() if p.is_dir())
    rows: list[dict] = []

    for case_dir in case_dirs:
        gold_path = case_dir / "labels" / "gold_standard.json"
        if not gold_path.is_file():
            # Cases without a gold are reported as "skipped: no gold".
            rows.append(
                {
                    "case": case_dir.name,
                    "status": "skipped",
                    "reason": "no gold_standard.json",
                    "pass_fraction": None,
                }
            )
            continue

        # Compile to a temporary file so we don't pollute case dirs.
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, prefix=f"matrix_{case_dir.name}_"
        ) as tmp:
            compiled_path = Path(tmp.name)

        cmd = [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(case_dir),
            "--out",
            str(compiled_path),
            "--skip-orbitbrief",
        ]
        if no_cache:
            cmd.append("--no-cache")

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "case": case_dir.name,
                    "status": "timeout",
                    "reason": f"exceeded {timeout_seconds}s",
                    "pass_fraction": None,
                    "duration_s": timeout_seconds,
                }
            )
            try:
                compiled_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        duration = time.monotonic() - start

        if proc.returncode != 0 or not compiled_path.is_file():
            rows.append(
                {
                    "case": case_dir.name,
                    "status": "compile_error",
                    "reason": (proc.stderr or proc.stdout)[:400],
                    "pass_fraction": None,
                    "duration_s": round(duration, 1),
                }
            )
            try:
                compiled_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
        compiled_payload = json.loads(compiled_path.read_text(encoding="utf-8"))
        try:
            compiled_path.unlink(missing_ok=True)
        except Exception:
            pass
        report = compare_to_gold(gold=gold_payload, compiled=compiled_payload)
        overall = report.get("overall", {})
        per_metric: dict[str, str] = {}
        fail_summary: list[str] = []
        for name, m in (report.get("metrics") or {}).items():
            verdict = m.get("verdict", "skipped")
            per_metric[name] = verdict
            if verdict == "fail":
                if "missing" in m and isinstance(m["missing"], list):
                    fail_summary.append(f"{name}({len(m['missing'])} missing)")
                elif "missing_count" in m:
                    fail_summary.append(f"{name}({m['missing_count']} missing)")
                elif "actual" in m:
                    fail_summary.append(
                        f"{name}({m['actual']}/{m.get('expected_min', '?')})"
                    )
                else:
                    fail_summary.append(name)
        rows.append(
            {
                "case": case_dir.name,
                "status": "compared",
                "pass_fraction": overall.get("pass_fraction", 0.0),
                "passed": overall.get("pass", 0),
                "total_checked": overall.get("total_checked", 0),
                "skipped": overall.get("skipped", 0),
                "duration_s": round(duration, 1),
                "fails": fail_summary,
                "per_metric": per_metric,
            }
        )

    # Aggregate
    pass_fractions = [r["pass_fraction"] for r in rows if r.get("pass_fraction") is not None]
    avg_pass_fraction = (
        sum(pass_fractions) / len(pass_fractions) if pass_fractions else 0.0
    )
    summary = {
        "total_cases": len(rows),
        "compared": sum(1 for r in rows if r["status"] == "compared"),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "timeouts": sum(1 for r in rows if r["status"] == "timeout"),
        "compile_errors": sum(1 for r in rows if r["status"] == "compile_error"),
        "average_pass_fraction": round(avg_pass_fraction, 4),
    }
    out_payload = {"summary": summary, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    if markdown_out is not None:
        markdown_lines = [
            "# Parser-OS Compatibility Matrix",
            "",
            f"**Total cases**: {summary['total_cases']}",
            f"**Compared**: {summary['compared']} / Skipped: {summary['skipped']} / Timeouts: {summary['timeouts']} / Compile errors: {summary['compile_errors']}",
            f"**Average pass-fraction (across compared cases)**: {summary['average_pass_fraction']:.0%}",
            "",
            "| Case | Status | Pass | Skipped | Duration | Failing metrics |",
            "|---|---|---:|---:|---:|---|",
        ]
        for r in rows:
            if r["status"] == "compared":
                pass_pct = f"{(r['pass_fraction'] or 0.0)*100:.0f}% ({r['passed']}/{r['total_checked']})"
                fails = ", ".join(r["fails"]) if r["fails"] else "—"
                duration = f"{r['duration_s']}s"
                markdown_lines.append(
                    f"| {r['case']} | ok | {pass_pct} | {r['skipped']} | {duration} | {fails} |"
                )
            elif r["status"] == "timeout":
                markdown_lines.append(
                    f"| {r['case']} | **TIMEOUT** | — | — | {r.get('duration_s', '?')}s | exceeded budget |"
                )
            elif r["status"] == "compile_error":
                markdown_lines.append(
                    f"| {r['case']} | **ERROR** | — | — | {r.get('duration_s', '?')}s | {r['reason'][:80]} |"
                )
            else:
                markdown_lines.append(
                    f"| {r['case']} | skipped | — | — | — | {r.get('reason','')} |"
                )
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    typer.echo(json.dumps(summary, indent=2))


@app.command("report")
def report(
    project_dir: Path = typer.Argument(..., help="Project directory to compile."),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        help="Where to write the production report bundle.  Will be created.",
    ),
    domain_pack: str | None = typer.Option(
        None,
        "--domain-pack",
        help="Pin a domain pack (otherwise auto-routed).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable incremental artifact cache reuse.",
    ),
    skip_orbitbrief: bool = typer.Option(
        False,
        "--skip-orbitbrief",
        help="Skip the OrbitBrief envelope and the per-compile review folder.",
    ),
    no_zip: bool = typer.Option(
        False,
        "--no-zip",
        help="Skip writing the <out_dir>.zip bundle.",
    ),
    abstain_threshold: float = typer.Option(0.70, "--abstain-threshold"),
    allow_errors: bool = typer.Option(False, "--allow-errors"),
    allow_unverified_receipts: bool = typer.Option(
        False, "--allow-unverified-receipts"
    ),
) -> None:
    """Generate a single production hand-off report for a project.

    Runs ``compile`` end-to-end, auto-detects ``labels/gold_standard.json``
    and runs ``compare`` against it, builds the OrbitBrief envelope, writes
    the per-compile review folder, then renders a 1-page executive
    ``REPORT.md`` summarizing what was produced and what needs review.
    Optionally bundles everything into a ZIP.

    Hand the resulting directory (or ZIP) to a tester / reviewer — they
    can read ``REPORT.md`` top-down without having to know the internal
    data shapes.
    """
    from app.core.production_report import build_production_report

    summary = build_production_report(
        project_dir=project_dir,
        out_dir=out_dir,
        domain_pack=domain_pack,
        no_cache=no_cache,
        skip_orbitbrief=skip_orbitbrief,
        zip_bundle=not no_zip,
        abstain_threshold=abstain_threshold,
        allow_errors=allow_errors,
        allow_unverified_receipts=allow_unverified_receipts,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def health() -> None:
    """Simple CLI health check."""
    typer.echo("ok")


if __name__ == "__main__":
    app()
