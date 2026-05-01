from __future__ import annotations

import json
from pathlib import Path

import typer

from app.core.compiler import compile_project

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


@app.command()
def health() -> None:
    """Simple CLI health check."""
    typer.echo("ok")


if __name__ == "__main__":
    app()
