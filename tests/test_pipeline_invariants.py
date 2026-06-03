"""End-to-end pipeline invariants.

These guard the *ordering* contracts that unit tests miss — the kind of
latent bug where a stage mints atoms after source_replay (so they reach
the quality gate with no receipts) or where a post-pipeline builder
references a helper that isn't in scope. Both shipped silently and only
blew up on a real compile; this file runs the whole pipeline on a tiny
fixture that exercises those exact paths, no LLM required.
"""

from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.orbitbrief_envelope import build_orbitbrief_envelope, envelope_to_markdown


def _make_geo_fixture(tmp_path: Path) -> Path:
    """A deal whose only locational anchor is a bare City, ST ZIP — this
    forces site_geo_fallback to mint a physical_site atom *after*
    source_replay, the exact condition that broke the receipt gate."""
    project = tmp_path / "geo_deal"
    project.mkdir()
    (project / "notes.md").write_text(
        "# Project Notes\n\n"
        "Replace approximately 110 displays at the resort.\n"
        "location Santa Fe, NM 87506\n"
        "Manual TV configuration, approx 15 mins per unit.\n",
        encoding="utf-8",
    )
    return project


def test_every_atom_with_source_refs_has_receipts(tmp_path: Path) -> None:
    """The receipt-backfill invariant: no atom reaches the end of the
    pipeline carrying source_refs but no receipts. Regression guard for
    the geo-fallback / late-stage atom-minting crash."""
    project = _make_geo_fixture(tmp_path)
    result = compile_project(project_dir=project, use_cache=False)

    offenders = [
        a.id for a in result.atoms
        if getattr(a, "source_refs", None) and not getattr(a, "receipts", None)
    ]
    assert not offenders, f"atoms with source_refs but no receipts: {offenders}"


def test_geo_fallback_produces_a_site(tmp_path: Path) -> None:
    """Confirms the fixture actually exercises the late atom-minting path
    (otherwise the receipt invariant above would pass vacuously)."""
    project = _make_geo_fixture(tmp_path)
    result = compile_project(project_dir=project, use_cache=False)
    site_atoms = [a for a in result.atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "physical_site"]
    assert site_atoms, "geo_fallback did not mint a physical_site atom"


def test_envelope_build_does_not_crash_on_physical_site(tmp_path: Path) -> None:
    """The site-attribute passthrough referenced an undefined helper and
    silently failed on every compile with a physical_site. Build the full
    envelope + markdown and assert no stage-failure warning leaks."""
    project = _make_geo_fixture(tmp_path)
    result = compile_project(project_dir=project, use_cache=False)

    envelope = build_orbitbrief_envelope(project_dir=project, compile_result=result)
    md = envelope_to_markdown(envelope)
    assert isinstance(md, str) and md

    # No stage should have silently failed.
    bad = [w for w in (result.warnings or []) if "failed:" in w or "_atom_type_str" in w]
    assert not bad, f"stage-failure warnings present: {bad}"


def test_no_stage_failure_warnings(tmp_path: Path) -> None:
    """Any 'WARNING: <stage> failed:' means a pipeline stage threw and was
    swallowed — a silent regression. The clean fixture must produce none."""
    project = _make_geo_fixture(tmp_path)
    result = compile_project(project_dir=project, use_cache=False)
    failures = [w for w in (result.warnings or []) if "failed:" in w]
    assert not failures, f"swallowed stage failures: {failures}"
