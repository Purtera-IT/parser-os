"""Tests for the deal-level census reconciliation helpers.

``reconciled_census`` folds the independent per-file region inventory across an
entire deal into one census and reconciles it against the emitted atoms, so
``uncovered()`` is the never-detected denominator that
``complaint_router.route`` consumes for its NEEDS_EXTRACTOR bucket.
``compiler.project_census`` is the thin entry point that reuses the compiler's
own ``_iter_artifacts`` discovery, guaranteeing the census denominator is drawn
from the exact file set the deal compiled from.
"""

from __future__ import annotations

from pathlib import Path

from app.core.content_census import CoverageStatus
from app.parsers.census import reconciled_census


class _Atom:
    """Minimal EvidenceAtom stand-in for reconciliation (mirrors the census tests)."""

    def __init__(self, raw_text: str = "", value: dict | None = None) -> None:
        self.raw_text = raw_text
        self.value = value or {}


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_reconciled_census_covers_lines_present_in_atoms(tmp_path: Path) -> None:
    f = _write(tmp_path / "scope.txt",
               "Replace 110 existing TVs across the lobby.\n"
               "Hidden contact: yonah sapir, site lead.\n")
    census = reconciled_census(
        [f],
        [_Atom(raw_text="The plan is to replace 110 existing TVs across the lobby.")],
    )
    statuses = {r.text[:20]: census.status(rid)
                for rid, r in census.regions.items()}
    # The TV line is represented by an atom → COVERED; the contact line is not.
    assert statuses["Replace 110 existing"] is CoverageStatus.COVERED
    assert statuses["Hidden contact: yona"] is CoverageStatus.UNCOVERED


def test_reconciled_census_uncovered_is_the_never_detected_set(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.txt", "alpha line one\nbeta line two\ngamma line three\n")
    census = reconciled_census([f], [_Atom(raw_text="alpha line one matched")])
    uncovered_texts = sorted(r.text for r in census.uncovered())
    assert uncovered_texts == ["beta line two", "gamma line three"]
    assert not census.invariant_ok()


def test_reconciled_census_combines_multiple_files(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "one.txt", "site located in austin texas\n")
    f2 = _write(tmp_path / "two.txt", "payment net 30 days\n")
    census = reconciled_census(
        [f1, f2],
        [_Atom(raw_text="the site located in austin texas is primary")],
    )
    # Regions from BOTH files are present in one census.
    arts = {r.artifact for r in census.regions.values()}
    assert arts == {"one", "two"}
    uncovered = [r.text for r in census.uncovered()]
    assert uncovered == ["payment net 30 days"]


def test_reconciled_census_skips_bad_paths_and_never_raises(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.txt", "real content here\n")
    missing = tmp_path / "does_not_exist.txt"
    census = reconciled_census([missing, good], [_Atom(raw_text="nothing matches")])
    # The missing file is skipped; the good file still inventoried.
    assert any(r.text == "real content here" for r in census.regions.values())


def test_reconciled_census_empty_atoms_marks_everything_uncovered(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.txt", "one\ntwo\n")
    census = reconciled_census([f], [])
    assert len(census.uncovered()) == 2


def test_project_census_reuses_iter_artifacts(tmp_path: Path) -> None:
    from app.core.compiler import project_census

    proj = tmp_path / "deal"
    (proj / "artifacts").mkdir(parents=True)
    _write(proj / "artifacts" / "scope.txt",
           "deploy 12 access points\nuncovered orphan requirement\n")
    census = project_census(proj, [_Atom(raw_text="we deploy 12 access points total")])
    uncovered = [r.text for r in census.uncovered()]
    assert uncovered == ["uncovered orphan requirement"]
    assert census.artifact == "deal"
