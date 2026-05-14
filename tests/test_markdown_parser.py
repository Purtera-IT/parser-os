"""Markdown parser regression tests.

Closes the largest blocker cluster from the corpus review: every
``*_managed_services_package.md`` was producing zero atoms because
parser-os had no Markdown extractor configured. After PR 1 these
files emit structured atoms with line-range locators so source replay
can verify them and brains can cite them.
"""
from __future__ import annotations

from pathlib import Path

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.parsers.markdown_parser import MarkdownParser


def test_managed_services_markdown_emits_atoms(tmp_path: Path):
    p = tmp_path / "CASE_managed_services_package.md"
    p.write_text(
        """# Scope Overview

Spring Lake Public Schools requires an execution-ready package for
copper cabling and low-voltage AV cabling.

## Scope Includes

- Quote includes 186 Belden Cat6 CMP drops with RJ45 termination.
- Provide Fluke certification per TIA-568.2-D.

## Exclusions

- Fire alarm work is excluded.
- Owner to provide ceiling access.

## Open Questions

- Confirm after-hours access window?
""",
        encoding="utf-8",
    )

    artifact_id = stable_id("art", str(p))
    out = MarkdownParser().parse_artifact(
        project_id="TEST_CASE",
        artifact_id=artifact_id,
        path=p,
    )

    assert len(out.atoms) >= 7, [a.atom_type.value for a in out.atoms]
    assert any(a.atom_type == AtomType.quantity for a in out.atoms)
    assert any(a.atom_type == AtomType.exclusion for a in out.atoms)
    assert any(a.atom_type == AtomType.open_question for a in out.atoms)
    assert all("line_start" in a.source_refs[0].locator for a in out.atoms)
    assert all("section_path" in a.source_refs[0].locator for a in out.atoms)


def test_markdown_extension_match(tmp_path: Path):
    p = tmp_path / "demo.md"
    p.write_text("# T\n\n- item 1\n", encoding="utf-8")
    m = MarkdownParser().match(p, sample_text=None, domain_pack=None)
    assert m.confidence > 0.9
    assert m.parser_name == "markdown"


def test_markdown_parser_registered_by_default():
    """The default parser registry must include the Markdown parser."""
    from app.parsers.registry import get_registered_parsers

    names = {p.capability.parser_name for p in get_registered_parsers()}
    assert "markdown" in names
