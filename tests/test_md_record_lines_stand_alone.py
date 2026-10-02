"""An intake form's record lines are their own blocks, not wrapped prose.

Shape of a real intake request .md (deal 010353): under "## Job site" the
address and the next field sit on consecutive lines, and under "## Request
summary" a sentence is followed by a "Label: value" field. Joined as one
paragraph, the address line never stood as an atom of its own -- it came out
glued to the next field ("<address> Height requirement: 30 ft"). Text below is
synthetic; only the structure mirrors the real file.
"""

from __future__ import annotations

from app.parsers.markdown_parser import MarkdownParser, _iter_markdown_blocks

ADDRESS = "4410 State Route 9, Cedarville, OH 45314"

INTAKE_MD = f"""# Ridge Yard
**Request:** Install · High Priority
## Request summary
Install one (1) PTZ camera and one (1) solar unit mounted on a 20 foot mast at the Ridge Yard site.
Requested completion: 2026-11-02
## Job site
{ADDRESS}
Height requirement: 20 ft
## Dispatch brief
Please provide a quote for the installation at Ridge Yard located at {ADDRESS}.
Please review the site access notes before scheduling.
"""


def _paragraphs(md: str):
    return [(b.text, b.line_start, b.line_end, b.section_path[-1])
            for b in _iter_markdown_blocks(md) if b.block_kind == "paragraph"]


def test_job_site_address_line_is_its_own_atom(tmp_path):
    p = tmp_path / "INTAKE_REQUEST.md"
    p.write_text(INTAKE_MD, encoding="utf-8")
    atoms = MarkdownParser().parse_artifact("p", "a", p).atoms
    hits = [a for a in atoms if a.raw_text == ADDRESS]
    assert hits, [a.raw_text for a in atoms]
    loc = hits[0].source_refs[0].locator
    assert (loc["line_start"], loc["line_end"]) == (7, 7)
    assert loc["section_path"][-1] == "Job site"
    assert not [a for a in atoms if ADDRESS in a.raw_text and "Height" in a.raw_text]


def test_field_lines_split_from_the_prose_beside_them():
    paras = _paragraphs(INTAKE_MD)
    texts = [t for t, *_ in paras]
    assert "Requested completion: 2026-11-02" in texts
    assert "Height requirement: 20 ft" in texts
    assert ("Install one (1) PTZ camera and one (1) solar unit mounted on a 20 foot mast "
            "at the Ridge Yard site.") in texts


def test_prose_that_mentions_an_address_still_wraps_as_one_paragraph():
    md = ("## Notes\n\nThe crew meets at the gate of\n"
          f"{ADDRESS} and checks in\nwith the yard office before starting.\n")
    assert [t for t, *_ in _paragraphs(md)] == [
        f"The crew meets at the gate of {ADDRESS} and checks in with the yard office before starting."
    ]
