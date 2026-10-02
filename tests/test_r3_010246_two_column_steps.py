"""A two-column numbered step page reads one whole step per atom.

Deal 010246's install guide (re-run on #268), page 5: two columns of
numbered steps, the step number set in a large face to the left of a step
that wraps over three or four lines. #268 glued a free-standing number to
the sentence on ITS line -- but a number centred on a wrapped step sits
beside a middle line, so the step was cut in two ("Flip the two breakers" /
"1 marked PUMP to the off position..."). And with the number column set
further left than three characters, the numbers stayed a column of figures,
the page read as a four-column table and both columns of steps were
interleaved row by row. Now the marker is glued to the first line of the
block it heads, and a 1, 2, 3 ... rail of markers reaches its text across a
wide number column.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

STEPS = [
    "Flip the two breakers marked PUMP to the off position before you begin any work on the enclosure.",
    "Loosen the turnbuckles on both guy wires until the mast can be lowered safely by hand.",
    "Mount the camera bracket to the mast using the four supplied stainless steel bolts.",
    "Route the power cable through the weatherproof gland and tighten the gland nut firmly.",
    "Connect the cellular modem antenna to the port labeled ANT on the bottom of the unit.",
    "Raise the mast back to vertical and retension the turnbuckles evenly on both sides.",
    "Restore power at the breakers and wait for the status light to turn solid green.",
    "Confirm the live image in the portal and record the serial number on the install sheet.",
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _wrap(t: str, n: int = 24) -> list[str]:
    out, cur = [], ""
    for w in t.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    out.append(cur)
    return out


def _build(path: Path, gap: int, centered: bool) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "INSTALLATION STEPS", fontsize=16, fontname="hebo")
    for x, steps, n0 in ((72, STEPS[:4], 1), (330, STEPS[4:], 5)):
        y = 110
        for k, s in enumerate(steps):
            lines = _wrap(s)
            h = 13 * len(lines)
            page.insert_text((x, y + (h / 2 if centered else 0) + 2), str(n0 + k), fontsize=20, fontname="hebo")
            for i, ln in enumerate(lines):
                page.insert_text((x + gap, y + 10 + 13 * i), ln, fontsize=10, fontname="helv")
            y += h + 40
    doc.save(path)


@pytest.mark.parametrize("gap,centered", [(26, True), (60, True), (90, True), (90, False)])
def test_each_step_is_one_whole_atom(tmp_path: Path, gap: int, centered: bool) -> None:
    path = tmp_path / "Install Guide.pdf"
    _build(path, gap, centered)
    out = OrbitBriefPdfParser().parse_artifact("p", "a", path)
    atoms = out if isinstance(out, list) else out.atoms
    texts = [" ".join(a.raw_text.split()) for a in atoms]
    for n, step in enumerate(STEPS, 1):
        assert any(t == f"{n} {step}" for t in texts), (n, texts)
    # never a line from both columns in one atom
    for t in texts:
        assert not ("Flip" in t and "Connect" in t), t


SHORT = [
    "Flip the two breakers marked PUMP to the off position before work.",
    "Loosen the turnbuckles on both guy wires until the mast lowers.",
    "Mount the camera bracket with the four stainless steel bolts.",
    "Route the power cable through the weatherproof gland firmly.",
    "Connect the cellular antenna to the port labeled ANT below.",
]


def test_figure_lines_through_steps_are_not_a_table(tmp_path: Path) -> None:
    """A figure frame ruled across the step text was read as a table by
    find_tables(strategy="lines"), and its walls cut the words: "breakers
    ma: rked PUMP", "camera brack | et". A grid drawn through words is not a
    table; the steps read whole."""
    path = tmp_path / "Install Guide.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "STEP DETAILS", fontsize=14, fontname="hebo")
    y = 100
    for i, s in enumerate(SHORT, 1):
        page.insert_text((80, y), f"{i}. {s}", fontsize=10, fontname="helv")
        y += 30
    for x in (70, 200, 330, 460, 560):
        page.draw_line((x, 85), (x, y - 15))
    for yy in range(85, y, 30):
        page.draw_line((70, yy), (560, yy))
    page.draw_line((70, y - 15), (560, y - 15))
    doc.save(path)
    out = OrbitBriefPdfParser().parse_artifact("p", "a", path)
    atoms = out if isinstance(out, list) else out.atoms
    texts = [" ".join(a.raw_text.split()) for a in atoms]
    for step in SHORT:
        assert any(step in t for t in texts), (step, texts)
    assert not any("ma: rked" in t or "brack |" in t for t in texts), texts
