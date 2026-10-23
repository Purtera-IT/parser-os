"""Step 0 (regex_retirement row 47): the held headings are read by the heads.

compiler.py holds heading atoms (_is_heading_atom) out of every rule stage;
the deal's read must still hold them (c3_type_base.also_read), as it holds the
admission-held chatter.
"""
from __future__ import annotations

from pathlib import Path

from app.core import c3_type_base, compiler


def test_held_headings_are_passed_to_also_read(demo_project: Path, monkeypatch) -> None:
    held: list = []

    def _heading(atom) -> bool:  # the first atom the compile asks about is "a heading"
        if not held:
            held.append(atom)
        return atom is held[0]

    monkeypatch.setattr(compiler, "_is_heading_atom", _heading)
    monkeypatch.setattr(c3_type_base, "enabled", lambda: True)
    monkeypatch.setattr(c3_type_base, "set_deal", lambda *a, **k: None)
    monkeypatch.setattr(c3_type_base, "deal_documents", lambda *a, **k: [])
    got: list = []
    monkeypatch.setattr(c3_type_base, "also_read", lambda atoms: got.append(list(atoms)))
    compiler.compile_project(project_dir=demo_project, project_id="demo_project")
    assert held and got, "no heading held, or the deal's read was never given its extra lines"
    assert any(a is held[0] for a in got[0])
