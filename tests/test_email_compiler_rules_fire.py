"""Prove universal compiler stages fire on email-derived atoms."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.atom_substance_gate import apply_substance_gate
from app.core.atom_type_sanity import apply_type_sanity
from app.core.compiler import compile_project
from app.core.schemas import AtomType
from app.parsers.email_parser import EmailParser

_GECKO_EMAIL = "\n".join(
    [
        "From: patrick@purtera-it.com",
        "To: etroci@nmcms.com",
        "Subject: 010058 - Ubiquiti Configuration Gecko Robotics",
        "Date: 2026-06-24T18:18:29.931Z",
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "",
        "Eddie,",
        "",
        "By the end of the meeting customer clarified:",
        "Include:",
        "",
        "  *   Badge/access control setup",
        "  *   Okta integration",
        "",
        "Exclude:",
        "",
        "  *   Network buildout",
        "",
        "Customer specifically said:",
        '"Network build out does not need to be built into this."',
        "",
        "Thanks,",
        "Patrick Kelly",
    ]
)


def _write_gecko_eml(tmp_path: Path) -> Path:
    p = tmp_path / "010058-hs-email-111652731176.eml"
    p.write_text(_GECKO_EMAIL, encoding="utf-8")
    return p


def test_substance_gate_drops_label_lead_ins_keeps_list_items(tmp_path):
    atoms = EmailParser().parse_artifact("p", "art_email", _write_gecko_eml(tmp_path))
    kept, dropped = apply_substance_gate(atoms)
    dropped_texts = {a.raw_text.strip() for a in dropped}
    kept_sections = {
        (a.raw_text.strip(), a.value.get("list_section"))
        for a in kept
        if a.value.get("list_section")
    }
    assert "Customer specifically said:" in dropped_texts
    assert ("Okta integration", "include") in kept_sections
    assert any(t == "Network buildout" and s == "exclude" for t, s in kept_sections)


def test_type_sanity_preserves_email_include_list_items(tmp_path):
    atoms = EmailParser().parse_artifact("p", "art_email", _write_gecko_eml(tmp_path))
    # Simulate typed classifier promoting an include-list micro-label to task.
    for a in atoms:
        if a.raw_text.strip() == "Okta integration":
            a.atom_type = AtomType.task
    atoms, demoted, _surfaced = apply_type_sanity(atoms, project_id="p")
    okta = next(a for a in atoms if a.raw_text.strip() == "Okta integration")
    assert okta.atom_type == AtomType.scope_item
    assert demoted >= 1


@pytest.mark.parametrize("stage_name", [
    "substance_gate",
    "atom_type_sanity",
    "quote_context_head",
    "quote_line_head",
    "typed_atom_classification",
])
def test_compiler_stages_run_on_email_project(tmp_path, stage_name: str):
    eml = _write_gecko_eml(tmp_path)
    (tmp_path / "project.yaml").write_text("project_id: gecko_test\n", encoding="utf-8")

    seen: list[str] = []

    def on_stage_end(stage, _all_stages) -> None:
        seen.append(stage.stage_name)

    result = compile_project(
        tmp_path,
        project_id="gecko_test",
        use_cache=False,
        allow_unverified_receipts=True,
        stage_callback=on_stage_end,
    )
    assert eml.name in str(result.manifest.parser_routing)
    assert stage_name in seen, f"stage {stage_name!r} never ran; saw {seen}"

    email_atoms = [
        a for a in result.atoms
        if a.artifact_id and "email" in str(getattr(a.source_refs[0], "filename", "")).lower()
    ] or [a for a in result.atoms if a.value and a.value.get("kind") == "email_body_line"]
    assert email_atoms, "expected email body atoms in compile result"
    assert any(a.value.get("list_section") == "include" for a in email_atoms if isinstance(a.value, dict))
