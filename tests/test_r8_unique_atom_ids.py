"""No compile emits two atoms with the same id.

Real case (010353, structure only): an intake markdown "Job site" line holding
only a street address, typed physical_site with no site id. semantic_dedup's
physical-site pass demoted it to deal_metadata (it names no site) and kept it;
with no site left standing, its "never leave a deal site-blind" net then picked
the same atom again and appended it a second time. The envelope held two atoms
with one id, one locator and two label keys.
"""
from __future__ import annotations

import collections
import shutil
from pathlib import Path

import pytest

from app.core.atom_id_invariant import DuplicateAtomIdError, enforce_unique_atom_ids
from app.core.compiler import compile_project
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import _dedupe_physical_site_atoms

REPO = Path(__file__).resolve().parents[1]


def _atom(aid: str, art: str, text: str, atype: AtomType, value: dict, locator: dict) -> EvidenceAtom:
    ref = SourceRef(id=f"src_{aid}", artifact_id=art, artifact_type=ArtifactType.txt, filename=f"{art}.md",
                    locator=locator, extraction_method="t", parser_version="t")
    return EvidenceAtom(id=aid, project_id="p", artifact_id=art, atom_type=atype, raw_text=text,
                        normalized_text=text.lower(), value=value, entity_keys=[], source_refs=[ref],
                        authority_class=AuthorityClass.machine_extractor, confidence=0.8, review_flags=[],
                        review_status=ReviewStatus.auto_accepted, parser_version="t")


def _job_site_line() -> EvidenceAtom:
    return _atom(
        "atm_job_site", "art_md", "4200 State Route 9, Riverton, OH 45001", AtomType.physical_site,
        {"address": "4200 State Route 9", "city": "Riverton", "state": "OH", "zip": "45001"},
        {"line_start": 13, "line_end": 13, "section_path": ["Customer", "Job site"],
         "block_kind": "paragraph", "block_index": 6},
    )


def _ids(atoms) -> collections.Counter:
    return collections.Counter(a.id for a in atoms)


def test_demoted_job_site_line_goes_out_once() -> None:
    job = _job_site_line()
    other = _atom("atm_other", "art_md", "Height requirement: 30 ft", AtomType.constraint, {}, {"line_start": 14})
    out = _dedupe_physical_site_atoms([job, other])
    assert _ids(out) == {"atm_job_site": 1, "atm_other": 1}
    # It keeps its words and loses only the site claim, as before.
    assert job.atom_type == AtomType.deal_metadata


def test_name_only_site_folded_onto_the_line_does_not_double_it() -> None:
    job = _job_site_line()
    name = _atom("atm_name", "art_json", "site.name: Riverton Yard", AtomType.physical_site,
                 {"name": "Riverton Yard"}, {"kind": "json_value", "json_pointer": "/site/name"})
    out = _dedupe_physical_site_atoms([job, name])
    assert all(n == 1 for n in _ids(out).values()), _ids(out)


def test_invariant_raises_when_strict() -> None:
    job = _job_site_line()
    with pytest.raises(DuplicateAtomIdError):
        enforce_unique_atom_ids([job, job], strict=True)


def test_invariant_in_production_drops_a_repeat_and_keeps_a_collision() -> None:
    job = _job_site_line()
    out, warnings = enforce_unique_atom_ids([job, job], strict=False)
    assert out == [job] and len(warnings) == 1
    twin = _job_site_line()
    twin.raw_text = "a different line"
    out, warnings = enforce_unique_atom_ids([job, twin], strict=False)
    assert out == [job, twin] and "different atoms share it" in warnings[0]


def _compile_inputs() -> list[tuple[str, Path]]:
    cases = [("demo_project", REPO / "tests" / "fixtures" / "demo_project")]
    for case in sorted((REPO / "real_data_cases").glob("*/")):
        src = case / "artifacts" if (case / "artifacts").is_dir() else case
        if any(p.is_file() and p.suffix.lower() not in {".md", ".json"} for p in src.iterdir()):
            cases.append((case.name, src))
    return cases


@pytest.mark.parametrize("name,src", _compile_inputs(), ids=lambda v: v if isinstance(v, str) else "")
def test_compile_emits_unique_atom_ids(name: str, src: Path, tmp_path: Path) -> None:
    proj = tmp_path / name
    proj.mkdir()
    for p in src.iterdir():
        if p.is_file() and (name == "demo_project" or p.suffix.lower() not in {".md", ".json"}):
            shutil.copy(p, proj / p.name)
    # Strict mode (conftest) makes compile itself raise on a duplicate; the
    # assertion below says which ids if it ever stops being strict.
    result = compile_project(project_dir=proj, project_id=name, allow_errors=True,
                             allow_unverified_receipts=True, use_cache=False)
    dups = {k: v for k, v in _ids(result.atoms).items() if v > 1}
    assert result.atoms and not dups


def test_generated_demo_fixtures_emit_unique_atom_ids(demo_project: Path) -> None:
    result = compile_project(project_dir=demo_project, project_id="demo_project", allow_errors=True,
                             allow_unverified_receipts=True, use_cache=False)
    dups = {k: v for k, v in _ids(result.atoms).items() if v > 1}
    assert result.atoms and not dups
