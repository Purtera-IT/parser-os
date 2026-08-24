"""End-to-end gold-compare against the synthetic ``SCHEMATIC_*`` corpus.

Each case under ``real_data_cases/SCHEMATIC_*`` ships a single PDF
plus a ``gold_standard.json``.  This test compiles every case,
projects the result through the gold-compare metrics, and asserts
that every metric the gold file declares grades ``pass``.

Synthetic fixtures stand in for real customer drawings — they
cover the same atom families (legend, target set, detections,
warnings, rooms, keyed notes, schedules, line runs, sheet metadata)
so a regression in any one of those surfaces as a failed metric.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from app.core.gold_compare import compare_to_gold
from app.core.ids import stable_id
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "real_data_cases"
CASE_IDS = [
    "SCHEMATIC_LV_FLOORPLAN",
    "SCHEMATIC_SECURITY_RISER",
    "SCHEMATIC_FIRE_RISER",
    "SCHEMATIC_ELECTRICAL_ONELINE",
    "SCHEMATIC_LV_RASTER_SCAN",
]


def _compile_case(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = CASES_DIR / case_id
    pdf_path = case_dir / "artifacts" / "drawings.pdf"
    gold_path = case_dir / "labels" / "gold_standard.json"
    if not pdf_path.is_file() or not gold_path.is_file():
        pytest.skip(f"case {case_id} not built; run scripts/_build_schematic_gold_corpus.py")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pack_id = gold.get("recommended_domain_pack") or "security_camera"
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack(pack_id)
    art = stable_id("art", str(pdf_path))
    out = parser.parse_artifact("proj_test", art, pdf_path, domain_pack=pack)
    compiled = {
        "project_id": "proj_test",
        "atoms": [a.model_dump(mode="json") for a in out.atoms],
        "edges": [],
        "packets": [],
    }
    return gold, compiled


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_schematic_gold_case_passes_every_declared_metric(case_id: str) -> None:
    gold, compiled = _compile_case(case_id)
    result = compare_to_gold(gold=gold, compiled=compiled)
    failed = [
        (name, metric)
        for name, metric in result["metrics"].items()
        if metric.get("verdict") == "fail"
    ]
    assert not failed, (
        f"{case_id} failed gold metrics:\n"
        + "\n".join(f"  {name}: {metric}" for name, metric in failed)
    )


def test_every_schematic_case_dir_has_artifacts_and_labels() -> None:
    """Labels are committed; the drawings they describe are not.

    ``.gitignore`` ignores ``real_data_cases/*/artifacts/*`` and explicitly
    un-ignores ``real_data_cases/*/labels/*.json`` -- the drawings are real
    customer documents and stay out of the repository by design, while the gold
    labels that describe them are tracked.

    So the label half of this check is a real invariant and is asserted
    unconditionally. The artifact half can only hold where someone has synced
    the case data locally, and asserting it in a clean checkout reported a
    deliberate policy as a failure. It is checked when the drawings are present
    and skipped, with the reason, when they are not.
    """
    for case_id in CASE_IDS:
        case_dir = CASES_DIR / case_id
        assert (case_dir / "labels" / "gold_standard.json").is_file(), case_id

    missing = [
        case_id
        for case_id in CASE_IDS
        if not (CASES_DIR / case_id / "artifacts" / "drawings.pdf").is_file()
    ]
    if missing:
        pytest.skip(
            "schematic drawings are gitignored (real customer documents); "
            f"not synced locally for: {', '.join(sorted(missing))}"
        )


def test_schematic_corpus_has_at_least_five_cases() -> None:
    assert len(CASE_IDS) >= 5
