from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT.parent


def _compile_pack(pack_name: str, tmp_path: Path, *, llm: bool = False) -> dict:
    """Compile a fixture pack.

    By default LLM paths are disabled — CI runs without a model. Pass
    ``llm=True`` to exercise the LLM bridge + typed_atom_classifier;
    this is what catches regressions like ``OPTBOT-WEST-CAMPUS-V5``
    ghosts that only appear when the model is producing output. Skipped
    when no Ollama endpoint is reachable.
    """
    out = tmp_path / f"{pack_name}_envelope.json"
    env = os.environ.copy()
    if llm:
        # Strip pre-existing disable flags so every LLM-driven path runs.
        for k in (
            "SOWSMITH_MULTI_ENTITY_DISABLE",
            "SOWSMITH_SITE_LLM_DISABLE",
            "SOWSMITH_VISION_DISABLE",
            "SOWSMITH_TYPED_CLASSIFIER_DISABLE",
        ):
            env.pop(k, None)
        timeout = 1800
    else:
        env.update(
            {
                "SOWSMITH_MULTI_ENTITY_DISABLE": "1",
                "SOWSMITH_SITE_LLM_DISABLE": "1",
                "SOWSMITH_VISION_DISABLE": "1",
                "SOWSMITH_TYPED_CLASSIFIER_DISABLE": "1",
            }
        )
        timeout = 180
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(BUNDLE_ROOT / "test_deals" / pack_name),
            "--out",
            str(out),
            "--skip-orbitbrief",
            "--no-cache",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=timeout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(out.read_text())


def _atoms(envelope: dict, atom_type: str) -> list[dict]:
    return [a for a in envelope["atoms"] if a.get("atom_type") == atom_type]


def _expected_optbot_site_ids() -> set[str]:
    return {"ATL-HQ-01", "ATL-WEST-02", "ATL-AIR-03", "ATL-047-04", "ATL-CP-05"}


def test_optbot_pack_regression_counts_and_site_hygiene(tmp_path: Path):
    envelope = _compile_pack("optbot", tmp_path)
    counts = Counter(a["atom_type"] for a in envelope["atoms"])

    site_ids = {a["value"].get("site_id") or a["value"].get("id") for a in _atoms(envelope, "physical_site")}
    assert site_ids == _expected_optbot_site_ids()
    assert counts["risk"] <= 12
    assert counts["acceptance_criterion"] <= 12
    assert 5 <= counts["milestone_phase"] <= 7
    assert counts["bom_line"] >= 10
    assert counts["stakeholder"] >= 8
    assert not [
        a for a in envelope["atoms"]
        if isinstance(a.get("value"), dict) and a["value"].get("entity_type") == "site"
    ]

    bom_item_ids = {a["value"].get("item_id") for a in _atoms(envelope, "bom_line") if a["value"].get("item_id")}
    allocated_item_ids = {a["value"].get("item_id") for a in _atoms(envelope, "site_allocation") if a["value"].get("item_id")}
    assert bom_item_ids
    assert bom_item_ids <= allocated_item_ids


def test_optbot_pack_with_llm_no_ghost_sites(tmp_path: Path):
    """v57.2 — same site assertion but with LLM paths ENABLED.

    Skipped automatically when no Ollama endpoint is reachable (so CI
    without a model still passes the LLM-off variant). When Ollama is
    available, this is the real gate: typed_atom_classifier runs against
    the cell-bleed paragraph atoms from the OPTBOT site-roster PDF, and
    the assertion fails if any ``OPTBOT-XXX-V5`` / year-suffix ghost
    survives into the envelope.

    Set ``SOWSMITH_LLM_TEST=1`` to force-run, otherwise the test skips
    when no Ollama is detected. Locally: point ``OLLAMA_BASE_URL`` at a
    reachable model (e.g. ``http://localhost:11434``).
    """
    import pytest
    from app.core.typed_atom_classifier import _ollama_reachable

    if not _ollama_reachable() and not os.environ.get("SOWSMITH_LLM_TEST"):
        pytest.skip("Ollama not reachable; set SOWSMITH_LLM_TEST=1 to force-run")

    envelope = _compile_pack("optbot", tmp_path, llm=True)
    site_ids = {a["value"].get("site_id") or a["value"].get("id") for a in _atoms(envelope, "physical_site")}
    # The exact assertion as the LLM-off variant. No ghosts allowed.
    assert site_ids == _expected_optbot_site_ids(), (
        f"LLM produced ghost sites: {sorted(site_ids - _expected_optbot_site_ids())}"
    )


def test_aps_attachment_b_numeric_roster_regression(tmp_path: Path):
    from app.parsers.orbitbrief_pdf import _text_based_site_roster_extract

    sites = _text_based_site_roster_extract(
        pdf_path=BUNDLE_ROOT / "test_deals" / "aps_fiber" / "artifacts" / "APS_fiber_Attachment_B.pdf",
        project_id="aps_fiber",
        artifact_id="aps_attachment_b",
        parser_version="test",
        already_emitted=set(),
    )

    # The bundled Attachment B currently contains site_no 1..159 in the
    # extractable PDF text. The narrative brief says 132; this test trusts the
    # authoritative attachment rows rather than under-extracting to match the
    # stale narrative count.
    assert len(sites) == 159
    assert {int(a.value["site_no"]) for a in sites} == set(range(1, 160))
    required_fields = {"site_no", "administrative_site_name", "street", "city", "zip", "lat_long"}
    assert not [a for a in sites if any(not a.value.get(field) for field in required_fields)]
    assert not [a for a in sites if a.value.get("id") == a.value.get("address")]
    assert not [a for a in sites if "po box" in str(a.value.get("id", "")).lower()]
    assert not [a for a in sites if "albuquerque public schools" in str(a.value.get("id", "")).lower()]
