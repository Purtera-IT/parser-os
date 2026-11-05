"""The compile must be able to say whether the heads had anything to load.

Before this, `ml_capabilities._artifacts_loaded()` imported a module that did
not exist, swallowed the ImportError and recorded `{}` — so a worker that
fetched no artifacts produced an envelope indistinguishable from a healthy one,
with no `c3_brief` and nothing saying why.
"""
import json
import os

import pytest

from app.learning.fetch_ml import artifact_self_check, configured_artifacts


@pytest.fixture()
def ml_dir(tmp_path, monkeypatch):
    d = tmp_path / "ml"
    d.mkdir()
    monkeypatch.setenv("SOWSMITH_ML_ARTIFACT_DIR", str(d))
    # Start from a clean slate: a stray SOWSMITH_*_DIR from the real environment
    # would otherwise be counted as a configured artifact.
    for key in [k for k in os.environ if k.startswith("SOWSMITH_") and k.endswith(("_DIR", "_DB", "_PATH"))]:
        if key != "SOWSMITH_ML_ARTIFACT_DIR":
            monkeypatch.delenv(key, raising=False)
    return d


def _receipt(d, **kw):
    (d / "_fetch_ml_receipt.json").write_text(json.dumps(kw))


def test_no_receipt_is_unknown_not_ok(ml_dir):
    """A missing receipt is never reported as a healthy fetch."""
    got = artifact_self_check()
    assert got["fetch_ml"] == "absent"
    assert got["artifacts"] == "unknown"
    assert "artifacts_missing" not in got


def test_boot_failure_is_reported_with_its_reason(ml_dir):
    _receipt(ml_dir, ok=False, error="downloaded 0 artifact files from ml-artifacts")
    got = artifact_self_check()
    assert got["fetch_ml"] == "failed"
    assert "downloaded 0 artifact files" in got["fetch_ml_error"]


def test_configured_artifact_present_reads_ok(ml_dir, monkeypatch):
    head = ml_dir / "_type_head_gpu" / "best"
    head.mkdir(parents=True)
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_DIR", str(head))
    _receipt(ml_dir, ok=True, downloaded=42)
    got = artifact_self_check()
    assert got["fetch_ml"] == "ok"
    assert got["fetch_ml_files"] == 42
    assert got["artifacts"] == "ok"
    assert got["artifacts_present"] == 1


def test_configured_artifact_missing_is_named(ml_dir, monkeypatch):
    """The whole point: the envelope says WHICH head had nothing to load."""
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_DIR", str(ml_dir / "_type_head_gpu" / "best"))
    _receipt(ml_dir, ok=True, downloaded=42)
    got = artifact_self_check()
    assert got["artifacts"] == "missing"
    assert got["artifacts_missing"] == ["SOWSMITH_TYPE_HEAD_GPU_DIR"]
    assert got["artifacts_present"] == 0


def test_runtime_created_caches_are_not_required(ml_dir, monkeypatch):
    """Caches are written by the runtime, so their absence is not a fault."""
    monkeypatch.setenv("SOWSMITH_EMBED_CACHE_DB", str(ml_dir / "_embed_cache.db"))
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(ml_dir / "_ocr_cache.db"))
    assert configured_artifacts() == {}


def test_head_registry_is_not_required(ml_dir, monkeypatch):
    """fetch_ml deliberately skips the registry unless FETCH_ML_INCLUDE_REGISTRY=1."""
    monkeypatch.setenv("SOWSMITH_HEAD_REGISTRY_DIR", str(ml_dir / "_head_registry"))
    assert configured_artifacts() == {}


def test_paths_outside_the_ml_dir_are_not_claimed(ml_dir, monkeypatch):
    """SOWSMITH_FEEDBACK_STORE_DB=/tmp/_feedback.db is not fetched into /tmp/ml."""
    monkeypatch.setenv("SOWSMITH_FEEDBACK_STORE_DB", "/tmp/_feedback.db")
    assert configured_artifacts() == {}


def test_dead_config_is_not_counted_as_a_failed_fetch(ml_dir, monkeypatch):
    """A path the blob store has no source for is dead config, not breakage.

    SOWSMITH_CALIBRATOR_PATH is set on dev AND prod while the only calibrator
    ever in ml-artifacts is `_quarantine/calibrator_leaked_20260804.joblib`
    (quarantined 2026-08-04) and nothing reads it. Counting it as a missing
    artifact made the worker's strict boot refuse to start for something no
    fetch could ever fix.
    """
    head = ml_dir / "_type_head_gpu" / "best"
    head.mkdir(parents=True)
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_DIR", str(head))
    monkeypatch.setenv("SOWSMITH_CALIBRATOR_PATH", str(ml_dir / "_calibrator" / "calibrator.joblib"))
    _receipt(ml_dir, ok=True, downloaded=42, no_source=["SOWSMITH_CALIBRATOR_PATH"])
    got = artifact_self_check()
    assert got["artifacts_dead_config"] == ["SOWSMITH_CALIBRATOR_PATH"]
    assert got["artifacts"] == "ok", "dead config does not make the fetch look broken"
    assert "artifacts_missing" not in got


def test_a_head_is_checked_for_its_path_not_its_variable_name(ml_dir, monkeypatch):
    """SOWSMITH_ADMISSION_REGISTRY is an artifact path with no _DIR/_DB/_PATH suffix.

    Matching name suffixes let it straight through the gate -- and it is the one
    artifact here with a live call path and no C3 replacement (`col:admission`
    is `no_labels`, replacing "parser: span_admission (lessons)"). What decides
    is the path, not the spelling of the variable.
    """
    monkeypatch.setenv("SOWSMITH_ADMISSION_REGISTRY", str(ml_dir / "_admission_heads"))
    assert "SOWSMITH_ADMISSION_REGISTRY" in configured_artifacts()
    _receipt(ml_dir, ok=True, downloaded=9)
    got = artifact_self_check()
    assert got["artifacts"] == "missing"
    assert got["artifacts_missing"] == ["SOWSMITH_ADMISSION_REGISTRY"]
