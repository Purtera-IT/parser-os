"""Tests for ML capability manifest and artifact bootstrap."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# ``app.learning.fetch_ml`` is deploy tooling: entrypoint.sh in the worker
# image downloads the ML artifacts and drops that module beside them. It has
# never existed in this repository -- no commit in history adds it -- so in a
# plain checkout these tests cannot pass, and reporting them as failures only
# padded the failure count that real regressions had to be spotted among.
#
# Production already treats the module as optional:
# ``ml_capabilities._artifacts_loaded`` catches the ImportError and reports no
# artifacts, which is the correct answer when none were fetched. Skipping here
# matches that contract instead of contradicting it.
requires_fetch_ml = pytest.mark.skipif(
    importlib.util.find_spec("app.learning.fetch_ml") is None,
    reason="app.learning.fetch_ml ships in the worker image, not in this repo",
)



@requires_fetch_ml
def test_artifact_self_check_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_ML_ARTIFACT_DIR", str(tmp_path))
    from app.learning.fetch_ml import artifact_self_check

    checks = artifact_self_check()
    assert "training_log" in checks
    assert checks["training_log"] is False


@requires_fetch_ml
def test_artifact_self_check_training_log(tmp_path, monkeypatch):
    db = tmp_path / "_training_deepseek.db"
    db.write_bytes(b"sqlite")
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", str(db))
    from app.learning.fetch_ml import artifact_self_check

    assert artifact_self_check()["training_log"] is True


def test_build_compile_capabilities_lite(monkeypatch):
    monkeypatch.delenv("SOWSMITH_ML_PROFILE", raising=False)
    monkeypatch.setenv("SOWSMITH_RUNTIME", "service")
    from app.core.ml_capabilities import build_compile_capabilities

    cap = build_compile_capabilities()
    assert cap["runtime"] == "service"
    assert cap["ml_profile"] == "lite"
    assert "env_active" in cap


def test_build_compile_capabilities_full(monkeypatch):
    monkeypatch.setenv("SOWSMITH_ML_PROFILE", "full")
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU", "1")
    monkeypatch.setenv("SOWSMITH_RUNTIME", "worker-warm")
    from app.core.ml_capabilities import build_compile_capabilities

    cap = build_compile_capabilities()
    assert cap["ml_profile"] == "full"
    assert cap["runtime"] == "worker-warm"
    assert "SOWSMITH_TYPE_HEAD_GPU" in cap["env_active"]


@requires_fetch_ml
def test_fetch_skip(monkeypatch):
    monkeypatch.setenv("SOWSMITH_ML_FETCH_SKIP", "1")
    from app.learning.fetch_ml import fetch_ml_artifacts

    report = fetch_ml_artifacts()
    assert report.get("skipped") is True
