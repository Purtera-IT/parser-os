"""The main heads nightly: report parsing, held-out verdicts, and failing loudly (no torch needed)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ml.c3 import main_heads_report as rep

ROOT = Path(__file__).resolve().parents[1]


def test_fit_log_parsing():
    saved = ("rows train=433 validation=104\ncheckpoint col:label_type validation exact 0.512 (n=104)\n"
             "epoch 5 loss 1.0 validation exact 0.6\nsaved /x/refit_label_type.pt: col:label_type validation exact 0.512 -> 0.640\n")
    assert rep.parse_fit_log(saved) == {"status": "saved", "train": 433, "val": 104, "val_base": 0.512, "val_refit": 0.64}
    nogain = "rows train=5 validation=3\ncheckpoint col:supplier validation exact 0.667 (n=3)\nno epoch beat the checkpoint's 0.667; nothing saved\n"
    assert rep.parse_fit_log(nogain)["status"] == "no gain on validation; nothing saved"
    assert rep.parse_fit_log("no col:admission labels in train or validation; nothing to fit")["status"].startswith("no labels")
    assert rep.parse_fit_log("Traceback (most recent call last):\n boom")["status"] == "failed"


def test_report_has_every_section(tmp_path):
    (tmp_path / "queue").mkdir()
    (tmp_path / "rows").mkdir()
    (tmp_path / "queue" / "summary.json").write_text(json.dumps({
        "corrections": 3, "main_head_rows": 3, "trainable": 2, "deals": 1, "envelopes": "1/1",
        "by_head": {"col:label_type": {"person": 2, "skip:unresolved": 1}}, "no_route": {"read:address_level": "none"}}))
    (tmp_path / "rows" / "index.json").write_text(json.dumps({
        "heads": {"col:label_type": {"train": {"files": ["a"], "person": 2, "approved": 0, "draft": 1}}}, "counts": {}}))
    (tmp_path / "fit_label_type.log").write_text("no col:label_type labels in train or validation; nothing to fit")
    (tmp_path / "holdout.json").write_text(json.dumps({"files": 1, "heads": {"col:label_type": {
        "base": {"n": 4, "exact": 0.5, "majority": 0.75}, "refit": {"n": 4, "exact": 0.25}, "verdict": "refit WORSE on held-out"}}}))
    md = rep.to_md(rep.build(tmp_path, ["col:label_type"]), ["col:label_type"])
    for want in ("## 1 Queue", "## 2 Rows", "## 3 Refit", "## 4 Held-out", "refit WORSE on held-out",
                 "read:address_level", "2 (+1 draft)", "Report-only"):
        assert want in md, want


def test_heldout_verdicts():
    pytest.importorskip("torch")
    from ml.c3.main_heads_eval import compare
    out = compare({"a": {"n": 3, "exact": 0.5}, "b": {"n": 0}}, {"a": {"n": 3, "exact": 0.7}, "b": {"n": 0}})
    assert out["a"]["verdict"] == "refit better on held-out" and out["b"]["verdict"] == "no held-out labels"


@pytest.mark.skipif(shutil.which("zsh") is None, reason="the nightly is a zsh script")
def test_nightly_fails_loudly_on_missing_inputs(tmp_path):
    env = dict(os.environ, MAIN_HEADS_ROOT=str(tmp_path / "root"), NO_LOG="1", NO_REFIT="1",
               EXPORT_DIR=str(tmp_path / "missing-export"), PY="python3")
    r = subprocess.run(["zsh", str(ROOT / "scripts" / "main_heads_nightly.sh")], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3
    assert "FATAL main_heads_nightly: no corrections" in r.stdout


@pytest.mark.skipif(shutil.which("zsh") is None, reason="the export is a zsh script")
def test_export_fails_loudly_without_storage(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PY": "python3"}   # no az, no connection string, no SAS
    r = subprocess.run(["zsh", str(ROOT / "scripts" / "main_heads_export.sh"), str(tmp_path / "out")], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "FATAL main_heads_export: no storage access" in r.stdout
