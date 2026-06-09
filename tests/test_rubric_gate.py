"""Rubric gate runtime: the safe-abstain contract (no torch/model needed here).

Without a model dir present, every call must abstain (all-False) and never raise —
that's the guarantee that makes it safe to ship OFF/cold = byte-identical to the
LLM-only path.
"""
import os

from app.core import rubric_gate


def test_abstains_when_model_absent(monkeypatch, tmp_path):
    # point at an empty dir -> no model -> must abstain, never raise
    monkeypatch.setenv("SOWSMITH_RUBRIC_GATE_DIR", str(tmp_path / "nope"))
    rubric_gate._holder.clear()
    flags = rubric_gate.keep_deflect_flags(["some clause", "another clause"])
    assert flags == [False, False]
    assert rubric_gate.is_ready() is False


def test_empty_input():
    rubric_gate._holder.clear()
    assert rubric_gate.keep_deflect_flags([]) == []


def test_conf_bar_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_RUBRIC_GATE_CONF", raising=False)
    assert rubric_gate._conf_bar() == 0.97
    monkeypatch.setenv("SOWSMITH_RUBRIC_GATE_CONF", "0.9")
    assert rubric_gate._conf_bar() == 0.9
    monkeypatch.setenv("SOWSMITH_RUBRIC_GATE_CONF", "garbage")
    assert rubric_gate._conf_bar() == 0.97  # falls back, never raises
