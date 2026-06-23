"""GPU type-head runtime: the safe-abstain contract (no torch/model needed here).

Without a model dir present, every call must abstain (all-None) and never raise —
that's the guarantee that makes it safe to ship OFF/cold = byte-identical to the
LLM-only path.
"""
from app.core import type_head_gpu


def test_abstains_when_model_absent(monkeypatch, tmp_path):
    # point at an empty dir -> no model -> must abstain, never raise
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_DIR", str(tmp_path / "nope"))
    type_head_gpu._holder.clear()
    out = type_head_gpu.classify_batch(["some clause", "another clause"])
    assert out == [None, None]
    assert type_head_gpu.classify("lone clause") is None
    assert type_head_gpu.is_ready() is False


def test_empty_input():
    type_head_gpu._holder.clear()
    assert type_head_gpu.classify_batch([]) == []


def test_conf_bar_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_TYPE_HEAD_GPU_CONF", raising=False)
    assert type_head_gpu._conf_bar() == 0.85
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_CONF", "0.9")
    assert type_head_gpu._conf_bar() == 0.9
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_GPU_CONF", "garbage")
    assert type_head_gpu._conf_bar() == 0.85  # falls back, never raises
