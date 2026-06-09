"""GPU span tagger runtime: safe-abstain contract (no torch/models needed).

OFF or model-absent -> every call abstains (empty / no skip) and never raises, so
it's byte-identical to the CPU/LLM path. Also confirms augment_enrich_results still
works (CPU fallback) when the GPU module abstains.
"""
import types

from app.core import span_tagger_gpu


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_SPAN_GPU", raising=False)
    span_tagger_gpu._cache.clear()
    assert span_tagger_gpu.enabled() is False
    assert span_tagger_gpu.gpu_skip_relations() == {}
    assert span_tagger_gpu.has("requirements") is False
    assert span_tagger_gpu.gpu_admit([], "requirements") == []


def test_abstains_when_models_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_SPAN_GPU", "1")
    monkeypatch.setenv("SOWSMITH_SPAN_GPU_DIR", str(tmp_path / "nope"))
    span_tagger_gpu._cache.clear()
    assert span_tagger_gpu.gpu_skip_relations() == {}      # no models -> nothing skippable
    assert span_tagger_gpu.has("requirements") is False
    a = types.SimpleNamespace(raw_text="Provider shall install cabling")
    assert span_tagger_gpu.gpu_admit([a], "requirements") == []


def test_non_verbatim_relation_never_admits(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SPAN_GPU", "1")
    span_tagger_gpu._cache.clear()
    assert span_tagger_gpu.gpu_admit([], "commercial_line_items") == []
    assert span_tagger_gpu.has("commercial_line_items") is False


def test_augment_falls_back_to_cpu_when_gpu_off(monkeypatch):
    """With GPU off and no CPU heads, augment is a clean no-op (returns 0, no raise)."""
    from app.core import span_extractor
    monkeypatch.delenv("SOWSMITH_SPAN_GPU", raising=False)
    monkeypatch.setenv("SOWSMITH_SPAN_AUGMENT", "1")
    results = {}
    n = span_extractor.augment_enrich_results(results, atoms=[])
    assert n == 0  # nothing to add, never raised
