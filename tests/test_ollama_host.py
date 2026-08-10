"""The host the worker configures must be the host the code dials, and a
health check must prove the thing the caller is about to ask for.

The values here are the dev worker's real environment: it sets
``OLLAMA_BASE_URL`` to the mac proxy and ``OLLAMA_EMBED_URL`` to the embedding
endpoint, and sets no ``OLLAMA_HOST`` at all.
"""
from __future__ import annotations

import json

import pytest

from app.core import ollama_host

PROXY = "https://ollama-mac-proxy-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io"
EMBED = "https://ollama.ollamapurpulse.com/api/embed"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("OLLAMA_HOST", "OLLAMA_BASE_URL", "OLLAMA_EMBED_URL",
                "SOWSMITH_DISABLE_LLM", "PARSER_OS_LLM_PREFLIGHT_TTL"):
        monkeypatch.delenv(var, raising=False)
    ollama_host.reset_preflight_cache()
    yield
    ollama_host.reset_preflight_cache()


# ── host resolution ──────────────────────────────────────────────────────

def test_the_workers_base_url_is_used_when_no_host_is_set(monkeypatch):
    """The bug this module exists for: the worker sets OLLAMA_BASE_URL and
    nothing read it, so every LLM path dialled a hardcoded Tailscale address
    from inside Azure."""
    monkeypatch.setenv("OLLAMA_BASE_URL", PROXY)
    assert ollama_host.resolve_host() == PROXY


def test_explicit_host_wins_over_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", PROXY)
    assert ollama_host.resolve_host() == "http://localhost:11434"


def test_callers_own_default_survives_when_nothing_is_configured():
    assert ollama_host.resolve_host("http://box:11434") == "http://box:11434"
    assert ollama_host.resolve_host() == ollama_host.FALLBACK_HOST


def test_blank_env_is_not_a_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "   ")
    monkeypatch.setenv("OLLAMA_BASE_URL", PROXY)
    assert ollama_host.resolve_host() == PROXY


def test_trailing_slash_is_stripped_so_paths_do_not_double_up(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", PROXY + "/")
    assert ollama_host.resolve_host() == PROXY


def test_embed_url_is_trimmed_back_to_a_base(monkeypatch):
    """OLLAMA_EMBED_URL is configured as a full endpoint while every call
    site builds its own path."""
    monkeypatch.setenv("OLLAMA_EMBED_URL", EMBED)
    assert ollama_host.resolve_embed_host() == "https://ollama.ollamapurpulse.com"


def test_embed_falls_back_to_the_general_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", PROXY)
    assert ollama_host.resolve_embed_host() == PROXY


# ── preflight ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status = body.encode(), status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_model_that_answers_is_ready(monkeypatch):
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps({"response": "pong"})))
    assert ollama_host.generation_ready(PROXY, "qwen2.5:3b") is True


def test_a_wedged_model_is_not_ready(monkeypatch):
    """The real failure: /api/tags answers in milliseconds from a host whose
    generate endpoint never returns."""
    def _timeout(*a, **k):
        raise TimeoutError("The read operation timed out")
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _timeout)
    assert ollama_host.generation_ready(PROXY, "qwen3:14b") is False


def test_a_200_with_no_text_is_not_ready(monkeypatch):
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps({"response": "  "})))
    assert ollama_host.generation_ready(PROXY, "qwen3:14b") is False


def test_the_verdict_is_cached_so_a_compile_probes_once(monkeypatch):
    calls = []

    def _once(*a, **k):
        calls.append(1)
        return _Resp(json.dumps({"response": "pong"}))
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _once)
    for _ in range(5):
        ollama_host.generation_ready(PROXY, "qwen2.5:3b")
    assert len(calls) == 1


def test_each_model_is_probed_separately(monkeypatch):
    """One wedged model on a host says nothing about the others — today
    qwen3:14b times out on the same box that serves qwen2.5:3b in 0.41s."""
    seen = []

    def _by_model(req, *a, **k):
        model = json.loads(req.data)["model"]
        seen.append(model)
        if model == "qwen3:14b":
            raise TimeoutError("wedged")
        return _Resp(json.dumps({"response": "pong"}))
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _by_model)
    assert ollama_host.generation_ready(PROXY, "qwen3:14b") is False
    assert ollama_host.generation_ready(PROXY, "qwen2.5:3b") is True
    assert seen == ["qwen3:14b", "qwen2.5:3b"]


def test_the_kill_switch_dials_nothing(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("SOWSMITH_DISABLE_LLM must not reach the network")
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _boom)
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    assert ollama_host.generation_ready(PROXY, "qwen2.5:3b") is False


def test_the_preflight_is_bounded_well_below_the_call_timeout(monkeypatch):
    """180s per call is what makes a wedged host look like a hang. The probe
    exists to find that out cheaply, so it must stay far cheaper."""
    captured = {}

    def _capture(req, timeout=None, *a, **k):
        captured["timeout"] = timeout
        return _Resp(json.dumps({"response": "pong"}))
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _capture)
    ollama_host.generation_ready(PROXY, "qwen2.5:3b")
    assert captured["timeout"] <= 30


# ── embed preflight ──────────────────────────────────────────────────────

def test_embed_and_generate_are_separate_verdicts(monkeypatch):
    """A box that cannot generate may still embed, and vice versa. The generate
    fix missed this, and an ungated embed path is what put one deal in
    enrich_entities for 21 minutes against a dead endpoint."""
    seen = []

    def _by_endpoint(req, timeout=None, *a, **k):
        seen.append(req.full_url)
        if req.full_url.endswith("/api/generate"):
            raise TimeoutError("wedged")
        return _Resp(json.dumps({"embedding": [0.1, 0.2, 0.3]}))
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _by_endpoint)
    assert ollama_host.generation_ready(PROXY, "qwen3:14b") is False
    assert ollama_host.embed_ready(PROXY, "qwen3-embedding:8b") is True
    assert any(u.endswith("/api/embeddings") for u in seen)


def test_a_dead_embed_endpoint_is_not_ready(monkeypatch):
    def _timeout(*a, **k):
        raise TimeoutError("The read operation timed out")
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _timeout)
    assert ollama_host.embed_ready(PROXY, "qwen3-embedding:8b") is False


def test_a_200_with_no_vector_is_not_ready(monkeypatch):
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps({"embedding": []})))
    assert ollama_host.embed_ready(PROXY, "m") is False


def test_the_embed_verdict_is_cached_too(monkeypatch):
    calls = []

    def _once(*a, **k):
        calls.append(1)
        return _Resp(json.dumps({"embedding": [1.0]}))
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _once)
    for _ in range(4):
        ollama_host.embed_ready(PROXY, "m")
    assert len(calls) == 1


def test_the_kill_switch_covers_embeddings(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("SOWSMITH_DISABLE_LLM must not reach the network")
    monkeypatch.setattr(ollama_host.urllib.request, "urlopen", _boom)
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    assert ollama_host.embed_ready(PROXY, "m") is False
