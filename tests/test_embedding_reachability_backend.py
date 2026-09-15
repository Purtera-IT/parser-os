"""The store's reachability gate asks the embedding backend actually in use.

Live dev worker, 2026-09-15 from ~09:00: SOWSMITH_EMBED_BACKEND=azure, yet the
gate probed the tailnet Ollama's embed model; the Mac Studio was busy serving
a 46 GB vision model, the probe timed out, and every feedback-store lookup
abstained -- taught types, learned hours and judged documents all silently off
while Azure answered in under a second the whole time.
"""
import pytest

from app.core import embedding_retrieval as er
from app.core import ollama_host


@pytest.fixture(autouse=True)
def _azure(monkeypatch):
    monkeypatch.setenv("SOWSMITH_EMBED_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_BASE", "https://aoai.example/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_KEY", "k")
    monkeypatch.delenv("PARSER_OS_NO_LOCAL_MODELS", raising=False)
    er._REACH_CACHE.clear()
    yield
    er._REACH_CACHE.clear()


def test_with_azure_in_use_the_ollama_box_is_never_asked(monkeypatch):
    monkeypatch.setattr(ollama_host, "embed_ready", lambda *a, **k: pytest.fail("Ollama must not be probed"))
    calls = []
    monkeypatch.setattr(er, "_embed_azure", lambda texts: calls.append(texts) or [[0.1, 0.2]])
    assert er.embedding_endpoint_reachable() is True
    assert calls == [["probe"]]


def test_an_azure_failure_reads_as_unreachable_and_is_cached(monkeypatch):
    calls = []

    def _boom(texts):
        calls.append(1)
        raise TimeoutError("slow")

    monkeypatch.setattr(er, "_embed_azure", _boom)
    assert er.embedding_endpoint_reachable() is False
    assert er.embedding_endpoint_reachable() is False
    assert len(calls) == 1  # cached for the preflight TTL


def test_with_ollama_in_use_the_ollama_probe_still_decides(monkeypatch):
    monkeypatch.setenv("SOWSMITH_EMBED_BACKEND", "ollama")
    monkeypatch.setattr(er, "_embed_azure", lambda texts: pytest.fail("Azure must not be probed"))
    monkeypatch.setattr(ollama_host, "embed_ready", lambda host, model, **k: True)
    assert er.embedding_endpoint_reachable() is True
