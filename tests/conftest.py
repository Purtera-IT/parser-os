from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.make_demo_fixtures import create_demo_project


@pytest.fixture(autouse=True)
def _offline_llm_by_default(monkeypatch) -> None:
    """Keep the suite from depending on a box being up.

    Every LLM path in ``app.core`` reaches ``OLLAMA_HOST``, which defaults to
    a hardcoded Tailscale address, with a 180-second per-call timeout. When
    that host is wedged — accepting connections, answering ``/api/tags`` in
    milliseconds, and never returning from ``/api/generate`` — a compile test
    does not fail. It stalls three minutes per call, and the run looks hung
    rather than broken.

    ``tests/test_orbitbrief_envelope.py`` was doing exactly that: no failure,
    no output, killed at 560s in ``enrich_entities`` inside
    ``site_llm_verify._call_ollama``. With the kill-switch set it passes in
    well under the same budget.

    A handful of tests deliberately exercise the live-LLM branch; they already
    ``monkeypatch.delenv`` this and stub the transport, so they are unaffected.
    To run the suite against a real host, set ``PARSER_OS_TEST_LIVE_LLM=1``.
    """
    if os.environ.get("PARSER_OS_TEST_LIVE_LLM"):
        yield
        return
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    yield


@pytest.fixture(autouse=True)
def _reset_active_domain_pack() -> None:
    """Avoid cross-test leakage via mutable domain-pack singleton state."""
    from app.domain import load_domain_pack, set_active_domain_pack

    set_active_domain_pack(load_domain_pack(None))
    yield
    set_active_domain_pack(load_domain_pack(None))


@pytest.fixture(autouse=True)
def _reset_semantic_backend() -> None:
    """Avoid cross-test leakage via the frozen embedder verdict.

    ``semantic_rules`` resolves "can the embedder serve this run" ONCE and
    holds it, so that one compile cannot take the embedding path for one
    document and the lexical path for the next. A compile calls
    ``reset_semantic_backend`` when it starts; a test does not, so without
    this the first test to probe would decide for every test after it and the
    suite's result would depend on its order.
    """
    from app.core.semantic_rules import reset_semantic_backend

    reset_semantic_backend()
    yield
    reset_semantic_backend()


@pytest.fixture()
def demo_project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    return create_demo_project(root)
