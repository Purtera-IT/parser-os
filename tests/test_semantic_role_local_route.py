"""A caller that names a local model gets that model, hosted teacher or not.

document_job_scope measured gpt-4.1-mini (the worker's hosted teacher) against
qwen3:32b on seven deals; the teacher was wrong in both directions. So the
decision names its model, and the route honours it -- and its failures never
latch the process-wide unreachable flag that would switch every cheap
judgement off for the rest of the compile."""
import json

import pytest

from app.core import llm_client, ollama_host, semantic_role


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    semantic_role.reset_reachability()
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)  # conftest switches every model off
    monkeypatch.setenv("TEACHER_API_BASE", "https://teacher.example/v1")
    monkeypatch.setattr(ollama_host, "resolve_host", lambda default: "http://ollama.local:11434")
    yield
    semantic_role.reset_reachability()


def test_an_explicit_local_model_bypasses_the_hosted_teacher(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm_client, "complete", lambda *a, **k: pytest.fail("the hosted teacher must not be asked"))
    monkeypatch.setattr(ollama_host, "generation_ready", lambda host, model, timeout=None: seen.setdefault("ready", (model, timeout)) or True)

    class _Resp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    def _urlopen(req, timeout):
        seen["payload"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _Resp(json.dumps({"response": json.dumps({"role": "other_job", "confidence": 0.85})}).encode())

    monkeypatch.setattr(semantic_role.urllib.request, "urlopen", _urlopen)
    role, conf = semantic_role.classify_role("DEAL: x", ["this_deal", "other_job"], instruction="i",
                                             model="ollama:qwen3:32b", timeout=120)
    assert (role, conf) == ("other_job", 0.85)
    assert seen["payload"]["model"] == "qwen3:32b" and seen["timeout"] == 120
    assert seen["ready"] == ("qwen3:32b", 120)


def test_without_the_prefix_the_hosted_teacher_still_answers(monkeypatch):
    monkeypatch.setattr(llm_client, "complete", lambda prompt, **k: json.dumps({"role": "this_deal", "confidence": 0.9}))
    monkeypatch.setattr(ollama_host, "generation_ready", lambda *a, **k: pytest.fail("no local call expected"))
    assert semantic_role.classify_role("t", ["this_deal", "other_job"], instruction="i", model="qwen3:32b") == ("this_deal", 0.9)


def test_a_local_models_failure_does_not_switch_the_default_route_off(monkeypatch):
    monkeypatch.setattr(ollama_host, "generation_ready", lambda *a, **k: True)

    def _boom(req, timeout):
        raise TimeoutError("slow box")

    monkeypatch.setattr(semantic_role.urllib.request, "urlopen", _boom)
    assert semantic_role.classify_role("t", ["a", "b"], instruction="i", model="ollama:qwen3:32b") == (None, 0.0)
    assert semantic_role._llm_unreachable is False
    monkeypatch.setattr(llm_client, "complete", lambda prompt, **k: json.dumps({"role": "a", "confidence": 0.7}))
    assert semantic_role.classify_role("t2", ["a", "b"], instruction="i") == ("a", 0.7)
