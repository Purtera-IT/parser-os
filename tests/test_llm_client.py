"""Provider-agnostic teacher client (:mod:`app.core.llm_client`).

Network-free: we monkeypatch ``urllib.request.urlopen`` with a fake that
records requests and returns canned OpenAI-compatible bodies. These tests pin
the contracts the compile path relies on:

  * **default-off** — with no ``TEACHER_API_BASE`` the client is disabled and
    ``complete`` is a guess-free no-op (``""``), so callers keep their local
    Ollama path;
  * **request shape** — when enabled it POSTs ``{base}/chat/completions`` with a
    single user message, the Bearer key, and ``stream: False``;
  * **response parse** — pulls ``choices[0].message.content`` and strips a
    ``<think>`` reasoning block;
  * **degrade** — transport error / empty choices / non-JSON → ``""`` (never a
    guess); the caller treats empty as "no LLM result";
  * **retry/backoff** — retries on 429/5xx then succeeds, and gives up to ``""``
    on persistent failure without hanging.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.core import llm_client


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Start every test from a known-clean teacher config + no real sleeps.
    for var in (
        "TEACHER_API_BASE", "TEACHER_API_KEY", "TEACHER_MODEL",
        "TEACHER_API_TIMEOUT", "SOWSMITH_LLM_TIMEOUT", "TEACHER_API_RETRIES",
        "TEACHER_API_BACKOFF", "TEACHER_API_MAX_CONCURRENCY",
        "SOWSMITH_LLM_POOL_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_a, **_k: None)
    # Adaptive param-stripping is process-wide state; clear it so one test's
    # learned-unsupported param can't leak into another's request-shape asserts.
    llm_client._UNSUPPORTED_PARAMS.clear()


def _body(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


class _FakeResp(io.BytesIO):
    """Context-manager byte stream mimicking urlopen's return."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ── gating ───────────────────────────────────────────────────────────────────
def test_disabled_by_default_is_noop(monkeypatch):
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("must not hit the network when disabled")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    assert llm_client.teacher_api_enabled() is False
    assert llm_client.complete("hello") == ""
    assert called["n"] == 0


def test_enabled_only_with_base(monkeypatch):
    assert llm_client.teacher_api_enabled() is False
    monkeypatch.setenv("TEACHER_API_BASE", "https://api.deepseek.com/v1")
    assert llm_client.teacher_api_enabled() is True


def test_teacher_model_env_override(monkeypatch):
    assert llm_client.teacher_model() == "deepseek-chat"
    monkeypatch.setenv("TEACHER_MODEL", "deepseek-reasoner")
    assert llm_client.teacher_model() == "deepseek-reasoner"


# ── request shape ─────────────────────────────────────────────────────────────
def test_complete_posts_openai_shape(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://api.deepseek.com/v1")
    monkeypatch.setenv("TEACHER_API_KEY", "sk-test")
    monkeypatch.setenv("TEACHER_MODEL", "deepseek-chat")
    seen = {}

    def _fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_body("the answer"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    out = llm_client.complete("classify this", max_tokens=256)

    assert out == "the answer"
    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer sk-test"
    p = seen["payload"]
    assert p["model"] == "deepseek-chat"
    assert p["stream"] is False
    assert p["max_tokens"] == 256
    assert p["temperature"] == 0.0
    assert p["messages"] == [{"role": "user", "content": "classify this"}]


def test_no_auth_header_without_key(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    seen = {}

    def _fake(req, timeout=None):
        seen["headers"] = {k.lower() for k in req.headers}
        return _FakeResp(_body("ok"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == "ok"
    assert "authorization" not in seen["headers"]


# ── vision (multimodal) ───────────────────────────────────────────────────────
def test_complete_vision_posts_image_url_data_uri(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_MODEL", "claude-opus-4-8")
    seen = {}

    def _fake(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_body("table read"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    out = llm_client.complete_vision("read this", "QUJD", mime="image/png")

    assert out == "table read"
    content = seen["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "read this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    # vision call uses the (vision-capable) teacher model by default
    assert seen["payload"]["model"] == "claude-opus-4-8"


def test_complete_vision_model_override_env(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_MODEL", "deepseek-chat")  # text model, not visual
    monkeypatch.setenv("TEACHER_VISION_MODEL", "claude-sonnet-4-6")
    seen = {}

    def _fake(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_body("ok"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    llm_client.complete_vision("p", "QUJD")
    assert seen["payload"]["model"] == "claude-sonnet-4-6"


def test_complete_vision_empty_image_is_noop(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")

    def _boom(*_a, **_k):
        raise AssertionError("must not hit network with no image")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    assert llm_client.complete_vision("p", "") == ""


def test_complete_vision_noop_when_disabled(monkeypatch):
    # No TEACHER_API_BASE → guess-free no-op even with a real image.
    def _boom(*_a, **_k):
        raise AssertionError("must not hit network when teacher disabled")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    assert llm_client.complete_vision("p", "QUJD") == ""


# ── response parse ────────────────────────────────────────────────────────────
def test_extract_strips_think_block():
    body = _body("<think>scratch reasoning</think>  final text ").decode("utf-8")
    assert llm_client._extract_text(body) == "final text"


def test_extract_handles_empty_and_garbage():
    assert llm_client._extract_text("not json") == ""
    assert llm_client._extract_text(json.dumps({"choices": []})) == ""
    assert llm_client._extract_text(json.dumps({})) == ""


# ── degrade paths ─────────────────────────────────────────────────────────────
def test_transport_error_degrades_to_empty(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_API_RETRIES", "1")

    def _fake(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == ""


def test_non_retryable_http_returns_empty(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_API_RETRIES", "3")
    calls = {"n": 0}

    def _fake(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "bad", hdrs=None, fp=None)

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == ""
    assert calls["n"] == 1  # 400 is not retryable → one shot only


# ── adaptive param-stripping (self-heal on 400 "param deprecated") ─────────────
def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    """An HTTPError whose ``.read()`` yields ``body`` (real reasoning models
    return the offending param name in the 400 body)."""
    return urllib.error.HTTPError(
        url, code, "bad request", hdrs=None, fp=io.BytesIO(body.encode("utf-8"))
    )


def test_strips_unsupported_param_on_400_then_retries(monkeypatch):
    # Opus 4.x rejects ``temperature`` with a 400 naming it; we strip + retry.
    monkeypatch.setenv("TEACHER_API_BASE", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TEACHER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("TEACHER_API_RETRIES", "2")
    seen = []

    def _fake(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        seen.append(payload)
        if "temperature" in payload:
            raise _http_error(
                req.full_url, 400,
                '{"error":{"message":"`temperature` is deprecated for this model."}}',
            )
        return _FakeResp(_body("pong"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("ping") == "pong"
    # exactly two round-trips: the 400 with temperature, then the stripped retry
    assert len(seen) == 2
    assert "temperature" in seen[0]
    assert "temperature" not in seen[1]
    # remembered process-wide so future calls don't re-pay the failed round-trip
    assert "temperature" in llm_client._UNSUPPORTED_PARAMS


def test_unsupported_param_remembered_skips_it_next_call(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TEACHER_MODEL", "claude-opus-4-8")
    llm_client._UNSUPPORTED_PARAMS.add("temperature")
    seen = {}

    def _fake(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_body("ok"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == "ok"
    # first request already omits the known-bad param — no wasted round-trip
    assert "temperature" not in seen["payload"]


def test_400_without_known_param_still_degrades(monkeypatch):
    # A 400 that doesn't name a strippable param must NOT loop — return "".
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    calls = {"n": 0}

    def _fake(req, timeout=None):
        calls["n"] += 1
        raise _http_error(req.full_url, 400,
                          '{"error":{"message":"your input is malformed"}}')

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == ""
    assert calls["n"] == 1  # no strippable param named → one shot, no retry storm


# ── retry / backoff ───────────────────────────────────────────────────────────
def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_API_RETRIES", "3")
    state = {"n": 0}

    def _fake(req, timeout=None):
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", hdrs=None, fp=None)
        return _FakeResp(_body("recovered"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == "recovered"
    assert state["n"] == 3


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_API_RETRIES", "2")
    state = {"n": 0}

    def _fake(req, timeout=None):
        state["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 503, "down", hdrs=None, fp=None)

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    assert llm_client.complete("p") == ""
    assert state["n"] == 3  # initial try + 2 retries


# ── usage / cost meter ────────────────────────────────────────────────────────
def _usage_body(content: str, pt: int, ct: int) -> bytes:
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct},
    }).encode("utf-8")


def test_usage_accumulates_and_prices_known_model(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TEACHER_MODEL", "claude-sonnet-4-5")
    llm_client.reset_usage()

    def _fake(req, timeout=None):
        return _FakeResp(_usage_body("ok", 1_000_000, 1_000_000))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    llm_client.complete("a")
    llm_client.complete("b")

    snap = llm_client.usage_snapshot()
    assert snap["calls"] == 2
    assert snap["prompt_tokens"] == 2_000_000
    assert snap["completion_tokens"] == 2_000_000
    # Sonnet: $3/M in, $15/M out → per call $3 + $15 = $18; two calls = $36.
    assert abs(snap["cost_usd"] - 36.0) < 1e-6
    assert "est. $36.00" in llm_client.format_cost_report()
    llm_client.reset_usage()


def test_price_env_override_wins(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_MODEL", "some-unknown-model")
    monkeypatch.setenv("TEACHER_API_PRICE_IN", "1.0")
    monkeypatch.setenv("TEACHER_API_PRICE_OUT", "2.0")
    llm_client.reset_usage()

    def _fake(req, timeout=None):
        return _FakeResp(_usage_body("ok", 1_000_000, 500_000))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    llm_client.complete("a")
    # 1M*$1 + 0.5M*$2 = $2.00
    assert abs(llm_client.usage_snapshot()["cost_usd"] - 2.0) < 1e-6
    llm_client.reset_usage()


def test_unknown_model_reports_tokens_only(monkeypatch):
    monkeypatch.setenv("TEACHER_API_BASE", "https://x/v1")
    monkeypatch.setenv("TEACHER_MODEL", "mystery-llm-9000")
    llm_client.reset_usage()

    def _fake(req, timeout=None):
        return _FakeResp(_usage_body("ok", 100, 50))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    llm_client.complete("a")
    snap = llm_client.usage_snapshot()
    assert snap["prompt_tokens"] == 100 and snap["completion_tokens"] == 50
    assert snap["cost_usd"] == 0.0
    assert "unknown" in llm_client.format_cost_report()
    llm_client.reset_usage()


# ── pool budget ───────────────────────────────────────────────────────────────
def test_pool_budget_env_override(monkeypatch):
    monkeypatch.setenv("SOWSMITH_LLM_POOL_TIMEOUT", "45")
    assert llm_client.pool_budget_seconds() == 45


def test_pool_budget_floor_when_local(monkeypatch):
    # No hosted teacher → falls back to 3x default_per_call, floored at 600.
    assert llm_client.pool_budget_seconds() == 600
