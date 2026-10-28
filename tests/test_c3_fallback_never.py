"""SOWSMITH_C3_FALLBACK: what a heads call does when the service is in trouble.

``llm`` (default) is the old contract: unreachable, warming, 503, 504, a
timeout or any error reads as no answer and every atom goes to the LLM.
``never`` waits for the heads (retrying with backoff) and, past
SOWSMITH_C3_WAIT_MAX_S or on a request the service refuses, raises
C3HeadsUnavailable so the compile fails visibly. Hermetic: ``_request`` is a
script, sleeping is a recorder, and all text is made up."""
from __future__ import annotations

import io
import socket
import urllib.error

import pytest

import app.core.c3_type_base as c3
from tests.test_c3_type_base import _TEXTS, _atoms

URL = "http://heads.invalid"


def _http(code):
    return urllib.error.HTTPError(URL + "/read", code, f"HTTP {code}", {}, io.BytesIO(b""))


def _types(rows_or_atoms=None):
    return {"types": {}, "documents": {}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("SOWSMITH_C3_FALLBACK", "SOWSMITH_C3_WAIT_MAX_S", "SOWSMITH_C3_RETRY_S",
                 "SOWSMITH_C3_RETRY_MAX_S", "SOWSMITH_C3_BUDGET_S", "SOWSMITH_C3_NEVER_READ_S",
                 "SOWSMITH_C3_TOKEN", "SOWSMITH_C3_QUOTE", "SOWSMITH_C3_BRIEF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOWSMITH_C3_URL", URL)
    monkeypatch.setattr(c3, "_STATUS", {"state": "off"})
    monkeypatch.setattr(c3, "_DOC_STATUS", {"state": "off"})
    slept: list[float] = []
    monkeypatch.setattr(c3, "_sleep", slept.append)
    c3._DEAL.clear()
    yield slept
    c3._DEAL.clear()


def _script(monkeypatch, *steps):
    """``_request`` answers from ``steps`` in order: a dict is returned, an
    exception raised. /health and /read each take a step. Records calls."""
    calls: list[tuple[str, float]] = []
    queue = list(steps)

    def _req(path, body, timeout):
        calls.append((path, timeout))
        step = queue.pop(0) if queue else AssertionError(f"unexpected call {path}")
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(c3, "_request", _req)
    return calls


def _read_answer(atoms):
    _, _, keys = c3._rows(atoms)
    return {"types": {k: ["task", 0.93] for k in keys}}


READY = {"ready": True}
FAILURES = {
    "unreachable": (urllib.error.URLError(ConnectionRefusedError(61, "refused")),),
    "warming": ({"ready": False, "error": "loading"},),
    "503": (READY, _http(503)),
    "504": (READY, _http(504)),
    "timeout": (READY, socket.timeout("timed out")),
    "connection": (READY, ConnectionResetError(54, "reset")),
}


# ------------------------------------------------------------- llm (default)

@pytest.mark.parametrize("kind", sorted(FAILURES))
def test_default_type_read_still_sends_every_atom_to_the_llm(kind, monkeypatch, _env):
    calls = _script(monkeypatch, *FAILURES[kind])
    assert c3.fallback_mode() == "llm"
    assert c3.classify(_atoms()) == {}
    assert len(calls) == len(FAILURES[kind]), "one try, no retry"
    assert _env == [], "the default never sleeps"
    assert c3.status()["state"] in ("unreachable", "warming", "over_budget", "read_failed")


@pytest.mark.parametrize("kind", sorted(FAILURES))
def test_default_document_read_still_hands_back_to_the_judge(kind, monkeypatch, _env):
    calls = _script(monkeypatch, *FAILURES[kind])
    assert c3.documents(_atoms()) == {}
    assert len(calls) == len(FAILURES[kind]) and _env == []


def test_default_quote_still_returns_nothing_on_trouble(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    _script(monkeypatch, _http(504))
    assert c3.quote(_atoms(), {}) == {}


def test_an_unknown_value_is_the_default(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "sometimes")
    assert c3.fallback_mode() == "llm"


# ------------------------------------------------------------------ never

@pytest.mark.parametrize("kind", sorted(FAILURES))
def test_never_retries_each_failure_then_reads(kind, monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    atoms = _atoms()
    calls = _script(monkeypatch, *FAILURES[kind], *FAILURES[kind], READY, _read_answer(atoms))
    got = c3.classify(atoms)
    assert {a.id: got[id(a)] for a in atoms} == {k: ("task", 0.93) for k in _TEXTS}
    assert _env == [5.0, 10.0], "backoff doubles from SOWSMITH_C3_RETRY_S"
    assert calls[-1][0] == "/read" and calls[-1][1] >= 600, "a slow read is waited for, not abandoned"
    assert c3.status()["state"] == "read"


def test_never_backoff_stops_doubling_at_the_cap(monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    atoms = _atoms()
    down = urllib.error.URLError("down")
    _script(monkeypatch, *([down] * 6), READY, _read_answer(atoms))
    c3.classify(atoms)
    assert _env == [5.0, 10.0, 10.0, 10.0, 10.0, 10.0]


def test_never_backoff_cap_is_configurable(monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_RETRY_MAX_S", "60")
    atoms = _atoms()
    down = urllib.error.URLError("down")
    _script(monkeypatch, *([down] * 6), READY, _read_answer(atoms))
    c3.classify(atoms)
    assert _env == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]


def test_never_raises_past_the_ceiling(monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_WAIT_MAX_S", "0.05")
    monkeypatch.setenv("SOWSMITH_C3_RETRY_S", "0.01")
    monkeypatch.setattr(c3, "_sleep", __import__("time").sleep)
    monkeypatch.setattr(c3, "_request", lambda path, body, timeout: (_ for _ in ()).throw(_http(503)))
    with pytest.raises(c3.C3HeadsUnavailable, match=r"type read after .* attempts \(last: HTTP 503\)"):
        c3.classify(_atoms())
    st = c3.status()
    assert st["state"] == "unavailable" and st["waiting"]["attempts"] >= 2


def test_never_document_read_raises_past_the_ceiling(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_WAIT_MAX_S", "0")
    _script(monkeypatch, {"ready": False})
    with pytest.raises(c3.C3HeadsUnavailable, match="document read"):
        c3.documents(_atoms())
    assert c3.document_status()["state"] == "unavailable"


def test_never_quote_raises_past_the_ceiling(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_WAIT_MAX_S", "0")
    _script(monkeypatch, READY, _http(504))
    with pytest.raises(c3.C3HeadsUnavailable, match="quote read"):
        c3.quote(_atoms(), {})


@pytest.mark.parametrize("code", [400, 401, 404, 422])
def test_never_does_not_swallow_or_retry_a_refused_request(code, monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    calls = _script(monkeypatch, READY, _http(code))
    with pytest.raises(c3.C3HeadsUnavailable, match=f"refused the type read \\(HTTP {code}\\)"):
        c3.classify(_atoms())
    assert len(calls) == 2 and _env == []


@pytest.mark.parametrize("code", [408, 429, 500, 502])
def test_never_retries_408_429_and_5xx(code, monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    atoms = _atoms()
    _script(monkeypatch, READY, _http(code), READY, _read_answer(atoms))
    assert len(c3.classify(atoms)) == len(_TEXTS) and _env == [5.0]


def test_never_still_hands_unsure_lines_to_the_llm_by_content(monkeypatch):
    """Content handoff is unchanged: a line the heads have no answer for is
    simply absent from the reading, and the typing stage sends it on."""
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    atoms = _atoms()
    _, _, keys = c3._rows(atoms)
    _script(monkeypatch, READY, {"types": {keys[0]: ["task", 0.93], keys[1]: [None, 0.2]}})
    got = c3.classify(atoms)
    assert set(got) == {id(atoms[0])}


def test_a_shadow_read_never_waits_or_raises(monkeypatch, _env):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    _script(monkeypatch, _http(503))
    job = c3.start(_atoms())
    assert job.result(timeout=5) == {} and _env == []


def test_never_local_model_that_will_not_load_raises(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_URL")
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setattr(c3, "get_model", lambda: None)
    monkeypatch.setattr(c3, "_build", lambda atoms, shas=None: (None, []))
    with pytest.raises(c3.C3HeadsUnavailable, match="not loaded"):
        c3.classify(_atoms())
    with pytest.raises(c3.C3HeadsUnavailable, match="not loaded"):
        c3.documents(_atoms())


def test_never_local_deadline_overrun_raises(monkeypatch):
    from ml.c3 import predict

    def _slow(*a, **kw):
        raise TimeoutError("read passed its deadline at 1/2 windows")

    monkeypatch.delenv("SOWSMITH_C3_URL")
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setattr(predict, "read_deal", _slow)
    monkeypatch.setattr(c3, "get_model", lambda: (object(), object(), "cpu"))
    monkeypatch.setattr(c3, "_build", lambda atoms, shas=None: (None, []))
    with pytest.raises(c3.C3HeadsUnavailable, match="budget"):
        c3.classify(_atoms())
    with pytest.raises(c3.C3HeadsUnavailable, match="deadline"):
        c3.documents(_atoms())
    # and the default still abstains
    monkeypatch.delenv("SOWSMITH_C3_FALLBACK")
    assert c3.classify(_atoms()) == {} and c3.documents(_atoms()) == {}


def test_the_typing_stage_lets_the_failure_through(monkeypatch):
    """typed_atom_classifier abstains on any other C3 error; this one fails the compile."""
    import app.core.typed_atom_classifier as tac

    for name in ("SOWSMITH_TYPED_CLASSIFIER_DISABLE", "SOWSMITH_DISABLE_LLM", "SOWSMITH_C3_SURE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(tac, "_call_ollama", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))

    def _down(atoms, shas=None):
        raise c3.C3HeadsUnavailable("heads down")

    monkeypatch.setattr(c3, "classify", _down)
    with pytest.raises(c3.C3HeadsUnavailable):
        tac.classify_atoms(_atoms())


def test_the_document_job_judge_lets_the_failure_through(monkeypatch):
    import app.core.document_job_scope as djs

    def _down(atoms, *a, **k):
        raise c3.C3HeadsUnavailable("heads down")

    monkeypatch.setattr(djs, "c3_mode", lambda: "on")
    monkeypatch.setattr(c3, "documents", _down)
    from tests.test_c3_document_job import DEAL, SDWAN

    with pytest.raises(c3.C3HeadsUnavailable):
        djs.judge_documents(SDWAN, deal_name=DEAL)
