"""What the deal page is told about the heads (telemetry.compile_notices -> the worker's
compile-progress.json): ``heads_warming`` while a compile waits on a service that is starting
(cleared when it answers or gives up), ``heads_read_degraded`` when a read came back in smaller
windows than full size, ``large_deal`` past SOWSMITH_C3_LARGE_DEAL_LINES. Nothing for a warm,
full-size read. Hermetic: the service is a script, sleeping a recorder, all text made up."""
from __future__ import annotations

import io
import urllib.error

import pytest

import app.core.c3_type_base as c3
from app.core import telemetry
from app.core.compiler import compile_project

URL = "http://heads.invalid"


def _http(code):
    return urllib.error.HTTPError(URL + "/read", code, f"HTTP {code}", {}, io.BytesIO(b""))


def _body(n):
    return {"deal": {"deal_id": "d", "company": "purtera",
                     "atoms": [{"key": f"k{i}", "text": "x"} for i in range(n)]}, "budget_s": 90}


def _read(n, max_atoms=8192, oom=0, top=8192):
    return {"types": {}, "documents": {}, "lines": n, "max_atoms": max_atoms, "oom_retries": oom,
            "max_atoms_top": top, "read_ms": 1.0, "wait_ms": 0.0}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("SOWSMITH_C3_FALLBACK", "SOWSMITH_C3_WAIT_MAX_S", "SOWSMITH_C3_RETRY_S",
                 "SOWSMITH_C3_RETRY_MAX_S", "SOWSMITH_C3_LARGE_DEAL_LINES", "SOWSMITH_C3_QUOTE",
                 "SOWSMITH_C3_BRIEF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOWSMITH_C3_URL", URL)
    telemetry.clear_compile_notices()
    yield
    telemetry.clear_compile_notices()


def _script(monkeypatch, *steps, seen=None):
    queue = list(steps)

    def _raw(path, body, timeout):
        step = queue.pop(0) if queue else AssertionError(f"unexpected call {path}")
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(c3, "_request_raw", _raw)
    seen = [] if seen is None else seen
    # What the deal page would read at each pause (the worker writes on its own timer).
    monkeypatch.setattr(c3, "_sleep", lambda s: seen.append(telemetry.compile_notices()))
    return seen


def test_a_wait_on_a_starting_service_says_heads_warming_then_clears(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    seen = _script(monkeypatch,
                   urllib.error.URLError("connection refused"),     # scaled to zero
                   {"ready": False, "error": "still loading"},       # loading
                   {"ready": True}, _read(40))
    got = c3._wait_for_heads("/read", _body(40), {}, "type read")
    assert got["lines"] == 40
    assert len(seen) == 2
    first, second = seen[0]["heads_warming"], seen[1]["heads_warming"]
    assert first["what"] == "type read" and first["attempts"] == 1 and first["ceiling_s"] == 1800.0
    assert second["attempts"] == 2 and "warming" in second["reason"]
    assert first["since"] == second["since"] and first["since"].endswith("+00:00")
    # Answered, warm, full size, small deal: nothing left to say.
    assert telemetry.compile_notices() == {}


def test_giving_up_clears_heads_warming(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_WAIT_MAX_S", "0")
    _script(monkeypatch, _http(503))
    with pytest.raises(c3.C3HeadsUnavailable):
        c3._wait_for_heads("/read", _body(5), {}, "type read")
    assert "heads_warming" not in telemetry.compile_notices()


def test_a_refused_read_clears_heads_warming(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    _script(monkeypatch, _http(503), _http(400))
    with pytest.raises(c3.C3HeadsUnavailable):
        c3._wait_for_heads("/read", _body(5), {}, "type read")
    assert telemetry.compile_notices() == {}


def test_an_out_of_memory_read_is_degraded_and_a_big_deal_is_large(monkeypatch):
    _script(monkeypatch, _read(9000, max_atoms=4096, oom=1))
    c3._request("/read", _body(9000), timeout=5)
    n = telemetry.compile_notices()
    assert n["heads_read_degraded"] == {"lines": 9000, "max_atoms": 4096, "max_atoms_top": 8192,
                                        "oom_retries": 1, "reads": 1}
    assert n["large_deal"] == {"lines": 9000, "threshold": 8000}
    assert set(c3.notices()) == {"heads_read_degraded", "large_deal"}


def test_the_worst_read_wins_and_reads_are_counted(monkeypatch):
    _script(monkeypatch, _read(9000, max_atoms=4096, oom=1), _read(9000, max_atoms=2048, oom=2),
            _read(9000, max_atoms=4096, oom=1))
    for _ in range(3):
        c3._request("/read", _body(9000), timeout=5)
    d = telemetry.compile_notices()["heads_read_degraded"]
    assert (d["max_atoms"], d["oom_retries"], d["reads"]) == (2048, 2, 3)


def test_a_service_capped_below_the_training_window_is_degraded_for_a_deal_bigger_than_it(monkeypatch):
    _script(monkeypatch, _read(5000, max_atoms=4096, oom=0, top=4096), _read(3000, max_atoms=4096, top=4096))
    c3._request("/read", _body(5000), timeout=5)
    assert telemetry.compile_notices()["heads_read_degraded"]["max_atoms"] == 4096
    telemetry.clear_compile_notices()
    c3._request("/read", _body(3000), timeout=5)  # the whole deal fits one window: full read
    assert telemetry.compile_notices() == {}


def test_a_warm_full_size_read_says_nothing(monkeypatch):
    _script(monkeypatch, {"ready": True}, _read(7999))
    c3._request("/health", None, timeout=5)
    c3._request("/read", _body(7999), timeout=5)
    assert telemetry.compile_notices() == {}


def test_the_large_deal_bar_is_a_setting(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_LARGE_DEAL_LINES", "100")
    _script(monkeypatch, _read(101))
    c3._request("/read", _body(101), timeout=5)
    assert telemetry.compile_notices() == {"large_deal": {"lines": 101, "threshold": 100}}


def test_a_shadow_read_tells_the_pm_nothing(monkeypatch):
    _script(monkeypatch, _read(9000, max_atoms=2048, oom=2))
    c3._SHADOW.on = True
    try:
        c3._request("/read", _body(9000), timeout=5)
    finally:
        c3._SHADOW.on = False
    assert telemetry.compile_notices() == {}


def test_a_compile_read_in_smaller_windows_is_degraded_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "on")
    monkeypatch.setattr(c3, "warm", lambda: None)

    def _raw(path, body, timeout):
        if path == "/health":
            return {"ready": True}
        return _read(c3._read_lines(body), max_atoms=2048, oom=2)

    monkeypatch.setattr(c3, "_request_raw", _raw)
    d = tmp_path / "deal_c3_small_windows"
    d.mkdir()
    (d / "notes.md").write_text("Install four access points in the lobby.\nThe customer supplies the mounts.\n")
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    caps = result.compile_capabilities
    assert caps["compile_state"] == "degraded"
    assert caps["c3_ok"] is True  # the heads answered; nothing went to the LLM
    deg = caps["c3_notices"]["heads_read_degraded"]
    assert deg["max_atoms"] == 2048 and deg["oom_retries"] == 2
    assert any(w.startswith("WARNING: C3 heads read") and "2048-line windows" in w for w in result.warnings)


def test_a_new_compile_starts_with_no_notices(monkeypatch, tmp_path):
    telemetry.set_compile_notice("heads_read_degraded", {"lines": 1})
    monkeypatch.delenv("SOWSMITH_C3_URL", raising=False)
    d = tmp_path / "deal_quiet"
    d.mkdir()
    (d / "notes.md").write_text("Install four access points in the lobby.\n")
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    assert "c3_notices" not in (result.compile_capabilities or {})
    assert result.compile_capabilities["compile_state"] == "ok"
