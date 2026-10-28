"""The C3 base as a service (ml.c3.serve) and the live layer's client for it
(SOWSMITH_C3_URL). Hermetic: the model and its read are stubbed, the server
runs on a local port, and all text is made up."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import app.core.c3_type_base as c3
from ml.c3 import serve
from tests.test_c3_type_base import _TEXTS, _atoms


def _stub_read(model, deal, schema, **kw):
    if kw.get("deadline") and model == "slow":
        raise TimeoutError("read passed its deadline at 1/2 windows")
    lines = {}
    for a in deal.atoms:
        answer = "task" if "mount" in a.text else "_keep"
        lines[a.key] = {"reads": {serve.TYPE_KEY: {"answer": answer, "p": 0.91}}}
    return {"lines": lines}


@pytest.fixture(autouse=True)
def _fresh_deal_context():
    """No deal read left behind by another test (c3._DEAL is process-wide)."""
    c3._DEAL.clear()
    yield
    c3._DEAL.clear()


@pytest.fixture()
def service(monkeypatch):
    def _start(model="ok", token=""):
        base = serve.Base(load=lambda: (model, object(), "cuda"), read=_stub_read)
        base.load()
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(base, token))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    started: list[ThreadingHTTPServer] = []
    for name in ("SOWSMITH_C3_URL", "SOWSMITH_C3_TOKEN", "SOWSMITH_C3_TYPE", "SOWSMITH_C3_BUDGET_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(c3, "_STATUS", {"state": "off"})
    yield _start
    for s in started:
        s.shutdown()


def _get(url, token=""):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_health_says_ready_and_where(service):
    got = _get(service() + "/health")
    assert got["ready"] is True and got["device"] == "cuda"


def test_the_layer_reads_through_the_service(service, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_URL", service() + "/")
    atoms = _atoms()
    got = c3.classify(atoms)
    by = {a.id: got[id(a)] for a in atoms}
    assert by["a_task"] == ("task", 0.91)
    assert by["a_keep"] == ("_keep", 0.91)
    assert len(got) == len(_TEXTS)
    st = c3.status()
    assert st["state"] == "read" and st["remote"] is True and st["lines"] == len(_TEXTS)


def test_the_service_never_loads_a_model_in_the_worker(service, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_URL", service())
    monkeypatch.setattr(c3, "get_model", lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    assert c3.classify(_atoms())


def test_a_token_is_required_when_set(service, monkeypatch):
    url = service(token="s3cret")
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/health")
    assert err.value.code == 401
    assert _get(url + "/health", token="s3cret")["ready"] is True
    monkeypatch.setenv("SOWSMITH_C3_URL", url)
    monkeypatch.setenv("SOWSMITH_C3_TOKEN", "s3cret")
    assert len(c3.classify(_atoms())) == len(_TEXTS)


def test_a_read_over_budget_sends_every_atom_to_the_llm(service, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_URL", service(model="slow"))
    assert c3.classify(_atoms()) == {}
    assert c3.status()["state"] == "over_budget"


def test_a_service_that_is_down_sends_every_atom_to_the_llm(service, monkeypatch):
    url = service()
    monkeypatch.setenv("SOWSMITH_C3_URL", url.rsplit(":", 1)[0] + ":1")
    monkeypatch.setenv("SOWSMITH_C3_BUDGET_S", "1")
    assert c3.classify(_atoms()) == {}
    assert c3.status()["state"] == "unreachable"


def test_warm_pings_only_when_the_layer_is_on(service, monkeypatch):
    calls = []
    monkeypatch.setattr(c3, "_request", lambda path, body, timeout: calls.append(path) or {})
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://example.invalid")
    c3.warm()
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "shadow")
    c3.warm()
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon and t.name.startswith("Thread"):
            t.join(timeout=1)
    assert calls == ["/health"]


def test_the_payload_round_trips_what_the_base_reads():
    from ml.c3.data import DealExample

    deal = c3.atoms_to_deal(_atoms())
    back = DealExample.from_dict(json.loads(json.dumps(c3.deal_payload(deal))))
    assert [(a.key, a.text, a.doc_kind, a.order) for a in back.atoms] == \
        [(a.key, a.text, a.doc_kind, a.order) for a in deal.atoms]


def test_a_cold_service_answers_at_once_and_the_compile_goes_to_the_llm(service, monkeypatch):
    base = serve.Base(load=lambda: ("ok", object(), "cuda"), read=_stub_read)  # never loaded
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(base))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("SOWSMITH_C3_URL", f"http://127.0.0.1:{server.server_address[1]}")
        assert c3.classify(_atoms()) == {}
        st = c3.status()
        assert st["state"] == "warming" and st["reason"] == "still loading" and st["read_ms"] < 2000
    finally:
        server.shutdown()


def test_the_service_reads_with_the_layers_window():
    assert serve.MAX_ATOMS == c3.MAX_ATOMS


def test_the_client_needs_no_ml_code(service, monkeypatch):
    import builtins

    monkeypatch.setenv("SOWSMITH_C3_URL", service())
    real = builtins.__import__

    def _no_ml(name, *a, **kw):
        if name == "ml" or name.startswith("ml."):
            raise ImportError("the worker ships without ml/")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_ml)
    got = c3.classify(_atoms())
    monkeypatch.setattr(builtins, "__import__", real)
    assert len(got) == len(_TEXTS)


def test_an_unreachable_service_is_skipped_in_seconds_not_the_budget(service, monkeypatch):
    """2026-10-15: a URL with no route timed out the whole read budget."""
    import socket
    import time

    hole = socket.socket()
    hole.bind(("127.0.0.1", 0))
    hole.listen(0)  # accepts nothing: connects hang like a missing route
    try:
        monkeypatch.setenv("SOWSMITH_C3_URL", f"http://127.0.0.1:{hole.getsockname()[1]}")
        monkeypatch.setenv("SOWSMITH_C3_CONNECT_S", "0.5")
        monkeypatch.setenv("SOWSMITH_C3_BUDGET_S", "60")
        started = time.monotonic()
        assert c3.classify(_atoms()) == {}
        assert time.monotonic() - started < 5
        assert c3.status()["state"] == "unreachable"
    finally:
        hole.close()



def test_a_read_goes_through_a_path_prefix(monkeypatch):
    """The dev worker reaches the Mac through the Ollama relay at /c3."""
    seen = []
    monkeypatch.setenv("SOWSMITH_C3_URL", "https://relay.example/c3/")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: seen.append(req.full_url) or _Resp())
    c3._request("/health", None, timeout=1)
    assert seen == ["https://relay.example/c3/health"]


def test_a_quote_waits_for_the_embedding_model_start_sh_loads_beside_the_server(service, monkeypatch, tmp_path):
    """start.sh serves /read at once and pulls the quote heads' embedding model
    beside it; until its marker exists a quote is 503 (warming, retried by the
    worker), never an answer computed without the model."""
    class _QH:
        version = "t"

        def quote(self, deal, word=True):
            return {"job_type": "av", "lines": []}

    marker = tmp_path / ".embed_ready"
    monkeypatch.setenv("C3_EMBED_READY_FILE", str(marker))
    monkeypatch.setattr(serve, "_QH", [_QH()])
    url = service()
    body = json.dumps({"deal": {"atoms": []}}).encode()

    def _post():
        req = urllib.request.Request(url + "/v1/quote", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    with pytest.raises(urllib.error.HTTPError) as err:
        _post()
    assert err.value.code == 503
    assert _get(url + "/health")["ready"] is True, "reads are served while the model loads"
    marker.touch()
    assert _post()["job_type"] == "av"
    monkeypatch.delenv("C3_EMBED_READY_FILE")
    assert serve._embed_ready() is True, "no marker configured: nothing to wait for"
