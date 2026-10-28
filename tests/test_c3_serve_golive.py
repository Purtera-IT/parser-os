"""Go-live hardening of the C3 base service (ml.c3.serve) and its attention bias
(ml.c3.model.GraphBlock): half-dtype bias at read time only, OOM retry with smaller
windows, an unauthenticated /ready, a bounded read queue, dropped abandoned reads,
and a failed load that ends the process. CPU only; models and reads are stubbed."""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import app.core.c3_type_base as c3
from ml.c3 import serve
from ml.c3.model import GraphBlock
from tests.test_c3_type_base import _atoms

# ---------------------------------------------------------------- GraphBlock


def _old_forward(blk: GraphBlock, x, allowed, same_doc, same_sec):
    """GraphBlock.forward exactly as trained (76f84a47), the reference."""
    n, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).view(n, 3, blk.h, d // blk.h).permute(1, 2, 0, 3)
    bias = (blk.w_doc.view(-1, 1, 1) * same_doc + blk.w_sec.view(-1, 1, 1) * same_sec).to(q.dtype)
    bias = bias.masked_fill(~allowed, float("-inf"))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    x = x + blk.drop(blk.out(a.transpose(0, 1).reshape(n, d)))
    return x + blk.drop(blk.mlp(blk.ln2(x)))


def _inputs(n=24, d=32, heads=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    blk = GraphBlock(d, heads, dropout=0.0)
    with torch.no_grad():
        blk.w_doc.copy_(torch.randn(heads, generator=g))
        blk.w_sec.copy_(torch.randn(heads, generator=g))
    x = torch.randn(n, d, generator=g)
    doc = torch.randint(0, 3, (n,), generator=g)
    sec = torch.randint(0, 5, (n,), generator=g)
    same_doc = (doc[:, None] == doc[None, :]).float()
    same_sec = (sec[:, None] == sec[None, :]).float()
    allowed = torch.rand(n, n, generator=g) > 0.3
    allowed |= torch.eye(n, dtype=torch.bool)  # every row attends somewhere
    return blk, x, allowed, same_doc, same_sec


class _Shapes(TorchDispatchMode):
    """Every tensor an op produces: (dtype, shape)."""

    def __init__(self):
        super().__init__()
        self.seen: list[tuple[torch.dtype, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.seen.append((t.dtype, tuple(t.shape)))
        return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_half_read_matches_the_trained_bias_within_tolerance(dtype):
    blk, x, allowed, sd, ss = _inputs()
    blk.eval()
    blk.to(dtype)
    xh = x.to(dtype)
    with torch.no_grad():
        new = blk(xh, allowed, sd, ss)
        ref = _old_forward(blk, xh, allowed, sd, ss)
        blk32 = blk.float()
        full = _old_forward(blk32, x, allowed, sd, ss)
    assert new.dtype == dtype
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(new.float(), ref.float(), atol=tol, rtol=tol)
    assert torch.allclose(new.float(), full, atol=10 * tol, rtol=10 * tol)


def test_a_half_read_never_builds_the_float32_heads_by_n_by_n_bias(monkeypatch):
    blk, x, allowed, sd, ss = _inputs()
    n, h = x.shape[0], blk.h
    # The attention kernel itself is not under test (CPU's math kernel upcasts its scores).
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, attn_mask: torch.zeros_like(q))
    blk.eval()
    blk.to(torch.float16)
    with torch.no_grad(), _Shapes() as read:
        blk(x.half(), allowed, sd, ss)
    assert (torch.float32, (h, n, n)) not in read.seen
    assert (torch.float16, (h, n, n)) in read.seen
    # Training builds it as before (the old arithmetic, untouched).
    blk.train()
    with _Shapes() as train:
        blk(x.half(), allowed, sd, ss)
    assert (torch.float32, (h, n, n)) in train.seen


def test_training_numerics_are_bit_identical():
    blk, x, allowed, sd, ss = _inputs()
    blk.train()
    for autocast in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            got = blk(x, allowed, sd, ss)
            want = _old_forward(blk, x, allowed, sd, ss)
        assert torch.equal(got, want)
        g1 = torch.autograd.grad(got.sum(), [blk.w_doc, blk.w_sec])
        g2 = torch.autograd.grad(want.sum(), [blk.w_doc, blk.w_sec])
        assert all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_float32_and_grad_enabled_reads_take_the_old_path():
    blk, x, allowed, sd, ss = _inputs()
    blk.eval()
    with torch.no_grad():  # float32 read: unchanged
        assert torch.equal(blk(x, allowed, sd, ss), _old_forward(blk, x, allowed, sd, ss))
    with torch.autocast("cpu", dtype=torch.bfloat16):  # eval with autograd on: unchanged
        assert torch.equal(blk(x, allowed, sd, ss), _old_forward(blk, x, allowed, sd, ss))


# ---------------------------------------------------------------- service


def _payload():
    return c3.deal_payload(c3.atoms_to_deal(_atoms()))


def _ok(model, deal, schema, **kw):
    return {"lines": {a.key: {"reads": {serve.TYPE_KEY: {"answer": "task", "p": 0.9}}} for a in deal.atoms}}


class _Server:
    def __init__(self, base: serve.Base, token: str = ""):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(base, token))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path, token=""):
        req = urllib.request.Request(self.url + path, headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read()), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def post(self, budget=30.0, token=""):
        raw = json.dumps({"deal": _payload(), "budget_s": budget}).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.url + "/read", data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read()), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def close(self):
        self.server.shutdown()


@pytest.fixture()
def serving(monkeypatch):
    for name in ("C3_SERVE_QUEUE", "C3_SERVE_MAX_ATOMS", "C3_SERVE_MIN_ATOMS"):
        monkeypatch.delenv(name, raising=False)
    made: list[_Server] = []

    def _make(read=_ok, *, load=True, ollama=lambda: True, token=""):
        base = serve.Base(load=lambda: ("m", object(), "cuda"), read=read, ollama=ollama)
        if load:
            base.load()
        s = _Server(base, token)
        made.append(s)
        return base, s

    yield _make
    for s in made:
        s.close()


def test_windows_halve_to_the_floor():
    assert serve._windows(8192, 2048) == [8192, 4096, 2048]
    assert serve._windows(4096, 4096) == [4096]


def test_an_oom_frees_memory_and_retries_with_half_windows(serving, monkeypatch):
    freed, seen = [], []
    monkeypatch.setattr(serve, "_free_gpu", lambda: freed.append(1))

    def read(model, deal, schema, **kw):
        seen.append(kw["max_atoms"])
        if kw["max_atoms"] > 2048:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB")
        return _ok(model, deal, schema)

    _, s = serving(read)
    code, body, _ = s.post()
    assert code == 200
    assert seen == [8192, 4096, 2048] and len(freed) == 2
    assert body["max_atoms"] == 2048 and body["oom_retries"] == 2 and body["types"]
    assert body["max_atoms_top"] == 8192


def test_a_first_try_read_reports_the_full_window(serving):
    _, s = serving()
    code, body, _ = s.post()
    assert code == 200 and body["max_atoms"] == 8192 and body["oom_retries"] == 0


def test_oom_at_the_floor_is_503_with_retry_after(serving, monkeypatch):
    monkeypatch.setattr(serve, "_free_gpu", lambda: None)

    def read(model, deal, schema, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    base, s = serving(read)
    code, body, headers = s.post()
    assert code == 503 and body == {"error": "out of memory"} and int(headers["Retry-After"]) > 0
    assert not base._lock.locked()


def test_any_other_error_is_not_retried(serving):
    seen = []

    def read(model, deal, schema, **kw):
        seen.append(kw["max_atoms"])
        raise ValueError("bad deal")

    base, s = serving(read)
    code, _, _ = s.post()
    assert code == 500 and seen == [8192] and not base._lock.locked()


def test_ready_needs_no_token_and_says_nothing_else(serving):
    _, s = serving(token="secret")
    code, body, _ = s.get("/ready")
    assert code == 200 and body == {"ready": True}
    assert s.get("/health")[0] == 401  # /health unchanged: still behind the token
    assert s.get("/health", token="secret")[1]["ready"] is True


def test_ready_is_503_while_loading_or_without_ollama(serving):
    _, cold = serving(load=False)
    assert cold.get("/ready")[:2] == (503, {"ready": False})
    _, no_ollama = serving(ollama=lambda: False)
    assert no_ollama.get("/ready")[0] == 503
    assert no_ollama.get("/health")[1]["ready"] is True  # /health keeps its meaning


def test_ollama_probe_reads_api_tags(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    assert serve._ollama_ok() is True
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens
    assert serve._ollama_ok(timeout=0.5) is False


def _blocking_read():
    entered, release, calls = threading.Event(), threading.Event(), []

    def read(model, deal, schema, **kw):
        calls.append(kw["max_atoms"])
        if len(calls) == 1:
            entered.set()
            release.wait(20)
        return _ok(model, deal, schema)

    return read, entered, release, calls


def test_a_full_queue_is_refused_with_retry_after(serving):
    read, entered, release, calls = _blocking_read()
    base, s = serving(read)
    results = {}
    first = threading.Thread(target=lambda: results.__setitem__("first", s.post()))
    first.start()
    assert entered.wait(10)
    second = threading.Thread(target=lambda: results.__setitem__("second", s.post()))
    second.start()
    for _ in range(100):  # the second is waiting (queue depth 1)
        if base._waiting == 1:
            break
        time.sleep(0.02)
    assert base._waiting == 1
    code, body, headers = s.post()  # third: refused at once
    assert code == 503 and body == {"error": "busy"} and headers["Retry-After"]
    release.set()
    first.join(10)
    second.join(10)
    assert results["first"][0] == 200 and results["second"][0] == 200
    assert len(calls) == 2 and base._waiting == 0


def test_queue_depth_is_configurable(serving, monkeypatch):
    monkeypatch.setenv("C3_SERVE_QUEUE", "0")
    read, entered, release, calls = _blocking_read()
    _, s = serving(read)
    t = threading.Thread(target=s.post)
    t.start()
    assert entered.wait(10)
    assert s.post()[0] == 503  # nothing may wait
    release.set()
    t.join(10)


def test_a_read_whose_budget_ran_out_while_waiting_never_starts(serving):
    read, entered, release, calls = _blocking_read()
    _, s = serving(read)
    t = threading.Thread(target=s.post)
    t.start()
    assert entered.wait(10)
    code, body, _ = s.post(budget=1.0)  # waits ~1 s behind the first, then gives up
    assert code == 504 and body == {"error": "over budget"}
    release.set()
    t.join(10)
    assert len(calls) == 1


def test_a_read_whose_client_hung_up_is_dropped(serving):
    read, entered, release, calls = _blocking_read()
    base, s = serving(read)
    t = threading.Thread(target=s.post)
    t.start()
    assert entered.wait(10)
    raw = json.dumps({"deal": _payload(), "budget_s": 30}).encode()
    port = s.server.server_address[1]
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(b"POST /read HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
    for _ in range(100):
        if base._waiting == 1:
            break
        time.sleep(0.02)
    sock.close()  # the worker gave up
    release.set()
    t.join(10)
    for _ in range(100):
        if base._waiting == 0 and not base._lock.locked():
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert len(calls) == 1 and not base._lock.locked()


def test_a_failed_load_ends_the_process():
    codes = []

    def boom():
        raise FileNotFoundError("SOWSMITH_C3_CHECKPOINT is unset or missing")

    base = serve.Base(load=boom)
    serve.load_or_exit(base, exit=codes.append)
    assert codes == [3] and "FileNotFoundError" in base.error
    ok = serve.Base(load=lambda: ("m", object(), "cpu"))
    serve.load_or_exit(ok, exit=codes.append)
    assert codes == [3] and ok.ready()


def test_the_image_sets_expandable_segments():
    image = Path(serve.__file__).parent / "serve_image"
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in (image / "Dockerfile").read_text()
    assert "expandable_segments:True" in (image / "start.sh").read_text()
    assert 'setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")' in Path(serve.__file__).read_text()
