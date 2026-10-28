"""The C3 base as a small service: a deal's lines in, each line's type out.

    SOWSMITH_C3_CHECKPOINT=... SOWSMITH_C3_ENCODER=... [SOWSMITH_C3_TOKEN=...] \\
        python -m ml.c3.serve --port 8765

It loads the base once, on the GPU when there is one (the live layer's
``SOWSMITH_C3_*`` settings: checkpoint, encoder, device, dtype), and answers:

* ``GET /health``: ``{"ready": bool, "device", "load_ms"}``. The worker calls
  it when a compile starts, so a service that scaled to zero starts loading
  while the documents are still being read.
* ``POST /read``: body ``{"deal": <DealExample dict>, "budget_s": float,
  "line_heads"?: [key], "document_heads"?: [key]}`` (the rows
  ``c3_type_base.deal_payload`` builds). Answer: ``{"types": {key: [answer,
  p]}, "reads"?: {key: {head: [answer, p]}}, "documents"?: {doc_id: {head:
  [answer, p]}}, "read_ms", "lines"}``, or 504 when the read passes its
  budget, or 503 at once while it is still loading (that compile's atoms go
  to the LLM, so a cold start never slows a compile). One read at a time;
  the budget runs from the request's arrival, so time spent waiting for the
  read in front of it counts. More than ``C3_SERVE_QUEUE`` (default 1) reads
  already waiting -> 503 ``busy`` with ``Retry-After`` at once; a read whose
  budget ran out while it waited, or whose client hung up, is dropped before
  it starts. On CUDA out-of-memory the cache is freed and the deal is read
  again with half-size windows (``C3_SERVE_MAX_ATOMS`` 8192 -> ... ->
  ``C3_SERVE_MIN_ATOMS`` 2048); the answer says which (``max_atoms``).
* ``GET /ready``: no auth, no body beyond ``{"ready": bool}``: 200 only when
  the model is loaded and Ollama (``OLLAMA_URL``) answers, else 503. For the
  platform's startup / readiness probes.

A model that fails to load ends the process (exit 3) so the platform
restarts it, instead of answering 503 forever.

With ``SOWSMITH_C3_TOKEN`` set, ``/health``, ``/read`` and ``/v1/quote`` need
``Authorization: Bearer <token>``.
Nothing is logged but counts and times: no line text.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

log = logging.getLogger(__name__)

TYPE_KEY = "col:label_type"
MAX_BODY = 256 << 20
#: The window the base was trained and scored with; equal to
#: ``app.core.c3_type_base.MAX_ATOMS`` (a test keeps them equal). The service
#: ships with ml/ only, so it does not import the parser.
MAX_ATOMS = 8192
#: After a CUDA out-of-memory the deal is read again with half the window, down to this.
MIN_ATOMS = 2048


def _env_int(name: str, default: int, lo: int) -> int:
    try:
        return max(lo, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


class Busy(RuntimeError):
    """Too many reads already waiting: answer 503 now rather than queue."""


class OutOfMemory(RuntimeError):
    """Out of GPU memory even at the smallest window."""


class Gone(RuntimeError):
    """The client hung up while the read waited: nobody to answer."""


def _is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":  # torch.OutOfMemoryError / torch.cuda.OutOfMemoryError
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_gpu() -> None:
    """Give the failed window's blocks back (the caching allocator otherwise keeps them reserved)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - no torch / no CUDA
        pass


def _windows(top: int, floor: int) -> list[int]:
    """8192 -> [8192, 4096, 2048] for floor 2048."""
    out = [max(1, top)]
    while out[-1] // 2 >= max(1, floor):
        out.append(out[-1] // 2)
    return out


def _ollama_ok(timeout: float = 2.0) -> bool:
    """Ollama answers ``/api/tags`` (OLLAMA_URL). Unset -> not required here."""
    url = os.environ.get("OLLAMA_URL", "").strip()
    if not url:
        return True
    import urllib.request

    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _load() -> tuple[Any, Any, str]:
    """(model, schema, device) from the same settings as the live layer."""
    from dataclasses import asdict

    import torch

    from .gates import load_checkpoint
    from .loop import live_student
    from .readiness import digest
    from .schema import load_schema

    path = os.environ.get("SOWSMITH_C3_CHECKPOINT", "").strip()
    encoder = os.environ.get("SOWSMITH_C3_ENCODER", "").strip()
    if not path or not os.path.exists(path):
        raise FileNotFoundError("SOWSMITH_C3_CHECKPOINT is unset or missing")
    if not encoder:
        raise ValueError("SOWSMITH_C3_ENCODER is unset")
    device = os.environ.get("SOWSMITH_C3_DEVICE", "").strip().lower()
    if not device or device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    schema = load_schema()
    model = live_student(schema, encoder).to(device)
    load_checkpoint(path, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    from .loop import to_read_precision

    to_read_precision(model)
    if os.environ.get("SOWSMITH_C3_DTYPE", "").strip().lower() in ("bf16", "bfloat16"):
        model.to(torch.bfloat16)
    model.eval()
    refit = os.environ.get("SOWSMITH_C3_QUESTION_REFIT", "").strip()
    if refit and os.path.exists(refit):
        from .question_fit import load_refit, stale_refit

        why = stale_refit(refit, path)
        if why:
            # An overlay fitted on another base reads that base's features:
            # serve the head alone rather than a correction for other numbers.
            log.error("question refit not used: %s", why)
        else:
            model.question_refit = load_refit(refit, device)
    _prewarm_questions(model, device)
    return model, schema, device


def _prewarm_questions(model, device: str) -> None:
    """Encode the question catalog once at start (predict._query_features),
    so the first compile pays nothing for its questions."""
    try:
        import json as _json
        from pathlib import Path

        import torch

        from .predict import _query_features, _read_autocast

        path = Path(__file__).resolve().parents[2] / "c3_record" / "questions.json"
        texts = [q["text"] for q in _json.loads(path.read_text(encoding="utf-8"))["questions"]]
        with torch.no_grad(), _read_autocast(device):
            for i in range(0, len(texts), 64):
                _query_features(model, texts[i:i + 64])
        log.info("question catalog encoded: %d", len(texts))
    except Exception as exc:
        log.info("question catalog not encoded at start: %s", type(exc).__name__)


class Base:
    """The loaded model and one read at a time. ``load`` and ``read`` are
    seams for tests; by default they are the live layer's own."""

    def __init__(self, load: Callable[[], tuple[Any, Any, str]] | None = None,
                 read: Callable[..., dict[str, Any]] | None = None,
                 ollama: Callable[[], bool] | None = None):
        self._load_fn = load
        self._read_fn = read
        self._ollama_fn = ollama or _ollama_ok
        self.model: tuple[Any, Any, str] | None = None
        self.error = ""
        self.load_ms = 0.0
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        #: Reads allowed to wait for the one running (C3_SERVE_QUEUE, default 1).
        self.max_waiting = _env_int("C3_SERVE_QUEUE", 1, 0)
        self.max_atoms = _env_int("C3_SERVE_MAX_ATOMS", MAX_ATOMS, 1)
        self.min_atoms = min(self.max_atoms, _env_int("C3_SERVE_MIN_ATOMS", MIN_ATOMS, 1))
        self._waiting = 0
        self._count_lock = threading.Lock()

    def load(self) -> None:
        started = time.monotonic()
        try:
            self.model = (self._load_fn or _load)()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:300]
            log.error("C3 base did not load: %s", self.error)
        self.load_ms = round((time.monotonic() - started) * 1000, 1)
        self._loaded.set()

    def ready(self) -> bool:
        return self._loaded.is_set() and self.model is not None

    def serving(self) -> bool:
        """For the readiness probe: loaded and Ollama answering."""
        return self.ready() and bool(self._ollama_fn())

    def _acquire(self, deadline: float, gone: Callable[[], bool] | None) -> None:
        """Take the read lock, or raise: Busy (queue full), TimeoutError (the budget ran out
        while waiting), Gone (the client hung up while waiting)."""
        if not self._lock.acquire(blocking=False):
            with self._count_lock:
                if self._waiting >= self.max_waiting:
                    raise Busy("a read is running and the queue is full")
                self._waiting += 1
            try:
                got = self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
            finally:
                with self._count_lock:
                    self._waiting -= 1
            if not got:
                raise TimeoutError("budget ran out waiting for the read in front")
        if time.monotonic() >= deadline:
            self._lock.release()
            raise TimeoutError("budget ran out waiting for the read in front")
        if gone is not None and gone():
            self._lock.release()
            raise Gone("client disconnected before the read started")

    def read(self, payload: dict[str, Any], budget_s: float, *,
             line_heads: tuple[str, ...] = (), document_heads: tuple[str, ...] = (),
             full: bool = False, questions: tuple[str, ...] = (),
             table_heads: tuple[str, ...] = (), view: str = "",
             gone: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Line types always; with ``line_heads``, those readings of every
        line under ``reads``; with ``document_heads``, each document's group
        judgments under ``documents``. One pass of the model answers all of
        them, so asking for more costs the read almost nothing.

        ``view`` (``quoting`` | ``planning``, c3_record.view): the read the
        compile is serving. Only that view's heads are answered (a quoting read
        no PM head, a planning read no quote or SOW head, deal answers
        included), and a line sent without its own ``view`` takes it."""
        from c3_record import view as cview

        from .data import DealExample

        model, schema, device = self.model  # type: ignore[misc]
        v = cview.name(view)
        if v:
            line_heads = tuple(cview.heads(v, line_heads))
            document_heads = tuple(cview.heads(v, document_heads))
            table_heads = tuple(cview.heads(v, table_heads))
            payload = dict(payload, atoms=[a if not isinstance(a, dict) or a.get("view") else dict(a, view=v)
                                           for a in payload.get("atoms") or []])
        # The budget runs from arrival: a read that waited behind another has less left.
        deadline = time.monotonic() + budget_s
        deal = DealExample.from_dict(payload)
        read = self._read_fn
        if read is None:
            from .predict import read_deal as read
        wanted = tuple(dict.fromkeys((TYPE_KEY,) + tuple(line_heads)))
        kwargs: dict[str, Any] = {}
        if document_heads:
            kwargs["document_heads"] = tuple(document_heads)
        if questions:
            kwargs["questions"] = tuple(questions)
        if table_heads:
            kwargs["table_heads"] = tuple(table_heads)
        arrived = time.monotonic()
        self._acquire(deadline, gone)
        try:
            started = time.monotonic()
            wait_ms = round((started - arrived) * 1000, 1)
            doc = None
            tried: list[int] = []
            for window in _windows(self.max_atoms, self.min_atoms):
                if tried and time.monotonic() >= deadline:
                    raise TimeoutError(f"budget ran out after out-of-memory at max_atoms={tried}")
                if tried and gone is not None and gone():
                    raise Gone("client disconnected before the retry")
                tried.append(window)
                oom = False
                try:
                    # ``full``: the links between lines and the whole-deal answers
                    # too (primary service, billing, scopes, tier), from the same
                    # model -- what a brief is assembled from.
                    doc = read(model, deal, schema, device=device, lines_only=not full,
                               deadline=deadline, heads=wanted, max_atoms=window, **kwargs)
                except Exception as exc:
                    if not _is_oom(exc):
                        raise
                    oom = True
                    log.warning("read out of memory at max_atoms=%d (lines=%d)", window, len(deal.atoms))
                # Outside the except block: the traceback (and the window's tensors it holds) is gone.
                if not oom:
                    break
                _free_gpu()
            if doc is None:
                raise OutOfMemory(f"out of memory at max_atoms={tried}")
            read_ms = round((time.monotonic() - started) * 1000, 1)
        finally:
            self._lock.release()
        types: dict[str, list[Any]] = {}
        reads: dict[str, dict[str, list[Any]]] = {}
        for key, line in (doc.get("lines") or {}).items():
            got_all = line.get("reads") or {}
            got = got_all.get(TYPE_KEY)
            if got and got.get("answer") is not None:
                types[key] = [str(got["answer"]), round(float(got.get("p") or 0.0), 4)]
            for head in line_heads:
                r = got_all.get(head)
                if r and r.get("answer") is not None:
                    reads.setdefault(key, {})[head] = [str(r["answer"]), round(float(r.get("p") or 0.0), 4)]
                elif r and isinstance(r.get("answers"), list):
                    # A set-valued head (read:feeds_decision): the answers over 0.5 and every answer's p.
                    opp = schema.by_key().get(head)
                    names = [a.value for a in opp.answers] if opp is not None else []
                    reads.setdefault(key, {})[head] = [[str(a) for a in r["answers"]],
                                                       {n: round(float(p), 4) for n, p in zip(names, r.get("p") or [])}]
        out: dict[str, Any] = {"types": types, "read_ms": read_ms, "lines": len(deal.atoms),
                               "max_atoms": tried[-1], "oom_retries": len(tried) - 1, "wait_ms": wait_ms}
        if line_heads:
            out["reads"] = reads
        if full:
            out["links"] = [{"src": l["src"], "dst": l["dst"], "relation": l["relation"],
                             "p": round(float(l.get("p") or 0.0), 4)} for l in (doc.get("links") or [])]
            answered = doc.get("deal") or {}
            asked = set(cview.heads(v, list(answered))) if v else set(answered)
            out["deal"] = {k: ([str(a["answer"]), round(float(a.get("p") or 0.0), 4)] if "answer" in a
                               else [a.get("value"), None]) for k, a in answered.items() if k in asked}
        if document_heads:
            out["documents"] = {
                d: {k: [str(v["answer"]), round(float(v.get("p") or 0.0), 4)] for k, v in heads.items()}
                for d, heads in (doc.get("documents") or {}).items()}
        if table_heads:
            # Per table (doc_id||section): the roster / sheet verdicts, the
            # node's row type and the line count (predict._judge_tables).
            out["tables"] = {
                t: {k: ([str(v["answer"]), round(float(v.get("p") or 0.0), 4)] if isinstance(v, dict) else v)
                    for k, v in heads.items()}
                for t, heads in (doc.get("tables") or {}).items()}
        if questions:
            out["questions"] = doc.get("questions") or []
            # The bar the refit was chosen at on held-out deals (question_fit),
            # so the asker keeps a bar that matches the scores it gets.
            out["question_bar"] = getattr(getattr(model, "question_refit", None), "bar", None)
        return out


def _asked(value: Any) -> tuple[str, ...]:
    """A request's list of head keys (``jdg:document_job``), checked for
    shape only: an unknown key reads as nothing, never as an error."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("heads must be a list of strings")
    return tuple(dict.fromkeys(v.strip() for v in value if v.strip()))[:64]


#: Seconds a refused (busy / out-of-memory) read is told to wait.
RETRY_AFTER_S = _env_int("C3_SERVE_RETRY_AFTER_S", 30, 1)


def handler(base: Base, token: str = "") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # counts and times only, below
            pass

        def _send(self, code: int, body: dict[str, Any], retry_after: int = 0) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if retry_after:
                self.send_header("Retry-After", str(int(retry_after)))
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _allowed(self) -> bool:
            if token and self.headers.get("Authorization", "") != f"Bearer {token}":
                self._send(401, {"error": "unauthorized"})
                return False
            return True

        def _gone(self) -> bool:
            """The client closed its end (EOF readable without blocking)."""
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_GET(self):
            if self.path == "/ready":
                # No auth, no deal data: the platform's startup / readiness probe.
                ok = base.serving()
                return self._send(200 if ok else 503, {"ready": ok})
            if not self._allowed():
                return
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            device = base.model[2] if base.model else ""
            self._send(200, {"ready": base.ready(), "device": device,
                             "load_ms": base.load_ms, "error": base.error})

        def do_POST(self):
            if not self._allowed():
                return
            if self.path == "/v1/quote":
                return self._quote()
            if self.path != "/read":
                return self._send(404, {"error": "not found"})
            size = int(self.headers.get("Content-Length") or 0)
            if size <= 0 or size > MAX_BODY:
                return self._send(413, {"error": "body size"})
            try:
                body = json.loads(self.rfile.read(size))
                budget = float(body.get("budget_s") or 90.0)
                deal = body["deal"]
                line_heads = _asked(body.get("line_heads"))
                document_heads = _asked(body.get("document_heads"))
                table_heads = _asked(body.get("table_heads"))
                full = bool(body.get("full"))
                questions = tuple(str(q)[:500] for q in (body.get("questions") or [])[:400])
                view = str(body.get("view") or "")[:16]
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": "bad body"})
            started = time.monotonic()
            # A cold service answers at once instead of making the compile
            # wait: that compile's atoms go to the LLM, the next one is warm.
            if not base.ready():
                return self._send(503, {"error": base.error or "still loading"})
            try:
                out = base.read(deal, max(1.0, budget - (time.monotonic() - started)),
                                line_heads=line_heads, document_heads=document_heads, full=full,
                                questions=questions, table_heads=table_heads, view=view,
                                gone=self._gone)
            except Busy as exc:
                log.warning("read refused: %s", exc)
                return self._send(503, {"error": "busy"}, retry_after=RETRY_AFTER_S)
            except Gone as exc:
                log.warning("read dropped: %s", exc)
                return None
            except OutOfMemory as exc:
                log.error("read failed: %s (lines=%d)", exc, len((deal or {}).get("atoms") or []))
                return self._send(503, {"error": "out of memory"}, retry_after=RETRY_AFTER_S)
            except TimeoutError as exc:
                log.warning("read over budget: %s", exc)
                return self._send(504, {"error": "over budget"})
            except Exception as exc:
                # Where it failed (file:line:function per frame) and the
                # message's first 160 characters -- never the deal's text.
                import traceback as _tb

                frames = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                                    for f in reversed(_tb.extract_tb(exc.__traceback__)[-6:]))
                log.error("read failed: %s: %s | %s", type(exc).__name__, str(exc)[:160], frames)
                return self._send(500, {"error": type(exc).__name__})
            log.info("read lines=%d typed=%d read_ms=%.0f wait_ms=%.0f max_atoms=%d oom_retries=%d",
                     out["lines"], len(out["types"]), out["read_ms"], out["wait_ms"], out["max_atoms"],
                     out["oom_retries"])
            self._send(200, out)

        def _quote(self):
            """Deal Kit quote heads (ml/c3/quote_heads.py): a labor proposal from the deal's atoms and
            c3_brief. No trunk pass, so it never waits on a compile's /read; it is not dumped to disk."""
            size = int(self.headers.get("Content-Length") or 0)
            if size <= 0 or size > MAX_BODY:
                return self._send(413, {"error": "body size"})
            try:
                body = json.loads(self.rfile.read(size))
                deal = body["deal"]
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": "bad body"})
            qh = _quote_heads()
            if qh is None:
                return self._send(503, {"error": "quote heads not loaded"})
            started = time.monotonic()
            try:
                out = qh.quote(deal, word=bool(body.get("word", True)))
            except Exception as exc:
                log.error("quote failed: %s: %s", type(exc).__name__, str(exc)[:160])
                return self._send(500, {"error": type(exc).__name__})
            log.info("quote job=%s ms=%.0f", out.get("job_type"), (time.monotonic() - started) * 1000)
            self._send(200, out)

    return Handler


_QH: list[Any] = []
_QH_LOCK = threading.Lock()


def _quote_heads():
    """Load the quote-heads bundle once (SOWSMITH_C3_QUOTE_HEADS, default next to the checkpoint)."""
    with _QH_LOCK:
        if _QH:
            return _QH[0]
        path = os.environ.get("SOWSMITH_C3_QUOTE_HEADS", "").strip() or os.path.expanduser("~/c3-serve/c3_base/quote_heads_bundle.json")
        if not os.path.exists(path):
            return None
        from .quote_heads import QuoteHeads

        _QH.append(QuoteHeads(path))
        log.info("quote heads %s loaded", _QH[0].version)
        return _QH[0]


def load_or_exit(base: Base, exit: Callable[[int], Any] = os._exit) -> None:
    """Load the model; on failure end the process (exit 3) so the platform restarts the
    replica, rather than a service that answers 503 forever."""
    base.load()
    if not base.ready():
        log.critical("C3 base did not load, exiting: %s", base.error)
        logging.shutdown()
        exit(3)


def main() -> None:
    # Before torch is imported (the load thread): fewer fragmentation OOMs on long deals.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = Base()
    # Answer /health at once and load behind it, so a cold start is visible.
    threading.Thread(target=load_or_exit, args=(base,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port),
                                 handler(base, os.environ.get("SOWSMITH_C3_TOKEN", "").strip()))
    log.info("C3 base service on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
