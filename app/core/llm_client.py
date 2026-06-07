"""Provider-agnostic *teacher* LLM client (OpenAI-compatible chat completions).

The compile path's **teacher** calls — entity extraction
(:mod:`app.core.multi_entity_llm`) and typed-atom classification
(:mod:`app.core.typed_atom_classifier`) — can be served by either:

* a **local Ollama** (the default; set ``OLLAMA_HOST``), or
* any **OpenAI-compatible hosted API** — DeepSeek, OpenRouter, Together, Groq,
  Fireworks, OpenAI, ... — by setting ``TEACHER_API_BASE`` (+ ``TEACHER_API_KEY``
  and ``TEACHER_MODEL``).

Why offer a hosted teacher:

* **Better labels.** This whole system is distillation (teacher → head). The
  head can only get as good as the teacher's labels, so a stronger teacher
  raises the head's ceiling.
* **Real concurrency.** A local box (esp. a Mac) *serializes* the compile's
  many parallel extractor calls; a hosted API serves them at once, collapsing
  per-deal wall time.
* **No infra.** Just an endpoint + key — no GPU to provision, no model to pull.

Crucially, **only the teacher moves.** The *embedder* (``qwen3-embedding:8b``)
stays local and pinned: the trained heads, kNN store, and all thresholds live in
its vector space and must not change. This module never touches embeddings.

The hosted teacher is a one-time *teaching* cost, not a per-compile tax — the
head cutover (#70/#71) removes the teacher from the serve path entirely.

Hard contracts (mirror the local ``_call_ollama`` callers' expectations):

* **Enforced timeout** on every call — no infinite socket hang.
* **Bounded retry with backoff** on 429 / 5xx / transport errors (hosted APIs
  rate-limit; the compile fires many concurrent calls).
* **Global concurrency cap** (a semaphore) so nested extractor thread-pools
  can't open hundreds of sockets and trip provider rate limits.
* **Guess-free degrade.** On give-up it returns ``""`` — the caller treats empty
  as "no LLM result" and falls through to its deterministic path, exactly as it
  does for a local Ollama failure.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

# Defaults chosen for a hosted teacher: a tight-but-generous per-call timeout
# (hosted APIs answer in seconds, not the minutes a local box queues for) and a
# modest global concurrency cap to stay under provider rate limits.
_DEFAULT_API_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 1.5
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_MODEL = "deepseek-chat"

# Status codes worth retrying (transient): rate limit + the 5xx family.
_RETRY_STATUS = {429, 500, 502, 503, 504}

# Lazily-built global semaphore bounding concurrent in-flight API calls across
# every thread-pool in the process.
_SEM_LOCK = threading.Lock()
_SEM: threading.Semaphore | None = None
_SEM_SIZE = 0

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Adaptive param-stripping: some models reject request params we send (e.g.
# reasoning models like Opus 4.x deprecate ``temperature``). When a provider
# returns a 400 naming such a param, we drop it and remember it process-wide so
# subsequent calls don't re-pay the failed round-trip. Universal: no per-model
# special-casing — we strip whatever the provider says it won't accept.
_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_UNSUPPORTED_LOCK = threading.Lock()
_UNSUPPORTED_PARAMS: set[str] = set()
# Never strip these — without them the request is meaningless.
_ESSENTIAL_PARAMS = frozenset({"model", "messages", "max_tokens"})


def _strippable_param(err_body: str, payload: dict) -> str | None:
    """Return a request param the provider's 400 says it won't accept, or None.

    Looks for a backtick-quoted identifier in the error message that is a
    non-essential key currently in the payload (e.g. ``temperature``)."""
    for m in _BACKTICK_RE.finditer(err_body or ""):
        name = m.group(1)
        if name in payload and name not in _ESSENTIAL_PARAMS:
            return name
    return None

# ── usage / cost meter ────────────────────────────────────────────────────────
# Every OpenAI-compatible response carries a ``usage`` block; we accumulate it
# process-wide so a run can print exact tokens + estimated $ (instead of a guess).
_USAGE_LOCK = threading.Lock()
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

# Rough LIST prices, USD per 1M tokens (input, output). Matched by substring of
# the model id (lowercased). Prices drift — override per-run with
# ``TEACHER_API_PRICE_IN`` / ``TEACHER_API_PRICE_OUT`` (dollars per 1M tokens).
_PRICE_PER_M = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


def _price_for(model: str) -> "tuple[float, float] | None":
    """(input, output) $/1M for ``model``. Per-endpoint aware so a split-route
    vision model is priced correctly instead of inheriting the text teacher's
    override:

    * If ``model`` is the configured *vision* model and ``TEACHER_VISION_PRICE_IN
      /OUT`` are set, those win.
    * If ``model`` is the configured *text* teacher and ``TEACHER_API_PRICE_IN
      /OUT`` are set, those win.
    * Otherwise fall back to the longest matching known prefix in
      ``_PRICE_PER_M`` (so e.g. ``claude-sonnet-4-6`` → $3/$15), else ``None``.

    The old behavior applied ``TEACHER_API_PRICE_*`` to *every* model, which
    mis-priced a Claude vision teacher with DeepSeek's numbers (~10x low)."""
    m = (model or "").lower()
    text_model = (os.environ.get("TEACHER_MODEL") or "").lower()
    vision_model = (os.environ.get("TEACHER_VISION_MODEL") or "").lower()
    try:
        if vision_model and m == vision_model:
            vin = os.environ.get("TEACHER_VISION_PRICE_IN")
            vout = os.environ.get("TEACHER_VISION_PRICE_OUT")
            if vin is not None and vout is not None:
                return float(vin), float(vout)
        # Text-teacher override applies ONLY to the text model (or when no text
        # model is pinned, preserving the original single-endpoint behavior).
        elif not text_model or m == text_model:
            pin = os.environ.get("TEACHER_API_PRICE_IN")
            pout = os.environ.get("TEACHER_API_PRICE_OUT")
            if pin is not None and pout is not None:
                return float(pin), float(pout)
    except Exception:
        pass
    best: "tuple[float, float] | None" = None
    best_len = -1
    for key, price in _PRICE_PER_M.items():
        if key in m and len(key) > best_len:
            best, best_len = price, len(key)
    return best


def _record_usage(body: str, model: str) -> None:
    """Pull ``usage`` from a response body and fold it into the running totals."""
    try:
        obj = json.loads(body)
    except Exception:
        return
    usage = obj.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    price = _price_for(model)
    cost = 0.0
    if price is not None:
        cost = pt / 1_000_000 * price[0] + ct / 1_000_000 * price[1]
    with _USAGE_LOCK:
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += pt
        _USAGE["completion_tokens"] += ct
        _USAGE["cost_usd"] += cost


def usage_snapshot() -> dict:
    """Copy of the running totals: calls, prompt/completion tokens, cost_usd."""
    with _USAGE_LOCK:
        return dict(_USAGE)


def reset_usage() -> None:
    with _USAGE_LOCK:
        _USAGE.update(calls=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)


def format_cost_report() -> str:
    """One-line human summary of teacher spend so far."""
    s = usage_snapshot()
    tin, tout = s["prompt_tokens"], s["completion_tokens"]
    base = (
        f"teacher: {s['calls']:,} calls, "
        f"{tin:,} in + {tout:,} out tokens ({tin + tout:,} total)"
    )
    if _price_for(teacher_model()) is not None or (
        os.environ.get("TEACHER_API_PRICE_IN") and os.environ.get("TEACHER_API_PRICE_OUT")
    ):
        return base + f" — est. ${s['cost_usd']:.2f}"
    return base + " — $ unknown (set TEACHER_API_PRICE_IN/OUT for an estimate)"


def teacher_api_enabled() -> bool:
    """True iff a hosted teacher endpoint is configured. Default-off: with no
    ``TEACHER_API_BASE`` the callers keep using their local Ollama path."""
    return bool(os.environ.get("TEACHER_API_BASE"))


def teacher_model(default: str = _DEFAULT_MODEL) -> str:
    return os.environ.get("TEACHER_MODEL") or default


# ── vision response cache ─────────────────────────────────────────────────────
# Optional sqlite cache for vision calls keyed by sha256(model+prompt+mime+image).
# Vision teacher calls are the expensive ones (Claude/GPT-4o $/token >> text), and
# dev/training re-runs hit the SAME sheets repeatedly. With ``SOWSMITH_VISION_CACHE_DB``
# set, an identical (model, prompt, image) returns the stored reply for $0 and 0
# latency. Default-off (unset) → no behavior change. Best-effort: any cache error
# silently falls through to a live call.
_VCACHE_LOCK = threading.Lock()
_VCACHE_CONN = None
_VCACHE_PATH = None


def _vision_cache_conn():
    global _VCACHE_CONN, _VCACHE_PATH
    path = os.environ.get("SOWSMITH_VISION_CACHE_DB")
    if not path:
        return None
    with _VCACHE_LOCK:
        if _VCACHE_CONN is not None and _VCACHE_PATH == path:
            return _VCACHE_CONN
        try:
            import sqlite3
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vision_cache "
                "(k TEXT PRIMARY KEY, model TEXT, reply TEXT)"
            )
            conn.commit()
            _VCACHE_CONN, _VCACHE_PATH = conn, path
            return conn
        except Exception:
            return None


def _vision_cache_key(model: str, prompt: str, mime: str, image_b64: str) -> str:
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((prompt or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((mime or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((image_b64 or "").encode("utf-8"))
    return h.hexdigest()


def _vision_cache_get(key: str):
    conn = _vision_cache_conn()
    if conn is None:
        return None
    try:
        with _VCACHE_LOCK:
            row = conn.execute("SELECT reply FROM vision_cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _vision_cache_put(key: str, model: str, reply: str) -> None:
    conn = _vision_cache_conn()
    if conn is None or not reply:
        return
    try:
        with _VCACHE_LOCK:
            conn.execute(
                "INSERT OR REPLACE INTO vision_cache (k, model, reply) VALUES (?,?,?)",
                (key, model, reply),
            )
            conn.commit()
    except Exception:
        pass


def _max_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("TEACHER_API_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY))))
    except Exception:
        return _DEFAULT_MAX_CONCURRENCY


def _semaphore() -> threading.Semaphore:
    """Process-wide concurrency gate, re-sized if the env value changes."""
    global _SEM, _SEM_SIZE
    size = _max_concurrency()
    with _SEM_LOCK:
        if _SEM is None or size != _SEM_SIZE:
            _SEM = threading.Semaphore(size)
            _SEM_SIZE = size
        return _SEM


def _api_timeout() -> int:
    for var in ("TEACHER_API_TIMEOUT", "SOWSMITH_LLM_TIMEOUT"):
        v = os.environ.get(var)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    return _DEFAULT_API_TIMEOUT


def pool_budget_seconds(default_per_call: int = _DEFAULT_API_TIMEOUT) -> int:
    """Overall wall-clock budget for a *pool* of teacher calls — the belt to the
    per-call timeout's suspenders. A wedged half-open socket can make a single
    ``urlopen(timeout=...)`` hang past its nominal timeout (seen with a Tailscale
    tunnel to a vanished host); bounding the whole pool guarantees a deal can
    never freeze on the teacher. Override with ``SOWSMITH_LLM_POOL_TIMEOUT``."""
    v = os.environ.get("SOWSMITH_LLM_POOL_TIMEOUT")
    if v:
        try:
            return max(30, int(v))
        except Exception:
            pass
    per = _api_timeout() if teacher_api_enabled() else default_per_call
    return max(per * 3, 600)


def complete(prompt: str, *, max_tokens: int = 1024, model: str | None = None,
             timeout: int | None = None) -> str:
    """One OpenAI-compatible ``/chat/completions`` text call. Returns the
    assistant text, or ``""`` on any failure (so callers degrade exactly as they
    do for a local Ollama miss). No-op (``""``) if no hosted teacher configured."""
    return _chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, model=model, timeout=timeout,
    )


def _clamp_image_b64(image_b64: str, max_side: int = 1568,
                     max_bytes: int = 4_500_000) -> tuple[str, str]:
    """Downscale an image that exceeds a vision model's input limits BEFORE
    sending. Hosted VLMs (Anthropic ~8000px/5MB, OpenAI similar) reject
    oversized images with a 4xx and the call silently returns "" — i.e. a whole
    sheet's data lost with no error. E-size CAD pages at 200 DPI are ~8400px and
    trip this. We resize the long side to ``max_side`` (Anthropic's recommended
    ~1568) when over budget. Returns (b64, mime); always PNG after a resize.
    Best-effort: on any failure return the input unchanged."""
    try:
        approx_bytes = len(image_b64) * 3 // 4
        # quick check on encoded size; decode only if we must inspect dims
        from PIL import Image
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if max(w, h) <= max_side and approx_bytes <= max_bytes:
                return image_b64, "image/png"
            sc = max_side / float(max(w, h))
            im2 = im.convert("RGB").resize((max(1, int(w * sc)), max(1, int(h * sc))))
            buf = io.BytesIO()
            im2.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/png"
    except Exception:
        return image_b64, "image/png"


def complete_vision(prompt: str, image_b64: str, *, mime: str = "image/png",
                    max_tokens: int = 2048, model: str | None = None,
                    timeout: int | None = None) -> str:
    """Multimodal ``/chat/completions`` call: a text prompt + one base64 image,
    sent as an OpenAI-compatible ``image_url`` data-URI (Anthropic's compat
    endpoint accepts this shape). Returns the assistant text, or ``""`` on any
    failure / no image / no hosted teacher. The teacher model must be
    vision-capable (Opus/Sonnet/GPT-4o are); override with ``TEACHER_VISION_MODEL``.

    **Split-route.** Vision routes *independently* of the text teacher so a
    text-only teacher (e.g. DeepSeek) can label prose while a multimodal model
    (Sonnet/GPT-4o) handles images in the same run. The endpoint falls back to
    the text teacher when no vision-specific endpoint is set:

    * ``TEACHER_VISION_API_BASE`` → else ``TEACHER_API_BASE``
    * ``TEACHER_VISION_API_KEY``  → else ``TEACHER_API_KEY``
    * ``TEACHER_VISION_MODEL``    → else ``TEACHER_MODEL``/default

    So with no vision env set, behavior is exactly as before (vision shares the
    text endpoint); set the three vision vars to peel vision onto its own model.
    """
    if not image_b64:
        return ""
    model = model or os.environ.get("TEACHER_VISION_MODEL") or teacher_model()
    # Clamp oversized images so the model never silently rejects + returns "".
    image_b64, mime = _clamp_image_b64(image_b64)
    # Cache hit on identical (model, prompt, image) → $0, no API call.
    ck = _vision_cache_key(model, prompt, mime, image_b64)
    cached = _vision_cache_get(ck)
    if cached is not None:
        return cached
    # Vision endpoint defaults to the text endpoint; override to split routes.
    base = os.environ.get("TEACHER_VISION_API_BASE") or os.environ.get("TEACHER_API_BASE", "")
    key = os.environ.get("TEACHER_VISION_API_KEY") or os.environ.get("TEACHER_API_KEY", "")
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ]
    reply = _chat(
        [{"role": "user", "content": content}],
        max_tokens=max_tokens, model=model, timeout=timeout,
        base=base, key=key,
    )
    _vision_cache_put(ck, model, reply)
    return reply


def _chat(messages: list, *, max_tokens: int, model: str | None,
          timeout: int | None, base: str | None = None,
          key: str | None = None) -> str:
    """Shared OpenAI-compatible chat-completions transport for text and vision:
    bounded retry on 429/5xx/transport errors, global concurrency semaphore,
    usage metering. Returns assistant text or ``""``.

    ``base``/``key`` override the env endpoint for that one call (used by the
    vision split-route); when ``None`` they fall back to ``TEACHER_API_BASE`` /
    ``TEACHER_API_KEY`` so the text path is unchanged."""
    base = (base if base is not None else os.environ.get("TEACHER_API_BASE", "")).rstrip("/")
    if not base:
        return ""
    key = key if key is not None else os.environ.get("TEACHER_API_KEY", "")
    model = model or teacher_model()
    timeout = timeout or _api_timeout()
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    # Preemptively drop params this provider has already told us it rejects, so
    # we don't re-pay a failed round-trip on every call after the first.
    with _UNSUPPORTED_LOCK:
        for name in _UNSUPPORTED_PARAMS:
            payload.pop(name, None)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    retries = _DEFAULT_RETRIES
    backoff = _DEFAULT_BACKOFF
    try:
        retries = int(os.environ.get("TEACHER_API_RETRIES", str(_DEFAULT_RETRIES)))
        backoff = float(os.environ.get("TEACHER_API_BACKOFF", str(_DEFAULT_BACKOFF)))
    except Exception:
        pass

    sem = _semaphore()
    strips_left = len(payload) - len(_ESSENTIAL_PARAMS)  # bound: can't strip forever
    attempt = 0
    while attempt <= retries:
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            sem.acquire()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
            finally:
                sem.release()
            _record_usage(body, model)
            return _extract_text(body)
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_STATUS and attempt < retries:
                _sleep_backoff(backoff, attempt, exc)
                attempt += 1
                continue
            # Self-heal: a 400 naming an unsupported request param → strip it,
            # remember it process-wide, retry WITHOUT consuming the retry budget.
            if exc.code == 400 and strips_left > 0:
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = ""
                drop = _strippable_param(err_body, payload)
                if drop:
                    payload.pop(drop, None)
                    with _UNSUPPORTED_LOCK:
                        _UNSUPPORTED_PARAMS.add(drop)
                    data = json.dumps(payload).encode("utf-8")
                    strips_left -= 1
                    continue
            return ""
        except Exception:
            if attempt < retries:
                _sleep_backoff(backoff, attempt, None)
                attempt += 1
                continue
            return ""
    return ""


def _sleep_backoff(backoff: float, attempt: int, exc: "urllib.error.HTTPError | None") -> None:
    # Honor Retry-After when the provider sends it, else exponential + jitter.
    delay = backoff * (2 ** attempt) + random.random()
    if exc is not None:
        try:
            ra = exc.headers.get("Retry-After") if exc.headers else None
            if ra:
                delay = max(delay, float(ra))
        except Exception:
            pass
    time.sleep(min(delay, 30.0))


def _extract_text(body: str) -> str:
    """Pull the assistant content out of an OpenAI-compatible response body,
    stripping any ``<think>`` reasoning block a model might prepend."""
    try:
        obj = json.loads(body)
    except Exception:
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    text = str(msg.get("content") or "")
    return _THINK_RE.sub("", text).strip()


__all__ = [
    "teacher_api_enabled",
    "teacher_model",
    "pool_budget_seconds",
    "complete",
    "complete_vision",
    "usage_snapshot",
    "reset_usage",
    "format_cost_report",
]
