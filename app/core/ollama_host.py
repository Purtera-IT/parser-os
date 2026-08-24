"""Where the local Ollama lives, and whether it can actually answer.

Two defects this module exists to close, both found by reading the dev
worker's configuration against the code that consumes it.

**The host is configured but never read.** Nine modules resolve the local
Ollama with ``os.environ.get("OLLAMA_HOST", "http://100.114.102.122:11434")``.
The worker sets ``OLLAMA_BASE_URL`` — the mac proxy — and does not set
``OLLAMA_HOST``. So in production every one of those paths ignores the proxy
and dials a hardcoded Tailscale address from inside Azure. Same story for the
embedder: the worker sets ``OLLAMA_EMBED_URL`` and nothing reads it, which is
why compile telemetry shows ``semantic_dedupe_embedder: deterministic-hash-v1``
where it once showed ``qwen3-embedding:8b``.

**The health check lies.** ``/api/tags`` answers in milliseconds from a host
whose generate endpoint never returns — a model can be resident, and pinned in
memory, with no headroom left to run. Listing models proves the server is up,
not that it can serve. Today that is exactly the state of the box: TCP connect
0.02s, ``/api/tags`` 200 in 0.05s listing seven models, and five tokens out of
``qwen3:14b`` timing out at 25s, while ``qwen2.5:3b`` answers in 0.41s.

The consequence is not an error. Callers use a 180-second per-call timeout, so
a wedged model turns every LLM stage into a three-minute stall that reports
nothing. ``generation_ready`` asks the only question that matters — give me
five tokens, now — once per host and model, and lets the caller fall to its
deterministic path in seconds instead.

Pure environment resolution plus one bounded probe. No retries: a host that
cannot produce five tokens inside the preflight budget is not a host this
compile should wait on.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

# The address every module hardcoded before this one existed. Kept as the
# last resort so behaviour is unchanged when nothing is configured.
FALLBACK_HOST = "http://100.114.102.122:11434"

# Five tokens is enough to prove the model can run; the budget is deliberately
# far below the 180s per-call timeout, because the whole point is to find out
# cheaply.
_DEFAULT_PREFLIGHT_TIMEOUT = 8
_DEFAULT_PREFLIGHT_TOKENS = 5

# How long a preflight verdict stands before it is re-probed. Long enough that
# a compile pays for it once, short enough that a box coming back up is picked
# up without a redeploy.
_DEFAULT_PREFLIGHT_TTL = 300

_lock = threading.Lock()
_verdicts: dict[tuple, tuple[bool, float]] = {}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def resolve_host(default: str = FALLBACK_HOST) -> str:
    """Return the base URL of the local Ollama.

    ``OLLAMA_HOST`` first, because anything that sets it means it. Then
    ``OLLAMA_BASE_URL``, which is what the container app actually sets. Then
    the caller's own default, so a module that has always shipped its own
    constant keeps it.
    """
    for var in ("OLLAMA_HOST", "OLLAMA_BASE_URL"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value.rstrip("/")
    return (default or FALLBACK_HOST).rstrip("/")


def resolve_embed_host(default: str = FALLBACK_HOST) -> str:
    """Return the base URL for embedding calls.

    ``OLLAMA_EMBED_URL`` is configured as a full endpoint
    (``https://…/api/embed``) while every call site builds its own path, so
    the endpoint suffix is trimmed back to a base here rather than at each
    caller.
    """
    raw = (os.environ.get("OLLAMA_EMBED_URL") or "").strip()
    if not raw:
        return resolve_host(default)
    base = raw.rstrip("/")
    for suffix in ("/api/embeddings", "/api/embed"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def preflight_timeout() -> int:
    return _int_env("PARSER_OS_LLM_PREFLIGHT_TIMEOUT", _DEFAULT_PREFLIGHT_TIMEOUT)


def generation_ready(host: str, model: str, *, timeout: int | None = None) -> bool:
    """Can ``model`` on ``host`` produce a few tokens right now?

    Cached per host and model for ``PARSER_OS_LLM_PREFLIGHT_TTL`` seconds, so
    a compile pays the probe once rather than per atom. ``SOWSMITH_DISABLE_LLM``
    short-circuits to ``False`` — nothing should be dialled at all.

    Any failure is a ``False``: timeout, connection refused, non-200, malformed
    body, or a 200 carrying no text. The caller's contract is unchanged — it
    treats a falsy result exactly as it treats an empty completion, and takes
    its deterministic path.
    """
    if os.environ.get("SOWSMITH_DISABLE_LLM"):
        return False
    if not host or not model:
        return False

    key = ("generate", host.rstrip("/"), model)
    ttl = _int_env("PARSER_OS_LLM_PREFLIGHT_TTL", _DEFAULT_PREFLIGHT_TTL)
    now = time.monotonic()
    with _lock:
        cached = _verdicts.get(key)
        if cached is not None and (now - cached[1]) < ttl:
            return cached[0]

    verdict = _probe(key[1], model, timeout if timeout is not None else preflight_timeout())
    with _lock:
        _verdicts[key] = (verdict, time.monotonic())
    return verdict


def embed_ready(host: str, model: str, *, timeout: int | None = None) -> bool:
    """Can ``model`` on ``host`` return an embedding right now?

    The generate preflight above does not cover this: embeddings go to
    ``/api/embeddings``, so a box that cannot generate may still embed, and a box
    that can generate may have unloaded the embed model. They are separate
    verdicts and must be probed separately.

    This gap is why the generate fix was incomplete. Every generate caller
    degrades in seconds while ``embed_texts`` still paid the full
    ``SOWSMITH_EMBED_TIMEOUT`` — 180s by default — on every call, and
    ``enrich_atoms`` embeds repeatedly. One deal sat in ``enrich_entities`` for
    21 minutes against a dead endpoint with nothing to short-circuit it.
    """
    if os.environ.get("SOWSMITH_DISABLE_LLM"):
        return False
    if not host or not model:
        return False

    key = ("embed", host.rstrip("/"), model)
    ttl = _int_env("PARSER_OS_LLM_PREFLIGHT_TTL", _DEFAULT_PREFLIGHT_TTL)
    now = time.monotonic()
    with _lock:
        cached = _verdicts.get(key)
        if cached is not None and (now - cached[1]) < ttl:
            return cached[0]

    verdict = _probe_embed(
        key[1], model, timeout if timeout is not None else preflight_timeout()
    )
    with _lock:
        _verdicts[key] = (verdict, time.monotonic())
    return verdict


def _probe_embed(host: str, model: str, timeout: int) -> bool:
    payload = {"model": model, "prompt": "ping"}
    req = urllib.request.Request(
        f"{host}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return False
    try:
        vector = json.loads(body).get("embedding")
        return bool(isinstance(vector, list) and vector)
    except (json.JSONDecodeError, AttributeError):
        return False


def _probe(host: str, model: str, timeout: int) -> bool:
    payload = {
        "model": model,
        "prompt": "ping",
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": _DEFAULT_PREFLIGHT_TOKENS},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return False
    try:
        return bool(str(json.loads(body).get("response") or "").strip())
    except (json.JSONDecodeError, AttributeError):
        return False


def reset_preflight_cache() -> None:
    """Drop cached verdicts. For tests, and for a caller that has just been
    told the host changed."""
    with _lock:
        _verdicts.clear()


__all__ = [
    "FALLBACK_HOST",
    "embed_ready",
    "generation_ready",
    "preflight_timeout",
    "reset_preflight_cache",
    "resolve_embed_host",
    "resolve_host",
]
