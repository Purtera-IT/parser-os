"""Universal semantic role classifier (small local LLM, no keyword rules).

Several parser/enrichment defects share one shape: a *lexical* gate is asked
to make a *semantic* decision. A regex that finds ``City, ST ZIP`` cannot know
whether that address is the **job site** the work happens at or the vendor's
**letterhead/billing** address; a keyword list of "material/bom/equipment"
cannot know whether a priced sheet is an ordered bill of materials or a master
price book. Keyword lists never generalise — the next deal phrases it
differently and the gate misfires.

This module provides one reusable primitive, :func:`classify_role`, that asks
a *small* local LLM (default ``qwen2.5:3b`` via Ollama) to pick the best role
from a caller-supplied candidate set, returning ``(role, confidence)``.

Design contract — it must be safe to call anywhere, including offline:

* **Never raises.** Any transport / parse error returns ``(None, 0.0)``.
* **Fails closed on reachability.** The first failed call flips a process-wide
  flag so subsequent calls short-circuit instead of paying the timeout on
  every atom. Callers treat ``(None, 0.0)`` as "undecided" and fall back to
  their own conservative default (typically: don't change anything).
* **Cached.** Identical ``(text, candidates)`` questions are answered once.
* **Deterministic in tests.** Monkeypatch :func:`classify_role` (or set
  ``SOWSMITH_DISABLE_LLM=1``) to exercise the deterministic fallback paths.
"""

from __future__ import annotations

import json
import os
import urllib.request

# Reuse the same Ollama endpoint the rest of the pipeline targets.
from app.core.multi_entity_llm import DEFAULT_HOST

# A deliberately tiny, fast, NON-thinking model — role classification is a
# one-token decision, not a reasoning task. Overridable per deployment.
DEFAULT_ROLE_MODEL = "qwen2.5:3b"
# Short timeout: a 3B model answers a one-word classification in well under a
# second on a warm host; we do not want to stall the compile if it is cold.
DEFAULT_ROLE_TIMEOUT = 20

# Process-wide reachability latch + answer cache.
_llm_unreachable = False
_cache: dict[tuple[str, tuple[str, ...]], tuple[str | None, float]] = {}


def reset_reachability() -> None:
    """Test hook: clear the unreachable latch and the answer cache."""
    global _llm_unreachable
    _llm_unreachable = False
    _cache.clear()


def _llm_disabled() -> bool:
    return os.environ.get("SOWSMITH_DISABLE_LLM", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _post_generate(prompt: str, *, timeout: int, model: str | None = None) -> str:
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = model or os.environ.get("OLLAMA_ROLE_MODEL") or DEFAULT_ROLE_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 64},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return str(json.loads(body).get("response") or "")


def classify_role(
    text: str,
    candidates: list[str],
    *,
    instruction: str,
    context: str = "",
    timeout: int | None = None,
    model: str | None = None,
) -> tuple[str | None, float]:
    """Pick the best-fitting role for ``text`` from ``candidates``.

    Args:
        text: the snippet whose role is in question (e.g. an address line).
        candidates: the closed set of allowed roles (e.g.
            ``["job_site", "vendor_address"]``). The model is told to choose
            exactly one of these or ``"unknown"``.
        instruction: a one-line description of the decision, so the same
            primitive serves addresses, sheets, line items, etc.
        context: optional surrounding text that disambiguates the snippet.
        timeout: optional per-call timeout override (seconds).
        model: optional Ollama model override for this call. Lets a caller
            route a hard discrimination to a stronger model (e.g.
            ``qwen3:14b``) while cheap high-volume calls keep the tiny
            default. ``OLLAMA_ROLE_MODEL`` is used when this is ``None``.

    Returns:
        ``(role, confidence)`` where ``role`` is one of ``candidates`` and
        ``confidence`` is in ``[0.0, 1.0]``; ``(None, 0.0)`` when the model is
        disabled, unreachable, or returns nothing usable. Callers MUST treat
        ``None`` as "undecided" and apply their own safe default.
    """
    global _llm_unreachable

    if not text or not candidates:
        return (None, 0.0)
    if _llm_disabled() or _llm_unreachable:
        return (None, 0.0)

    key = ("|".join([model or "", instruction, context, text]), tuple(candidates))
    if key in _cache:
        return _cache[key]

    allowed = [c for c in candidates if isinstance(c, str) and c]
    allowed_with_unknown = allowed + ["unknown"]
    prompt = (
        "You are a precise classifier. Choose the single best role for the "
        "TEXT below.\n"
        f"Decision: {instruction}\n"
        f"Allowed roles (choose exactly one): {', '.join(allowed_with_unknown)}\n"
        "If you cannot tell, choose \"unknown\".\n"
        + (f"Context:\n{context.strip()[:1200]}\n" if context else "")
        + f"TEXT:\n{text.strip()[:600]}\n\n"
        "Respond with ONLY a JSON object of the form "
        '{"role": "<one allowed role>", "confidence": <0.0-1.0>}.'
    )

    t = int(timeout if timeout is not None else DEFAULT_ROLE_TIMEOUT)
    try:
        raw = _post_generate(prompt, timeout=t, model=model)
    except Exception:
        # First failure latches: don't retry the timeout on every atom.
        _llm_unreachable = True
        result = (None, 0.0)
        _cache[key] = result
        return result

    result = _parse_response(raw, allowed)
    _cache[key] = result
    return result


def _parse_response(raw: str, allowed: list[str]) -> tuple[str | None, float]:
    raw = (raw or "").strip()
    if not raw:
        return (None, 0.0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return (None, 0.0)
    if not isinstance(obj, dict):
        return (None, 0.0)
    role = obj.get("role")
    if not isinstance(role, str):
        return (None, 0.0)
    role = role.strip()
    # Case-insensitive match back onto the caller's exact candidate spelling.
    lc = {c.lower(): c for c in allowed}
    if role.lower() not in lc:
        return (None, 0.0)  # "unknown" or anything off-menu → undecided
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return (lc[role.lower()], conf)


__all__ = ["classify_role", "reset_reachability"]
