"""Skip veto for PDF embedded-image triage — a recorded second opinion.

A trained binary (meaningful vs skip) text classifier over the SAME
``gate_feature_text(caption, ocr)`` feature as :mod:`app.core.pdf_image_gate`.
Its first production job is catching WRONG skips (evidence loss), not
deflecting VLM calls: when the VLM or store gate rules an image skip and this
head is confident the image is meaningful, the disagreement is flagged for the
parser review queue. It NEVER changes routing — a vetoed skip is still
processed as a skip; the veto is a recorded flag on the verdict.

Mirrors the gate's guess-free contract exactly: OFF by default
(``SOWSMITH_PDF_IMAGE_VETO``), lazy singleton load, and model missing /
degenerate feature / low confidence / any failure -> abstain (``None``,
meaning no veto).

An abstain is not silence. :func:`check` is the production entry point: ONE
forward pass that returns both the band decision and a liveness receipt
(``model`` state + the probability the head actually produced), so
"the head ran and agreed with the skip" and "there is no head" stop looking
identical from outside (doctrine: silent-zero-for-infra, see
``docs/IMAGE_GATE_LOOP.md``). :func:`veto` / :func:`soft_veto` are thin band
readers over the same call.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from app.core.pdf_image_gate import gate_feature_text

_DEFAULT_CONF = 0.88
_DEFAULT_SOFT_CONF = 0.70
_lock = threading.Lock()
_holder: dict[str, Any] = {}

# Model states for the liveness receipt. 'absent' is a FACT (the checkpoint
# directory is not there, so this head can never fire); 'unknown' is the
# guess-free fallback for "we could not determine which" — never a stand-in
# for a verdict.
_STATE_LOADED = "loaded"
_STATE_ABSENT = "absent"
_STATE_UNKNOWN = "unknown"


def enabled() -> bool:
    return os.environ.get("SOWSMITH_PDF_IMAGE_VETO", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _veto_dir() -> str:
    return os.environ.get(
        "SOWSMITH_PDF_IMAGE_VETO_DIR", "/tmp/ml/_pdf_image_veto/best",
    )


def _conf_bar() -> float:
    try:
        return float(os.environ.get("SOWSMITH_PDF_IMAGE_VETO_CONF", str(_DEFAULT_CONF)))
    except ValueError:
        return _DEFAULT_CONF


def _soft_conf_bar() -> float:
    try:
        return float(os.environ.get(
            "SOWSMITH_PDF_IMAGE_VETO_SOFT_CONF", str(_DEFAULT_SOFT_CONF),
        ))
    except ValueError:
        return _DEFAULT_SOFT_CONF


def _load():
    if "loaded" in _holder:
        return _holder["loaded"]
    with _lock:
        if "loaded" in _holder:
            return _holder["loaded"]
        result = None
        state = _STATE_UNKNOWN
        try:
            d = _veto_dir()
            if not os.path.isdir(d):
                state = _STATE_ABSENT  # checkpoint not shipped: it CANNOT fire
            else:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(d)
                model = AutoModelForSequenceClassification.from_pretrained(d)
                model.eval()
                id2label = {
                    int(k): v for k, v in (model.config.id2label or {}).items()
                }
                result = (model, tok, torch, id2label)
                state = _STATE_LOADED
        except Exception:
            result = None
            state = _STATE_UNKNOWN  # a load that BLEW UP is not the same as absent
        _holder["loaded"] = result
        _holder["state"] = state
        return result


def _meaningful_prob(loaded, text: str) -> float | None:
    """P(meaningful) for one feature text, or None when the checkpoint has no
    'meaningful' label (a mis-shipped model must never veto)."""
    model, tok, torch, id2label = loaded
    idx = next(
        (i for i, lbl in id2label.items()
         if str(lbl).strip().lower() == "meaningful"),
        None,
    )
    if idx is None:
        return None
    with torch.no_grad():
        enc = tok([text], truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()[0]
    return float(probs[idx])


def check(caption: str, ocr: str) -> dict[str, Any]:
    """Consult the head ONCE for one skip verdict and report what happened.

    Returns a receipt dict, never raises, never guesses::

        {"model": "loaded" | "absent" | "unknown",
         "meaningful_prob": float | None,
         "band": "hard" | "soft" | None,
         "reason": str}              # only when meaningful_prob is None

    ``band`` is what the caller ROUTES on (``None`` = abstain, no veto).
    ``model`` / ``meaningful_prob`` are what the caller RECORDS: a receipt
    reading ``loaded, 0.03`` proves the head ran and agreed with the skip —
    something a bare abstain cannot, since a missing checkpoint abstains the
    same way. ``model`` describes the checkpoint only; ``reason`` says why a
    probability is missing (``disabled``, ``no_context``, ``model_absent``,
    ``load_failed``, ``no_meaningful_label``, ``infer_failed``,
    ``check_failed``), so "loaded but mis-shipped" is observable too.

    One forward pass serves both bands — :func:`veto` and :func:`soft_veto`
    read this, and the lazy singleton is loaded at most once per process.
    """
    def _receipt(
        model: str, *, prob: float | None = None, band: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": model, "meaningful_prob": prob, "band": band,
        }
        if reason:
            out["reason"] = reason
        return out

    try:
        if not enabled():
            # Configured OFF: we never looked, so we cannot claim a state.
            return _receipt(_STATE_UNKNOWN, reason="disabled")
        text = gate_feature_text(caption, ocr)
        if not text.strip() or text == "no context":
            # Degenerate feature: nothing to disagree ON. Still report the
            # checkpoint's state — that is the half of the receipt that says
            # whether a head exists at all.
            return _receipt(model_state(), reason="no_context")
        loaded = _load()
        state = model_state()
        if loaded is None:
            return _receipt(state, reason=(
                "model_absent" if state == _STATE_ABSENT else "load_failed"
            ))
        try:
            prob = _meaningful_prob(loaded, text)
        except Exception:
            return _receipt(state, reason="infer_failed")
        if prob is None:
            # Checkpoint present but mis-shipped (no 'meaningful' label): it
            # is loaded and can never fire — say both.
            return _receipt(state, reason="no_meaningful_label")
        band = None
        if prob >= _conf_bar():
            band = "hard"
        elif prob >= _soft_conf_bar():
            band = "soft"
        return _receipt(state, prob=prob, band=band)
    except Exception:
        return _receipt(_STATE_UNKNOWN, reason="check_failed")


def veto(caption: str, ocr: str) -> float | None:
    """Return P(meaningful) when the head confidently disagrees with a skip,
    else None (abstain — no veto).

    Fires only when the model is present AND the probability clears the
    confidence bar. A degenerate feature ("no context") never vetoes — there
    is no evidence to disagree on. Any failure -> None (guess-free).
    """
    rec = check(caption, ocr)
    return rec["meaningful_prob"] if rec["band"] == "hard" else None


def soft_veto(caption: str, ocr: str) -> float | None:
    """Return P(meaningful) when it lands in the UNCERTAIN band
    [soft_bar, hard_bar) — an active-learning harvest signal, never a PM flag.

    A soft veto marks the model's uncertain zone so humans grade exactly the
    examples the head cannot call — it must NEVER become a hard veto downstream
    (no ``gate_verdict['veto']`` key, no PM culprit card). Same guess-free
    contract as :func:`veto`: any failure -> None (no soft veto).
    """
    rec = check(caption, ocr)
    return rec["meaningful_prob"] if rec["band"] == "soft" else None


def model_state() -> str:
    """``'loaded'`` | ``'absent'`` | ``'unknown'`` — what the lazy singleton
    actually holds, for the liveness receipt.

    Reuses the same one-shot load as :func:`veto` / :func:`is_ready`, so
    asking never costs a second load. A model object in hand IS ``loaded``;
    with nothing in hand, only the load itself can say which kind of nothing
    it was, and anything it did not record stays ``unknown``.
    """
    if _load() is not None:
        return _STATE_LOADED
    state = _holder.get("state")
    return str(state) if state in (_STATE_ABSENT, _STATE_UNKNOWN) else _STATE_UNKNOWN


def is_ready() -> bool:
    return _load() is not None
