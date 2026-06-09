"""Runtime keep-vs-typed GATE — the 0.864 bge-base rubric classifier (#70 cutover).

This is the deployable half of B: the gate trained on rubric-consistent labels
(`train_gate_rubric.py`, held-out acc 0.864, recall 0.96 on typed). At the typing
seam it DEFLECTS confidently-`_keep` atoms off the LLM stage — they stay `_keep`,
skipping the ~98s qwen call for that slice.

Guess-free + safe BY DIRECTION: it only ever acts on a confident `_keep` verdict
(it never emits a positive type here), so a wrong abstain just falls through to the
LLM exactly as today. The confidence bar is deliberately high (softmax ~0.95 maps
to ~0.91 real precision on this model, so default 0.97 targets ~0.95 true
precision on the deflected slice).

SAFE TO SHIP EVEN IF NOT READY: if torch/transformers or the model dir are absent,
every call abstains -> byte-identical to the LLM-only path. OFF by default
(`SOWSMITH_RUBRIC_GATE`); flip on once the worker has the deps + the model
(fetched from blob `gate_rubric_best.tgz` -> `SOWSMITH_RUBRIC_GATE_DIR`).
"""
from __future__ import annotations

import os
import threading
from typing import Any

_KEEP_INDEX = 0   # train_gate_rubric labels: 0 = _keep, 1 = typed
_DEFAULT_CONF = 0.97
_lock = threading.Lock()
_holder: dict[str, Any] = {}


def _gate_dir() -> str:
    return os.environ.get("SOWSMITH_RUBRIC_GATE_DIR", "_gate_rubric/best")


def _conf_bar() -> float:
    try:
        return float(os.environ.get("SOWSMITH_RUBRIC_GATE_CONF", str(_DEFAULT_CONF)))
    except ValueError:
        return _DEFAULT_CONF


def _load():
    """Lazy-load model+tokenizer once. Returns (model, tok, torch) or None on any
    failure (missing deps / missing model dir) — caller then abstains."""
    if "loaded" in _holder:
        return _holder["loaded"]
    with _lock:
        if "loaded" in _holder:
            return _holder["loaded"]
        result = None
        try:
            d = _gate_dir()
            if os.path.isdir(d):
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(d)
                model = AutoModelForSequenceClassification.from_pretrained(d)
                model.eval()
                result = (model, tok, torch)
        except Exception:
            result = None
        _holder["loaded"] = result
        return result


def keep_deflect_flags(texts: list[str], *, batch_size: int = 64) -> list[bool]:
    """For each text, True iff the gate is confident it's `_keep` (deflect off the
    LLM). All-False on any failure (abstain). Never raises."""
    n = len(texts)
    if not n:
        return []
    loaded = _load()
    if loaded is None:
        return [False] * n
    model, tok, torch = loaded
    bar = _conf_bar()
    out: list[bool] = []
    try:
        with torch.no_grad():
            for i in range(0, n, batch_size):
                chunk = texts[i:i + batch_size]
                enc = tok(chunk, truncation=True, max_length=128,
                          padding=True, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
                for p in probs:
                    keep_idx = int(p.argmax())
                    out.append(keep_idx == _KEEP_INDEX and float(p[keep_idx]) >= bar)
    except Exception:
        return [False] * n
    return out


def is_ready() -> bool:
    """True if the gate model is loadable (for diagnostics / flag gating)."""
    return _load() is not None
