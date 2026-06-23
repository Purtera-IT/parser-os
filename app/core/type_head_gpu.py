"""Runtime atom-type deflector — the GPU fine-tuned transformer head (#70).

The CPU :mod:`app.core.type_head` head is an sklearn LR over FROZEN embeddings and
caps at ~0.65 held-out (the frozen space can't separate the 43 overlapping types).
This module serves the GPU-fine-tuned encoder
(:mod:`runpod_detector.train_type_head_gpu`, held-out **0.814**, 53% of atoms
deflectable @ **0.966** precision) — a HuggingFace
``AutoModelForSequenceClassification`` saved to ``<dir>/best`` alongside a
``labels.json`` (the index-aligned class list).

Same runtime contract as the CPU head — ``classify_batch(texts) -> [(type, conf) |
None]`` — so it drops into the existing value-light deflection seam in
``typed_atom_classifier``. Guess-free: abstains on a ``_keep`` prediction or below
the confidence bar, so a wrong abstain just falls through to the LLM as today.

SAFE TO SHIP EVEN IF NOT READY: if torch/transformers or the model dir are absent,
every call abstains -> byte-identical to the LLM-only path. OFF by default
(``SOWSMITH_TYPE_HEAD_GPU``); flip on once the worker has the deps + the model
(fetched from blob ``type_head_gpu.tgz`` -> ``SOWSMITH_TYPE_HEAD_GPU_DIR``).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

_KEEP = "_keep"
_DEFAULT_CONF = 0.85   # trainer cutover measured here: ~53% deflect @ 0.966 precision
_lock = threading.Lock()
_holder: dict[str, Any] = {}


def _head_dir() -> str:
    return os.environ.get("SOWSMITH_TYPE_HEAD_GPU_DIR", "_type_head_gpu/best")


def _conf_bar() -> float:
    try:
        return float(os.environ.get("SOWSMITH_TYPE_HEAD_GPU_CONF", str(_DEFAULT_CONF)))
    except ValueError:
        return _DEFAULT_CONF


def _load():
    """Lazy-load (model, tok, labels, torch) once. Returns None on any failure
    (missing deps / missing model dir / missing labels) — caller then abstains."""
    if "loaded" in _holder:
        return _holder["loaded"]
    with _lock:
        if "loaded" in _holder:
            return _holder["loaded"]
        result = None
        try:
            d = _head_dir()
            lp = os.path.join(d, "labels.json")
            if os.path.isdir(d) and os.path.exists(lp):
                labels = json.load(open(lp, encoding="utf-8")).get("labels")
                if labels:
                    import torch
                    from transformers import (AutoModelForSequenceClassification,
                                              AutoTokenizer)
                    tok = AutoTokenizer.from_pretrained(d)
                    model = AutoModelForSequenceClassification.from_pretrained(d)
                    model.eval()
                    result = (model, tok, list(labels), torch)
        except Exception:
            result = None
        _holder["loaded"] = result
        return result


def classify_batch(texts: list[str], *, batch_size: int = 64) -> list[tuple[str, float] | None]:
    """Per text: ``(specific_type, conf)`` to DEFLECT off the LLM, or ``None`` to
    abstain (caller -> LLM). Guess-free: abstains on a ``_keep`` prediction or below
    the confidence bar. All-``None`` on any failure. Never raises."""
    n = len(texts)
    if not n:
        return []
    loaded = _load()
    if loaded is None:
        return [None] * n
    model, tok, labels, torch = loaded
    bar = _conf_bar()
    out: list[tuple[str, float] | None] = []
    try:
        with torch.no_grad():
            for i in range(0, n, batch_size):
                chunk = texts[i:i + batch_size]
                enc = tok(chunk, truncation=True, max_length=128,
                          padding=True, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
                for p in probs:
                    j = int(p.argmax())
                    label, conf = labels[j], float(p[j])
                    out.append((label, conf) if label != _KEEP and conf >= bar else None)
    except Exception:
        return [None] * n
    return out


def classify(text: str) -> tuple[str, float] | None:
    """Single-text convenience wrapper around :func:`classify_batch`."""
    return classify_batch([text])[0]


def is_ready() -> bool:
    """True if the GPU type head is loadable (for diagnostics / flag gating)."""
    return _load() is not None
