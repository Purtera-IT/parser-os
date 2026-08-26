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
"""
from __future__ import annotations

import os
import threading
from typing import Any

from app.core.pdf_image_gate import gate_feature_text

_DEFAULT_CONF = 0.88
_lock = threading.Lock()
_holder: dict[str, Any] = {}


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


def _load():
    if "loaded" in _holder:
        return _holder["loaded"]
    with _lock:
        if "loaded" in _holder:
            return _holder["loaded"]
        result = None
        try:
            d = _veto_dir()
            if os.path.isdir(d):
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(d)
                model = AutoModelForSequenceClassification.from_pretrained(d)
                model.eval()
                id2label = {
                    int(k): v for k, v in (model.config.id2label or {}).items()
                }
                result = (model, tok, torch, id2label)
        except Exception:
            result = None
        _holder["loaded"] = result
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


def veto(caption: str, ocr: str) -> float | None:
    """Return P(meaningful) when the head confidently disagrees with a skip,
    else None (abstain — no veto).

    Fires only when the model is present AND the probability clears the
    confidence bar. A degenerate feature ("no context") never vetoes — there
    is no evidence to disagree on. Any failure -> None (guess-free).
    """
    if not enabled():
        return None
    text = gate_feature_text(caption, ocr)
    if not text.strip() or text == "no context":
        return None
    loaded = _load()
    if loaded is None:
        return None
    try:
        prob = _meaningful_prob(loaded, text)
    except Exception:
        return None
    if prob is None or prob < _conf_bar():
        return None
    return prob


def is_ready() -> bool:
    return _load() is not None
