"""CPU gate for PDF embedded-image triage — distills the VLM classify step.

Trained on ``pdf_image_kind`` rows in the TrainingLog (silver from VLM gate
decisions + gold from PM chip corrections on the ``image`` head). At runtime
when ``SOWSMITH_PDF_IMAGE_GATE_CPU=1`` and the model is present, this replaces
the cheap VLM gate call. Any failure or low confidence -> abstain -> VLM gate
runs exactly as today (guess-free).

Feature text is caption + OCR snippet (no image tensor — keeps the head tiny
and CPU-fast, same philosophy as rubric_gate).
"""
from __future__ import annotations

import os
import threading
from typing import Any

_SKIP_LABELS = frozenset({"skip", "logo", "decorative", "signature", "empty"})
_DEFAULT_CONF = 0.88
_lock = threading.Lock()
_holder: dict[str, Any] = {}


def enabled() -> bool:
    return os.environ.get("SOWSMITH_PDF_IMAGE_GATE_CPU", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _gate_dir() -> str:
    return os.environ.get(
        "SOWSMITH_PDF_IMAGE_GATE_DIR", "/tmp/ml/_pdf_image_gate/best",
    )


def _conf_bar() -> float:
    try:
        return float(os.environ.get("SOWSMITH_PDF_IMAGE_GATE_CONF", str(_DEFAULT_CONF)))
    except ValueError:
        return _DEFAULT_CONF


def gate_feature_text(caption: str, ocr: str, *, max_ocr: int = 500) -> str:
    parts: list[str] = []
    if caption.strip():
        parts.append(f"caption: {caption.strip()}")
    if ocr.strip():
        parts.append(f"ocr: {ocr.strip()[:max_ocr]}")
    return "\n".join(parts) or "no context"


def _load():
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
                id2label = {
                    int(k): v for k, v in (model.config.id2label or {}).items()
                }
                result = (model, tok, torch, id2label)
        except Exception:
            result = None
        _holder["loaded"] = result
        return result


def classify(caption: str, ocr: str) -> tuple[bool, str] | None:
    """Return (meaningful, image_kind) or None to abstain (fall through to VLM)."""
    if not enabled():
        return None
    text = gate_feature_text(caption, ocr)
    loaded = _load()
    if loaded is None:
        return None
    model, tok, torch, id2label = loaded
    bar = _conf_bar()
    try:
        with torch.no_grad():
            enc = tok([text], truncation=True, max_length=256,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        if conf < bar:
            return None
        label = str(id2label.get(pred_idx, "")).strip().lower()
        if not label:
            return None
        if label in _SKIP_LABELS or label == "skip":
            return False, label
        return True, label
    except Exception:
        return None


def is_ready() -> bool:
    return _load() is not None
