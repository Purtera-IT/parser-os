"""Runtime GPU span taggers (#71 deploy) — recall-tuned per-relation admission
heads that let us SKIP the enrich LLM call for verbatim-value relations.

GPU fine-tuning lifted held-out recall past the 0.93 skip bar: requirements 0.947,
site_clusters 1.0 (commercial 0.875 — not yet, stays on the LLM). At runtime, for a
skippable verbatim relation the tagger scans the FULL atom set and admits atoms
above its saved threshold; since the value is verbatim (the atom text IS the
requirement / site name), an admitted atom needs no LLM.

Guess-free + safe:
  - only VERBATIM relations, only when the saved held-out recall >= SKIP bar.
  - missing torch/transformers/models/meta -> abstain (empty) -> the caller falls
    back to the CPU span head / the LLM. Byte-identical to today when OFF.
OFF by default (SOWSMITH_SPAN_GPU). Models fetched from blob span_heads_gpu.tgz ->
extracted under SOWSMITH_SPAN_GPU_DIR (per relation: span_<rel>/best/).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

_VERBATIM = {"requirements": "text", "site_clusters": "name"}
_SKIP_BAR = 0.93
_lock = threading.Lock()
_cache: dict[str, Any] = {}


def enabled() -> bool:
    return os.environ.get("SOWSMITH_SPAN_GPU", "").strip().lower() in ("1", "true", "yes", "on")


def _base() -> str:
    return os.environ.get("SOWSMITH_SPAN_GPU_DIR", "_span_gpu")


def _rel_dir(rel: str) -> str | None:
    base = _base()
    for cand in (os.path.join(base, f"span_{rel}", "best"),
                 os.path.join(base, rel, "best"),
                 os.path.join("runs", f"span_{rel}", "best")):
        if os.path.isdir(cand):
            return cand
    return None


def _load(rel: str):
    """Lazy-load (model, tok, torch, meta) for a relation, or None on any failure."""
    key = rel
    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:
            return _cache[key]
        result = None
        try:
            d = _rel_dir(rel)
            mp = os.path.join(d, "span_meta.json") if d else None
            if d and mp and os.path.exists(mp):
                meta = json.load(open(mp, encoding="utf-8"))
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(d)
                model = AutoModelForSequenceClassification.from_pretrained(d)
                model.eval()
                result = (model, tok, torch, meta)
        except Exception:
            result = None
        _cache[key] = result
        return result


def has(rel: str) -> bool:
    """True if a usable GPU tagger (model + meta) is loadable for this relation."""
    return enabled() and rel in _VERBATIM and _load(rel) is not None


def gpu_skip_relations() -> dict[str, float]:
    """Verbatim relations the GPU tagger certifies skippable (saved recall >= bar)."""
    if not enabled():
        return {}
    out: dict[str, float] = {}
    for rel in _VERBATIM:
        loaded = _load(rel)
        if loaded is None:
            continue
        r = float(loaded[3].get("recall", 0.0))
        if r >= _SKIP_BAR:
            out[rel] = r
    return out


def gpu_admit(atoms: list, rel: str, *, batch_size: int = 64) -> list[str]:
    """Verbatim texts admitted by the tagger for ``rel`` (softmax[:,1] >= threshold).
    Empty on any failure (abstain). Never raises."""
    loaded = _load(rel)
    if loaded is None or rel not in _VERBATIM:
        return []
    model, tok, torch, meta = loaded
    thr = float(meta.get("threshold", 0.5))
    texts = [(getattr(a, "raw_text", "") or "").strip() for a in atoms]
    idx = [i for i, t in enumerate(texts) if t]
    out: list[str] = []
    try:
        with torch.no_grad():
            for s in range(0, len(idx), batch_size):
                chunk_idx = idx[s:s + batch_size]
                chunk = [texts[i] for i in chunk_idx]
                enc = tok(chunk, truncation=True, max_length=128, padding=True, return_tensors="pt")
                prob = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()[:, 1]
                for i, p in zip(chunk_idx, prob):
                    if float(p) >= thr:
                        out.append(texts[i])
    except Exception:
        return []
    return out
