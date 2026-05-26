"""Vision-LLM symbol detector — Phase A unlock.

Today's ``detect_symbols`` finds text strings matching legend
symbols ("CR" / "PTZ" / "WN"). On real DD/CD drawings the symbol IS
an icon, not a text tag — so the text-matching detector returns 0
on the Marriott Atlanta DD even though the legend has 30 distinct
symbols.

This module sends each region-proposal crop (from
``region_proposals.propose_regions``) to a vision-LLM with the
legend symbol crops as visual context. The LLM matches the region
to one of the known legend entries (or rejects it as "no_match")
and returns a tight bounding box + confidence.

Architecture:

  legend_symbol_crops + region_proposals
                         ↓
            vision_symbol_detector
                         ↓ (per region)
       Ollama / OpenAI-compatible vision endpoint
                         ↓
              SymbolDetection records


Determinism contract:

* Temperature 0.0 + a fixed seed in the prompt.
* Results cached by ``(page_image_sha256, legend_crop_sha256s, prompt_version)``.
* Cache lives at ``<artifacts>/.orbitbrief_vision_detect_cache.jsonl`` so re-runs
  are free (matches the polish-cache pattern).

Fallback: when the vision endpoint is unreachable, returns ``[]``
and the caller falls back to the existing text-tag detector.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ── Constants ────────────────────────────────────────────────────

DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Tight DPI on region crops — vision LLM doesn't need fine detail
# but needs enough pixels to recognize stroke patterns. 200 DPI is
# a good balance (~50 KB / region at 40×40 pt).
REGION_CROP_DPI = 200

# Confidence floor — emit detections with prob >= this; emit
# `low_confidence_candidate` warnings for 0.40-0.69 so PMs can
# spot-check.
HIGH_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.40

# Cache key version — bump when prompt or input schema changes
PROMPT_VERSION = "v1"


# ── Output schema ─────────────────────────────────────────────────


@dataclass(frozen=True)
class VisionDetection:
    """One vision-LLM-confirmed symbol detection."""

    page_index: int
    bbox_pdf: tuple[float, float, float, float]
    matched_entry_id: str                              # which legend entry it matched
    matched_symbol_text: str                           # "CR" / "PTZ" / ...
    matched_label_text: str                            # "CARD READER" / "PTZ CAMERA" / ...
    confidence: float                                  # 0.0 - 1.0
    rationale: str
    model_used: str
    elapsed_ms: int


# ── Prompt ────────────────────────────────────────────────────────


def _build_prompt(
    *,
    legend_crops: Sequence[Any],
    region_index: int,
    region_bbox: tuple[float, float, float, float],
    total_regions: int,
) -> str:
    """Construct the user prompt for one region. The legend crops are
    attached separately as the image-context messages."""
    legend_descriptions: list[str] = []
    for i, crop in enumerate(legend_crops):
        sym = crop.symbol_text or "?"
        lbl = crop.label_text or "?"
        legend_descriptions.append(f"  {i+1}. symbol={sym!r} label={lbl!r}")

    return (
        f"Region {region_index + 1} of {total_regions} on an architectural drawing.\n"
        f"Region bbox (PDF points): {tuple(round(v, 1) for v in region_bbox)}\n"
        "\n"
        "Above the region image, the legend symbol crops are attached.\n"
        f"There are {len(legend_crops)} candidates:\n"
        f"{chr(10).join(legend_descriptions)}\n"
        "\n"
        "Task: Match the region image to ONE of the legend candidates or 'no_match'.\n"
        "Strict rules:\n"
        "  1. Only match if the region clearly depicts ONE icon from the legend.\n"
        "  2. Do not match background grid lines, dimensions, or text labels.\n"
        "  3. If the region contains multiple icons, return the most prominent.\n"
        "  4. If unsure, return 'no_match' — false positives are worse than false negatives.\n"
        "\n"
        "Respond with JSON only (no prose, no markdown):\n"
        "  {\n"
        "    \"match\": \"no_match\" | <legend_index 1-N>,\n"
        "    \"confidence\": 0.0-1.0,\n"
        "    \"rationale\": \"<one sentence>\"\n"
        "  }\n"
    )


# ── Image helpers ────────────────────────────────────────────────


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _png_bytes_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def _crop_region_png(
    *,
    page: Any,
    bbox_pdf: tuple[float, float, float, float],
    dpi: int = REGION_CROP_DPI,
) -> bytes | None:
    """Render the region as a PNG. Returns the PNG bytes."""
    try:
        import fitz                                   # type: ignore[import-not-found]
    except Exception:                                 # pragma: no cover
        return None
    try:
        clip = fitz.Rect(*bbox_pdf)
        scale = dpi / 72.0
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=clip,
            alpha=False,
        )
        return pix.tobytes("png")
    except Exception:                                 # pragma: no cover
        return None


# ── Cache ────────────────────────────────────────────────────────


@dataclass
class _VisionCache:
    """File-backed cache so re-runs are free.

    Key: sha256(region_png || legend_crop_sha256s || prompt_version || model)
    Value: full VisionDetection or "no_match" sentinel
    """

    path: Path
    _mem: dict[str, dict[str, Any]] = field(default_factory=dict)
    _loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._mem[row["key"]] = row["value"]
                except Exception:                     # pragma: no cover
                    continue
        except OSError:                               # pragma: no cover
            pass

    def get(self, key: str) -> dict[str, Any] | None:
        self.load()
        return self._mem.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.load()
        self._mem[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        except OSError:                               # pragma: no cover
            pass


def _make_cache_key(
    *,
    region_png_sha: str,
    legend_shas: Sequence[str],
    model: str,
) -> str:
    h = hashlib.sha256()
    h.update(region_png_sha.encode("ascii"))
    h.update(b"\x00")
    for s in legend_shas:
        h.update(s.encode("ascii"))
        h.update(b"\x00")
    h.update(PROMPT_VERSION.encode("ascii"))
    h.update(b"\x00")
    h.update(model.encode("ascii"))
    return h.hexdigest()[:32]


# ── Vision HTTP call (Ollama-compatible /api/chat with images) ────


def _call_vision_endpoint(
    *,
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Sequence[str],
    timeout_s: float = 60.0,
) -> dict[str, Any] | None:
    """Single HTTP call to Ollama-compatible vision endpoint.

    Returns the parsed model output as a dict, or None on any
    transport / decode error.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 7},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": list(images_b64),
            }
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    text = (
        (payload.get("message") or {}).get("content")
        or payload.get("response")
        or ""
    )
    if not text:
        return None
    # Strip qwen <think> blocks if any
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ── Top-level driver ─────────────────────────────────────────────


def detect_symbols_via_vision(
    *,
    page: Any,
    page_index: int,
    region_proposals: Sequence[Any],
    legend_crops: Sequence[Any],
    cache_path: Path | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_regions: int = 100,
) -> list[VisionDetection]:
    """Run the vision-LLM detector across every region proposal on a page.

    Tolerant: returns ``[]`` (with no exception) when:

    * the endpoint is unreachable
    * legend_crops is empty (nothing to match against)
    * region_proposals is empty

    Cached: identical (region_png, legend_crops, model) returns
    cached result.
    """
    if not legend_crops or not region_proposals:
        return []

    base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    model = model or os.environ.get("ORBITBRIEF_VISION_MODEL") or DEFAULT_VISION_MODEL

    cache: _VisionCache | None = None
    if cache_path is not None:
        cache = _VisionCache(path=Path(cache_path))

    # Pre-encode legend crops once (they're the same for every region call on this page)
    legend_b64s: list[str] = []
    legend_shas: list[str] = []
    for crop in legend_crops:
        png_path = getattr(crop, "png_absolute_path", None)
        if png_path is None or not Path(png_path).exists():
            continue
        try:
            legend_b64s.append(_image_to_base64(Path(png_path)))
            legend_shas.append(getattr(crop, "png_bytes_sha256", "") or "")
        except OSError:                               # pragma: no cover
            continue

    if not legend_b64s:
        return []

    out: list[VisionDetection] = []
    proposals = list(region_proposals)[:max_regions]

    for region_index, proposal in enumerate(proposals):
        bbox = getattr(proposal, "bbox_pdf", None)
        if bbox is None:
            continue
        region_png = _crop_region_png(page=page, bbox_pdf=bbox)
        if region_png is None:
            continue
        region_sha = hashlib.sha256(region_png).hexdigest()

        cache_key = _make_cache_key(
            region_png_sha=region_sha,
            legend_shas=legend_shas,
            model=model,
        )

        cached_value: dict[str, Any] | None = None
        if cache is not None:
            cached_value = cache.get(cache_key)

        if cached_value is not None:
            response = cached_value
        else:
            prompt = _build_prompt(
                legend_crops=legend_crops,
                region_index=region_index,
                region_bbox=bbox,
                total_regions=len(proposals),
            )
            t0 = time.monotonic()
            response_dict = _call_vision_endpoint(
                base_url=base_url,
                model=model,
                prompt=prompt,
                images_b64=[*legend_b64s, _png_bytes_to_base64(region_png)],
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            if response_dict is None:
                response = None  # type: ignore[assignment]
            else:
                response = {
                    "match": response_dict.get("match", "no_match"),
                    "confidence": float(response_dict.get("confidence") or 0.0),
                    "rationale": str(response_dict.get("rationale") or "")[:300],
                    "elapsed_ms": elapsed,
                }
                if cache is not None:
                    cache.put(cache_key, response)

        if response is None:
            continue

        match = response.get("match")
        conf = float(response.get("confidence") or 0.0)
        if not match or match == "no_match" or conf < LOW_CONFIDENCE:
            continue

        # ``match`` may come back as int or stringified int (legend_index 1-N)
        try:
            legend_idx = int(match) - 1
        except (TypeError, ValueError):
            continue
        if legend_idx < 0 or legend_idx >= len(legend_crops):
            continue

        crop = legend_crops[legend_idx]
        out.append(
            VisionDetection(
                page_index=page_index,
                bbox_pdf=tuple(bbox),
                matched_entry_id=getattr(crop, "entry_id", "") or "",
                matched_symbol_text=getattr(crop, "symbol_text", "") or "",
                matched_label_text=getattr(crop, "label_text", "") or "",
                confidence=conf,
                rationale=str(response.get("rationale") or ""),
                model_used=model,
                elapsed_ms=int(response.get("elapsed_ms") or 0),
            )
        )

    return out


def is_vision_endpoint_reachable(
    base_url: str | None = None,
    *,
    model: str | None = None,
    timeout_s: float = 3.0,
) -> bool:
    """Quick health check: can we reach the Ollama tags endpoint and
    does the configured vision model show up in the list?"""
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    model = model or os.environ.get("ORBITBRIEF_VISION_MODEL") or DEFAULT_VISION_MODEL
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    models = [(m.get("name") or "") for m in payload.get("models", [])]
    return any(model in m for m in models)


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_VISION_MODEL",
    "HIGH_CONFIDENCE",
    "LOW_CONFIDENCE",
    "PROMPT_VERSION",
    "REGION_CROP_DPI",
    "VisionDetection",
    "detect_symbols_via_vision",
    "is_vision_endpoint_reachable",
]
