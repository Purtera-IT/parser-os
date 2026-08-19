"""Multi-backend OCR chain.

OCR is the universality gap for two artifact types:
  * Image artifacts (HEIC / PNG / JPG site survey photos, network diagrams).
  * Scanned PDF pages with no text layer (the parser detects these as
    "low-text" pages — <80 chars — and skips them today).

The chain tries each available backend in order until one returns text,
otherwise returns "" so the caller emits the existing marker atom:

  1. PyMuPDF built-in OCR (``page.get_textpage_ocr``) — requires
     a Tesseract binary on PATH. Best path on a configured server.
  2. pytesseract — same binary requirement; works on PIL images.
  3. easyocr — pure-Python (PyTorch). Heavy install, no system deps.
  4. Ollama vision model — calls a vision-capable Ollama model
     (llava / bakllava / qwen2.5vl) via the configured base URL.
     Pure HTTP; no local install needed beyond ``ollama pull <model>``.

Configuration (all env vars):

  PARSER_OS_OCR_OLLAMA_BASE_URL   URL of the Ollama server (default
                                  matches the Phase 1.5 setting).
  PARSER_OS_OCR_OLLAMA_VISION_MODEL  Model name (default ``llava``).
  PARSER_OS_OCR_DISABLE           Set to ``1`` to skip every backend
                                  and force the marker-only path.
  PARSER_OS_OCR_LANGUAGE          Tesseract language code (default ``eng``).

Returns dict shape:

  {
    "text": "...",
    "backend": "pymupdf_tesseract" | "pytesseract" | "easyocr" |
              "ollama_vision" | "" (when nothing fired),
    "confidence": float (best-effort; 0.0 when unknown),
    "notes": ["..."],   # diagnostics for the PM_HANDOFF degraded callout
  }
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any


# Disable ALL backends with one env var — useful for deterministic CI
# runs and for explicit "marker-only" mode.
def _ocr_disabled() -> bool:
    return os.environ.get("PARSER_OS_OCR_DISABLE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ocr_language() -> str:
    return os.environ.get("PARSER_OS_OCR_LANGUAGE", "eng").strip() or "eng"


def _ollama_base_url() -> str:
    # Default matches the established Mac Studio Tailscale URL the
    # Phase 1.5 / 1.75 envelope_backfill helpers use.
    return os.environ.get(
        "PARSER_OS_OCR_OLLAMA_BASE_URL",
        os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    ).rstrip("/")


def _ollama_vision_model() -> str:
    return os.environ.get("PARSER_OS_OCR_OLLAMA_VISION_MODEL", "llava").strip() or "llava"


def _empty(notes: list[str]) -> dict[str, Any]:
    return {"text": "", "backend": "", "confidence": 0.0, "notes": notes}


def ocr_pdf_page(page) -> dict[str, Any]:
    """OCR a PyMuPDF ``Page`` object.

    Tries PyMuPDF's built-in OCR first (uses Tesseract under the hood
    when the binary is on PATH). Falls back through pytesseract /
    easyocr / Ollama if the first path fails.
    """
    notes: list[str] = []
    if _ocr_disabled():
        return _empty(["OCR disabled via PARSER_OS_OCR_DISABLE"])

    # 1) PyMuPDF built-in OCR
    try:
        tp = page.get_textpage_ocr(language=_ocr_language(), full=True)
        text = page.get_text("text", textpage=tp) or ""
        if text.strip():
            return {
                "text": text,
                "backend": "pymupdf_tesseract",
                "confidence": 0.85,
                "notes": notes,
            }
        notes.append("pymupdf_tesseract returned no text")
    except Exception as exc:
        notes.append(f"pymupdf_tesseract unavailable: {type(exc).__name__}")

    # 2) Render page → PIL image and try pytesseract / easyocr / Ollama
    try:
        pix = page.get_pixmap(dpi=200, alpha=False)
        image_bytes = pix.tobytes("png")
    except Exception as exc:
        notes.append(f"page.get_pixmap failed: {exc}")
        return _empty(notes)

    return _ocr_image_bytes(image_bytes, notes)


def ocr_image_file(path: Path) -> dict[str, Any]:
    """OCR a PNG / JPG / HEIC etc. file."""
    notes: list[str] = []
    if _ocr_disabled():
        return _empty(["OCR disabled via PARSER_OS_OCR_DISABLE"])
    try:
        image_bytes = path.read_bytes()
    except Exception as exc:
        return _empty([f"read failed: {exc}"])
    return _ocr_image_bytes(image_bytes, notes)


def ocr_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    """OCR raw image bytes (inline email MIME parts, buffers, etc.)."""
    notes: list[str] = []
    if _ocr_disabled():
        return _empty(["OCR disabled via PARSER_OS_OCR_DISABLE"])
    if not image_bytes:
        return _empty(["empty image bytes"])
    return _ocr_image_bytes(image_bytes, notes)


def _ocr_image_bytes(image_bytes: bytes, notes: list[str]) -> dict[str, Any]:
    """Common chain for raw image bytes."""
    # 0) Azure Document Intelligence — best for HubSpot order screenshots.
    try:
        from app.core.doc_intel_ocr import doc_intel_available, extract_text_from_image_bytes

        if doc_intel_available():
            text = extract_text_from_image_bytes(image_bytes) or ""
            if text.strip():
                return {
                    "text": text.strip(),
                    "backend": "azure_doc_intel",
                    "confidence": 0.92,
                    "notes": notes,
                }
            notes.append("azure_doc_intel returned no text")
    except Exception as exc:
        notes.append(f"azure_doc_intel unavailable: {type(exc).__name__}")

    # 1) pytesseract
    try:
        import pytesseract
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) < 1400:
            scale = 2
            img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        config = f"--psm 6 --oem 3 -l {_ocr_language()}"
        text = pytesseract.image_to_string(gray, config=config) or ""
        if text.strip():
            return {
                "text": text.strip(),
                "backend": "pytesseract",
                "confidence": 0.80,
                "notes": notes,
            }
        notes.append("pytesseract returned no text")
    except Exception as exc:
        notes.append(f"pytesseract unavailable: {type(exc).__name__}")

    # 2) easyocr
    try:
        import easyocr  # type: ignore[import-not-found]
        reader = easyocr.Reader([_ocr_language()[:2]], gpu=False)  # type: ignore[arg-type]
        results = reader.readtext(image_bytes, detail=0, paragraph=True)
        text = "\n".join(results) if results else ""
        if text.strip():
            return {
                "text": text,
                "backend": "easyocr",
                "confidence": 0.78,
                "notes": notes,
            }
        notes.append("easyocr returned no text")
    except Exception as exc:
        notes.append(f"easyocr unavailable: {type(exc).__name__}")

    # 3) Ollama vision (llava / qwen2.5vl / bakllava / ...)
    try:
        import urllib.request
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": _ollama_vision_model(),
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are an OCR assistant. Transcribe ALL legible "
                        "text from this image, preserving structure (lines, "
                        "lists, table-like layouts). If the image is a "
                        "site survey photo with annotations, transcribe "
                        "every annotation. Output ONLY the transcribed "
                        "text — no commentary, no JSON."
                    ),
                    "images": [b64],
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            f"{_ollama_base_url()}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Short timeout — we don't want a hung Ollama to block every
        # OCR call in a batch. 30 s is generous for a single image.
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        text = ((body.get("message") or {}).get("content") or "").strip()
        if text:
            return {
                "text": text,
                "backend": f"ollama_vision/{_ollama_vision_model()}",
                "confidence": 0.65,  # vision-LLM OCR is lower-confidence than dedicated OCR
                "notes": notes,
            }
        notes.append(f"ollama_vision/{_ollama_vision_model()} returned no text")
    except Exception as exc:
        notes.append(f"ollama_vision unavailable: {type(exc).__name__}")

    return _empty(notes)


def available_backends() -> list[str]:
    """Smoke-check which backends are installed/reachable.

    Returns the names of backends that *would* fire if asked. Used by
    PM_HANDOFF to surface "OCR was attempted with backends [...]" so the
    PM knows whether to install one before re-running.
    """
    out: list[str] = []
    if _ocr_disabled():
        return out
    # pytesseract
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        # Try to detect Tesseract binary
        try:
            pytesseract.get_tesseract_version()
            out.append("pytesseract")
        except Exception:
            pass
    except Exception:
        pass
    # easyocr
    try:
        import easyocr  # noqa: F401
        out.append("easyocr")
    except Exception:
        pass
    # Ollama vision — check by reachability
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{_ollama_base_url()}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        names = {m.get("name", "") for m in (body.get("models") or [])}
        vm = _ollama_vision_model()
        if any(vm in n or n.startswith(vm.split(":", 1)[0]) for n in names):
            out.append(f"ollama_vision/{vm}")
    except Exception:
        pass
    return out
