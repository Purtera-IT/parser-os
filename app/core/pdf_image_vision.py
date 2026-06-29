"""PDF embedded-image understanding — a path SEPARATE from schematics.

Scope (hard boundary):
  * IN  — raster images embedded *inside* a PDF (photos, equipment shots,
          install-instruction graphics, screenshots, charts, table-as-image,
          maps, signatures). These arrive as ``image_marker`` atoms emitted by
          ``orbitbrief_pdf._emit_image_markers`` (saved crop + caption + page).
  * OUT — schematic / drawing PAGES (symbol legends, CAD sheets). Those go
          through ``orbitbrief_page_os.segmentation.schematic.*``. This module
          NEVER imports that package; the two paths share no code by design.

Two sub-paths, chosen by a cheap classify gate:
  1. DESCRIBE  (photo / diagram / chart / map) — a context-grounded natural
     description + typed facts. Quality comes from the CONTEXT ENVELOPE
     (page text + neighbour pages + caption + position), not model size.
  2. TRANSCRIBE (instructions / screenshot / label / table-image) — OCR the
     crop, then fuse OCR text + image in the VLM so commands / part numbers /
     values are transcribed VERBATIM (the VLM structures, OCR anchors the exact
     characters). A verbatim guard drops any line whose tokens aren't in OCR.

Design invariants (match the rest of the system):
  * Guess-free + abstain-first. No endpoint / no crop / guard-fail / any error
    -> emit nothing. Byte-identical to today when the flag is off.
  * Frozen teacher. The VLM is never fine-tuned here; gains come from context,
    OCR fusion and prompting. PM corrections on the emitted atoms feed the
    existing TrainingLog loop; only a cheap CPU *gate* is ever distilled later.
  * OFF by default (``SOWSMITH_PDF_IMAGE_VISION``). Additive: returns NEW atoms
    that upgrade the ``needs_extractor`` markers; never mutates existing atoms.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

logger = logging.getLogger(__name__)

# Image kinds routed to structured table-row extraction (BOM prompt).
_TABLE_KINDS = {"table_image"}
# Image kinds that carry verbatim text we must transcribe exactly (OCR fusion).
_TRANSCRIBE_KINDS = {"instructions", "screenshot", "label"}
# Image kinds that are noise — never worth a describe/transcribe call.
_SKIP_KINDS = {"logo", "decorative", "signature", "empty"}
# Image kinds that get the context-grounded describe path.
_DESCRIBE_KINDS = {"photo", "diagram", "chart", "map"}


# ── config ──────────────────────────────────────────────────────────


def enabled() -> bool:
    return os.environ.get("SOWSMITH_PDF_IMAGE_VISION", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _gate_model() -> str | None:
    """Cheap triage model (default: the configured vision model). A small VLM
    like qwen2.5vl:7b is plenty to decide meaningful-vs-noise + kind."""
    return os.environ.get("SOWSMITH_PDF_IMAGE_GATE_MODEL") or None


def _describe_model() -> str | None:
    """Higher-fidelity describe/transcribe model (e.g. qwen2.5vl:32b). Falls
    back to the configured vision model when unset."""
    return os.environ.get("SOWSMITH_PDF_IMAGE_DESCRIBE_MODEL") or None


@contextmanager
def _vision_model(name: str | None):
    """Temporarily point ``call_vision_llm`` at a specific Ollama vision model.
    No-op when ``name`` is None or a hosted teacher is configured (the teacher
    path ignores OLLAMA_VISION_MODEL)."""
    if not name:
        yield
        return
    prev = os.environ.get("OLLAMA_VISION_MODEL")
    os.environ["OLLAMA_VISION_MODEL"] = name
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("OLLAMA_VISION_MODEL", None)
        else:
            os.environ["OLLAMA_VISION_MODEL"] = prev


# ── small helpers ───────────────────────────────────────────────────


def _parse_json_obj(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from an LLM reply."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _tokens(s: str, min_len: int = 4) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if len(t) >= min_len}


def _iter_image_markers(atoms: list[Any]):
    """Yield (atom, pdf_name, page_index, region_ref, saved_path, caption) for
    every PDF image_marker atom that has a saved crop on disk."""
    for a in atoms:
        try:
            val = getattr(a, "value", None) or {}
            if not isinstance(val, dict) or val.get("kind") != "image_marker":
                continue
            region_ref = str(val.get("region_ref") or "")
            if not region_ref.startswith("page"):
                continue  # only PDF page images (page{n}/image{xref})
            saved = val.get("saved_path")
            if not saved:
                continue
            refs = getattr(a, "source_refs", None) or []
            pdf_name = (getattr(refs[0], "filename", "") if refs else "") or ""
            if not pdf_name.lower().endswith(".pdf"):
                continue
            m = re.match(r"page(\d+)/", region_ref)
            page_index = int(m.group(1)) if m else 0
            caption = val.get("expected_content") or ""
            yield a, pdf_name, page_index, region_ref, str(saved), str(caption)
        except Exception:
            continue


def _vision_reachable() -> bool:
    """True when the PDF-image path can actually call a VLM. When forcing Ollama
    (default), check the Ollama host — NOT the text teacher (DeepSeek is not
    multimodal and would make vision_endpoint_reachable() lie)."""
    if _use_ollama_for_pdf_images():
        host = os.environ.get("OLLAMA_HOST", "").rstrip("/")
        if not host:
            return False
        try:
            import requests
            r = requests.get(f"{host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
    try:
        from app.core.vision_extraction import vision_endpoint_reachable
        return vision_endpoint_reachable()
    except Exception:
        return False


def _ollama_vision_direct(
    image_bytes: bytes, prompt: str, *, model: str | None, max_tokens: int,
) -> str:
    """Call the local Ollama vision host directly — bypasses the text teacher
    (DeepSeek etc.) which is not multimodal and silently returns empty."""
    import base64
    import requests
    from app.core.vision_extraction import _DEFAULT_HOST, _DEFAULT_VISION_MODEL, _encode_image_b64
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    mdl = model or os.environ.get("OLLAMA_VISION_MODEL", _DEFAULT_VISION_MODEL)
    payload = {
        "model": mdl,
        "prompt": prompt,
        "images": [_encode_image_b64(image_bytes)],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        if r.status_code != 200:
            return ""
        return r.json().get("response", "") or ""
    except Exception as e:
        logger.warning("pdf_image_vision ollama call failed: %s", e)
        return ""


def _use_ollama_for_pdf_images() -> bool:
    """PDF embedded images always need a real VLM. Default ON so a text-only
    teacher (DeepSeek) never silently kills the path."""
    return os.environ.get("SOWSMITH_PDF_IMAGE_FORCE_OLLAMA", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _resolve_crop_path(saved_path: str) -> Path | None:
    """Resolve a parser-relative crop path to an on-disk file."""
    raw = Path(saved_path.replace("\\", "/"))
    candidates = [raw, Path.cwd() / raw]
    img_root = Path(os.environ.get("SOWSMITH_IMAGE_DIR", "_extracted_images"))
    if raw.as_posix().startswith("_extracted_images/"):
        candidates.append(Path(raw.as_posix()))
        candidates.append(Path.cwd() / raw.as_posix())
    else:
        candidates.append(img_root / raw.name)
        candidates.append(Path.cwd() / img_root / raw.name)
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            continue
    return None


def _load_crop(saved_path: str) -> bytes:
    p = _resolve_crop_path(saved_path)
    if p is None:
        return b""
    try:
        return p.read_bytes()
    except Exception:
        return b""


def _page_context(pdf_name: str, page_index: int, neighbor_chars: int):
    """Return (this_page_text, prev_tail, next_head, page_count) for grounding.
    Resolves basenames via the vision artifact-path registry. Abstains to empty
    on any failure."""
    try:
        import fitz  # type: ignore[import-not-found]
        from app.core.vision_extraction import _resolve_pdf_path
    except Exception:
        return "", "", "", 0
    path = _resolve_pdf_path(pdf_name)
    try:
        doc = fitz.open(path)
    except Exception:
        return "", "", "", 0
    try:
        n = doc.page_count
        this = prev = nxt = ""
        if 0 <= page_index < n:
            this = doc.load_page(page_index).get_text() or ""
        if page_index - 1 >= 0:
            prev = (doc.load_page(page_index - 1).get_text() or "")[-neighbor_chars:]
        if page_index + 1 < n:
            nxt = (doc.load_page(page_index + 1).get_text() or "")[:neighbor_chars]
        return this, prev, nxt, n
    except Exception:
        return "", "", "", 0
    finally:
        try:
            doc.close()
        except Exception:
            pass


_OCR_VLM_PROMPT = """You are an OCR engine. Transcribe ALL legible text in this
image EXACTLY as written — every command, IP address, part number, quantity and
value, preserving line order and grouping. Do not summarise, explain, or add
anything. Output ONLY the raw transcribed text.
/no_think
"""


def _ocr_crop(saved_path: str, crop: bytes | None = None, *, allow_vlm: bool = False) -> str:
    """Neutral OCR of the saved crop (NOT the schematic OCR).

    Tries the dedicated OCR chain first (tesseract / easyocr / a separately-
    configured Ollama OCR endpoint). When that yields nothing AND ``allow_vlm``
    is set, it falls back to the SAME ``call_vision_llm`` path the describe/table
    calls use (teacher API or the configured OLLAMA_HOST). This keeps the
    transcribe path's verbatim anchor alive on the worker — where no Tesseract
    binary is installed and the OCR chain's standalone Ollama URL is unset —
    with no extra env wiring.

    ``allow_vlm`` defaults False so the cheap classify gate never pays for an
    extra VLM OCR call; only the transcribe/table paths (which need verbatim
    grounding) opt in. Empty on total failure (caller then abstains)."""
    try:
        from app.parsers._ocr_chain import ocr_image_file
        res = ocr_image_file(Path(saved_path))
        text = (res.get("text") or "").strip() if isinstance(res, dict) else ""
        if text:
            return text
    except Exception:
        pass
    if not allow_vlm:
        return ""
    # VLM OCR fallback over the proven vision path (verbatim anchor).
    try:
        data = crop if crop is not None else _load_crop(saved_path)
        if not data:
            return ""
        out = _vlm(data, _OCR_VLM_PROMPT, model=_gate_model(), max_tokens=2000)
        return (out or "").strip()
    except Exception:
        return ""


def _position_label(page_index: int, page_count: int) -> str:
    if page_count <= 0:
        return f"page {page_index + 1}"
    return f"page {page_index + 1} of {page_count}"


# ── VLM calls (gate / describe / transcribe) ────────────────────────


def _vlm(image_bytes: bytes, prompt: str, *, model: str | None, max_tokens: int) -> str:
    if _use_ollama_for_pdf_images():
        with _vision_model(model):
            return _ollama_vision_direct(
                image_bytes, prompt, model=model, max_tokens=max_tokens,
            ) or ""
    from app.core.vision_extraction import call_vision_llm
    with _vision_model(model):
        return call_vision_llm(image_bytes, prompt, max_tokens=max_tokens) or ""


_GATE_PROMPT = """You are triaging an image embedded in a technical / managed-
services document. Classify it. Be strict: letterheads, logos, decorative
borders and signatures are NOT meaningful content.

Return JSON only:
{{"image_kind": "photo|diagram|chart|table_image|screenshot|instructions|label|map|logo|signature|decorative|empty",
  "has_text": true|false,
  "meaningful": true|false}}

Caption hint (may be empty): "{caption}"
/no_think
"""


_DESCRIBE_PROMPT = """You are describing an image embedded in a technical
document. Use the surrounding context to ground your description in THIS
document's reality. Describe ONLY what is visible; do not invent details.

{envelope}

Return JSON only:
{{"description": "what the image shows, grounded in the context above",
  "facts": [{{"kind": "equipment|site_condition|reading|component|label|connection|other",
              "text": "one concrete fact visible in the image"}}]}}
/no_think
"""


_TRANSCRIBE_PROMPT = """You are transcribing a text-bearing image (install
instructions / screenshot / labelled diagram) from a technical document.

CRITICAL: transcribe commands, IP addresses, part numbers, settings and values
EXACTLY as written. Use the OCR text below as the source of truth for exact
characters; use the image only to fix obvious OCR errors and to recover the
correct ORDER and grouping. Never paraphrase a command or a value.

{envelope}

OCR TEXT (verbatim source):
\"\"\"
{ocr}
\"\"\"

Return JSON only:
{{"summary": "one line: what this image instructs or shows",
  "steps": [{{"n": 1, "action": "imperative step text",
              "command": "exact command or value if any, else empty"}}]}}
/no_think
"""


def _build_envelope(
    *, pdf_name: str, position: str, caption: str,
    this_text: str, prev_tail: str, next_head: str, max_page_chars: int,
) -> str:
    parts = [f'DOCUMENT: "{pdf_name}"', f"LOCATION: {position}"]
    if caption:
        parts.append(f'CAPTION NEAR IMAGE: "{caption}"')
    if this_text.strip():
        parts.append("PAGE TEXT (this page):\n" + this_text.strip()[:max_page_chars])
    if prev_tail.strip():
        parts.append("PREVIOUS PAGE (tail):\n" + prev_tail.strip())
    if next_head.strip():
        parts.append("NEXT PAGE (head):\n" + next_head.strip())
    return "\n\n".join(parts)


# ── guards ──────────────────────────────────────────────────────────


def _context_guard(description: str, grounding: str, min_overlap: float) -> bool:
    """True if the description is sufficiently grounded in document text. A
    fully-invented description (no token overlap) is rejected. Disabled when
    min_overlap <= 0."""
    if min_overlap <= 0:
        return True
    desc_tok = _tokens(description)
    if not desc_tok:
        return False
    ground_tok = _tokens(grounding)
    if not ground_tok:
        # No text to verify against (e.g. image-only page) — allow, but the
        # caller keeps confidence low so a human still reviews.
        return True
    overlap = len(desc_tok & ground_tok) / max(1, len(desc_tok))
    return overlap >= min_overlap


def _verbatim_ok(text: str, ocr_norm: str) -> bool:
    """True if >=50% of a line's alnum tokens appear in the OCR text. When OCR
    is empty we cannot verify, so we reject (transcribe path requires OCR)."""
    if not ocr_norm:
        return False
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) >= 2]
    if not toks:
        return False
    hits = sum(1 for t in toks if t in ocr_norm)
    return hits / len(toks) >= 0.5


def _log_gate_silver(caption: str, ocr: str, image_kind: str, *, via: str) -> None:
    """Silver training row for the CPU gate distillation loop. Never raises."""
    try:
        from app.core.pdf_image_gate import gate_feature_text
        from app.core.training_log import TEACHER_LLM, TrainingRow, log_rows
        label = "skip" if image_kind in _SKIP_KINDS else image_kind
        feat = gate_feature_text(caption, ocr)
        log_rows([TrainingRow(
            relation="pdf_image_kind",
            label=label,
            raw_text=feat,
            masked_text=feat,
            label_kind="judgment",
            teacher=TEACHER_LLM,
            confidence=0.7,
            provenance={"stage": "pdf_image_vision_gate", "via": via},
        )])
    except Exception:
        pass


def _classify_image(
    *, crop: bytes, caption: str, saved_path: str,
) -> tuple[bool, str, str]:
    """Classify one crop. Returns (meaningful, image_kind, via_tag)."""
    ocr = _ocr_crop(saved_path, crop)  # cheap chain only (no VLM cost on the gate)
    try:
        from app.core import pdf_image_gate
        cpu = pdf_image_gate.classify(caption, ocr)
        if cpu is not None:
            meaningful, kind = cpu
            kind = kind if meaningful else "skip"
            _log_gate_silver(caption, ocr, kind, via="cpu_gate")
            return meaningful, kind, "cpu_gate"
    except Exception:
        pass
    gate_raw = _vlm(
        crop, _GATE_PROMPT.format(caption=caption[:160]),
        model=_gate_model(), max_tokens=120,
    )
    gate = _parse_json_obj(gate_raw)
    image_kind = str(gate.get("image_kind") or "").strip().lower()
    meaningful = bool(gate.get("meaningful"))
    if image_kind in _SKIP_KINDS:
        meaningful = False
    _log_gate_silver(caption, ocr, image_kind or "skip", via="vlm_gate")
    return meaningful, image_kind, "vlm_gate"


def _caption_overlap(caption: str, description: str) -> float:
    ct = _tokens(caption)
    if not ct:
        return 1.0
    dt = _tokens(description)
    if not dt:
        return 0.0
    return len(ct & dt) / len(ct)


def _apply_caption_mismatch(atoms: list[EvidenceAtom], caption: str, *, min_overlap: float):
    if not caption.strip() or not atoms:
        return
    head = next((a for a in atoms if a.value.get("fact_kind") == "image_description"), None)
    if head is None:
        return
    if _caption_overlap(caption, head.raw_text) >= min_overlap:
        return
    for a in atoms:
        a.review_flags = sorted(set(a.review_flags + ["image_answer_mismatch"]))


def _table_image_atoms(
    *, marker: Any, pdf_name: str, page_index: int, region_ref: str,
    crop: bytes, envelope: str, image_kind: str, ocr_text: str,
) -> list[EvidenceAtom]:
    """Extract BOM-style rows from a table screenshot using the page-level BOM prompt."""
    try:
        from app.core import vision_extraction as ve
    except Exception:
        return []
    prompt = f"Context:\n{envelope}\n\n{ve._BOM_PROMPT}"
    raw = _vlm(crop, prompt, model=_describe_model(), max_tokens=1500)
    parsed = ve._parse_vision_response(raw)
    rows = ve._normalize_to_rows(parsed, "BOM")
    atoms: list[EvidenceAtom] = []
    ocr_norm = " ".join(re.findall(r"[A-Za-z0-9]+", ocr_text.lower()))
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if ocr_norm and not _verbatim_ok(text, ocr_norm):
            continue
        rk = str(row.get("kind") or "table_row")
        a = _emit_atom(
            marker=marker, pdf_name=pdf_name, region_ref=region_ref,
            page_index=page_index, text=text, image_kind=image_kind,
            fact_kind=f"table_row:{rk}", confidence=0.65,
            atom_type=(
                AtomType.vendor_line_item
                if rk in ("money", "part_number") else AtomType.scope_item
            ),
        )
        if a:
            atoms.append(a)
    return atoms


# ── atom emission ───────────────────────────────────────────────────


def _emit_atom(
    *, marker: Any, pdf_name: str, region_ref: str, page_index: int,
    text: str, image_kind: str, fact_kind: str, confidence: float,
    atom_type: AtomType = AtomType.scope_item,
) -> EvidenceAtom | None:
    text = (text or "").strip()
    if not text:
        return None
    project_id = getattr(marker, "project_id", "") or ""
    artifact_id = getattr(marker, "artifact_id", "") or ""
    parser_version = getattr(marker, "parser_version", "") or "pdf_image_vision_v1"
    atom_id = stable_id("atm", artifact_id, "pdf_image_vision", region_ref, fact_kind, text[:80])
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=pdf_name,
        locator={
            "region_ref": region_ref,
            "page": page_index,
            "extraction": "pdf_image_vision_v1",
        },
        extraction_method="pdf_image_vision_v1",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=normalize_text(text),
        value={
            "via": "pdf_image_vision",
            "image_kind": image_kind,
            "fact_kind": fact_kind,
            "region_ref": region_ref,
            "source_marker_id": getattr(marker, "id", ""),
        },
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.meeting_note,
        confidence=confidence,
        confidence_raw=confidence,
        calibrated_confidence=confidence,
        review_status=ReviewStatus.needs_review,
        review_flags=["pdf_image_vision", f"image_kind:{image_kind}"],
        parser_version=parser_version,
    )


# ── main entry ──────────────────────────────────────────────────────


def process_image_markers(atoms: list[Any]) -> list[EvidenceAtom]:
    """Describe / transcribe embedded PDF images into NEW atoms.

    Returns [] when disabled, no vision endpoint, or nothing qualifies. Never
    raises and never mutates the input atoms — purely additive."""
    if not enabled() or not atoms:
        return []
    try:
        if not _vision_reachable():
            logger.info("pdf_image_vision: no vision endpoint; abstaining")
            return []
    except Exception:
        return []

    max_images = _int_env("SOWSMITH_PDF_IMAGE_MAX", 40)
    min_bytes = _int_env("SOWSMITH_PDF_IMAGE_MIN_BYTES", 3000)
    neighbor_chars = _int_env("SOWSMITH_PDF_IMAGE_NEIGHBOR_CHARS", 600)
    max_page_chars = _int_env("SOWSMITH_PDF_IMAGE_PAGE_CHARS", 4000)
    guard_min = _float_env("SOWSMITH_PDF_IMAGE_GUARD_MIN", 0.25)
    caption_min = _float_env("SOWSMITH_PDF_IMAGE_CAPTION_MIN", 0.2)

    out: list[EvidenceAtom] = []
    processed = 0
    seen_hashes: set[str] = set()
    for marker, pdf_name, page_index, region_ref, saved_path, caption in _iter_image_markers(atoms):
        if processed >= max_images:
            break
        crop = _load_crop(saved_path)
        if len(crop) < min_bytes:
            continue
        crop_hash = hashlib.sha256(crop).hexdigest()[:16]
        if crop_hash in seen_hashes:
            continue
        seen_hashes.add(crop_hash)
        processed += 1
        try:
            new_atoms = _process_one(
                marker=marker, pdf_name=pdf_name, page_index=page_index,
                region_ref=region_ref, saved_path=saved_path, caption=caption,
                crop=crop, neighbor_chars=neighbor_chars,
                max_page_chars=max_page_chars, guard_min=guard_min,
                caption_min=caption_min,
            )
            out.extend(new_atoms)
        except Exception as exc:  # one bad image never breaks the compile
            logger.warning("pdf_image_vision: %s %s failed: %s", pdf_name, region_ref, exc)
            continue
    if out:
        logger.info(
            "pdf_image_vision: %d atoms from %d images", len(out), processed,
        )
    return out


def _process_one(
    *, marker: Any, pdf_name: str, page_index: int, region_ref: str,
    saved_path: str, caption: str, crop: bytes,
    neighbor_chars: int, max_page_chars: int, guard_min: float,
    caption_min: float,
) -> list[EvidenceAtom]:
    meaningful, image_kind, via = _classify_image(
        crop=crop, caption=caption, saved_path=saved_path,
    )
    if not meaningful or image_kind in _SKIP_KINDS or not image_kind:
        return []

    this_text, prev_tail, next_head, page_count = _page_context(
        pdf_name, page_index, neighbor_chars,
    )
    envelope = _build_envelope(
        pdf_name=pdf_name, position=_position_label(page_index, page_count),
        caption=caption, this_text=this_text, prev_tail=prev_tail,
        next_head=next_head, max_page_chars=max_page_chars,
    )

    if image_kind in _TABLE_KINDS:
        ocr_text = _ocr_crop(saved_path, crop, allow_vlm=True)
        return _table_image_atoms(
            marker=marker, pdf_name=pdf_name, page_index=page_index,
            region_ref=region_ref, crop=crop, envelope=envelope,
            image_kind=image_kind, ocr_text=ocr_text,
        )
    if image_kind in _TRANSCRIBE_KINDS:
        return _transcribe(
            marker=marker, pdf_name=pdf_name, page_index=page_index,
            region_ref=region_ref, saved_path=saved_path, crop=crop,
            envelope=envelope, image_kind=image_kind,
        )
    atoms = _describe(
        marker=marker, pdf_name=pdf_name, page_index=page_index,
        region_ref=region_ref, crop=crop, envelope=envelope,
        image_kind=image_kind, caption=caption,
        grounding=" ".join((this_text, prev_tail, next_head, caption)),
        guard_min=guard_min,
    )
    _apply_caption_mismatch(atoms, caption, min_overlap=caption_min)
    if atoms and via == "cpu_gate":
        for a in atoms:
            a.value["gate_via"] = "cpu_gate"
    return atoms


def _describe(
    *, marker: Any, pdf_name: str, page_index: int, region_ref: str,
    crop: bytes, envelope: str, image_kind: str, caption: str,
    grounding: str, guard_min: float,
) -> list[EvidenceAtom]:
    raw = _vlm(
        crop, _DESCRIBE_PROMPT.format(envelope=envelope),
        model=_describe_model(), max_tokens=900,
    )
    obj = _parse_json_obj(raw)
    description = str(obj.get("description") or "").strip()
    if not description:
        return []
    if not _context_guard(description, grounding, guard_min):
        logger.info("pdf_image_vision: describe abstained (ungrounded) %s", region_ref)
        return []
    # Confidence: lower when there was no page text to verify against.
    conf = 0.6 if grounding.strip() else 0.45
    atoms: list[EvidenceAtom] = []
    head = _emit_atom(
        marker=marker, pdf_name=pdf_name, region_ref=region_ref,
        page_index=page_index, text=description, image_kind=image_kind,
        fact_kind="image_description", confidence=conf,
    )
    if head:
        atoms.append(head)
    facts = obj.get("facts") or []
    if isinstance(facts, list):
        for f in facts:
            if not isinstance(f, dict):
                continue
            ftext = str(f.get("text") or "").strip()
            if not ftext:
                continue
            fk = str(f.get("kind") or "other").strip().lower()
            a = _emit_atom(
                marker=marker, pdf_name=pdf_name, region_ref=region_ref,
                page_index=page_index, text=ftext, image_kind=image_kind,
                fact_kind=f"image_fact:{fk}", confidence=conf,
            )
            if a:
                atoms.append(a)
    return atoms


def _transcribe(
    *, marker: Any, pdf_name: str, page_index: int, region_ref: str,
    saved_path: str, crop: bytes, envelope: str, image_kind: str,
) -> list[EvidenceAtom]:
    ocr_text = _ocr_crop(saved_path, crop, allow_vlm=True)
    ocr_norm = " ".join(re.findall(r"[A-Za-z0-9]+", ocr_text.lower()))
    raw = _vlm(
        crop, _TRANSCRIBE_PROMPT.format(envelope=envelope, ocr=ocr_text[:4000]),
        model=_describe_model(), max_tokens=1200,
    )
    obj = _parse_json_obj(raw)
    atoms: list[EvidenceAtom] = []
    summary = str(obj.get("summary") or "").strip()
    if summary:
        a = _emit_atom(
            marker=marker, pdf_name=pdf_name, region_ref=region_ref,
            page_index=page_index, text=summary, image_kind=image_kind,
            fact_kind="image_instructions_summary", confidence=0.6,
        )
        if a:
            atoms.append(a)
    steps = obj.get("steps") or []
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict):
                continue
            action = str(s.get("action") or "").strip()
            command = str(s.get("command") or "").strip()
            if not action and not command:
                continue
            n = s.get("n")
            line = (f"Step {n}: " if n is not None else "") + action
            if command:
                line = f"{line} — `{command}`".strip()
            # Verbatim guard: the COMMAND must be present in OCR (exact chars,
            # spaces removed); the action must overlap OCR tokens. Either gate
            # failing drops the line (guess-free).
            cmd_ok = (not command) or (
                re.sub(r"\s+", "", command.lower())
                in re.sub(r"\s+", "", ocr_text.lower())
            )
            if not cmd_ok or not _verbatim_ok(action or command, ocr_norm):
                continue
            a = _emit_atom(
                marker=marker, pdf_name=pdf_name, region_ref=region_ref,
                page_index=page_index, text=line, image_kind=image_kind,
                fact_kind="image_instruction_step", confidence=0.7,
            )
            if a:
                atoms.append(a)
    return atoms
