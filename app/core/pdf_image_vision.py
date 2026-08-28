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
    that upgrade the ``needs_extractor`` markers; never removes or rewrites
    existing atoms (the only touch is RECORDING the triage verdict on skipped
    image markers — ``value['gate_verdict']`` — so a skip stays traceable).

Wall clock: the VLM calls dominate a compile (tens of seconds per image, a 90s
per-image timeout, up to 25 images), so ``process_image_markers`` runs them
through a small bounded pool. The stage is split into three phases so that
buys speed and nothing else:

  1. SELECT (serial, no VLM) — ``_build_work_list`` walks the markers in order
     and applies dedup, the min-bytes filter and the ``max_images`` cap. All the
     state that would race lives here, where there is only one thread.
  2. WORK (parallel) — ``_process_one`` per image, ``SOWSMITH_PDF_IMAGE_CONCURRENCY``
     wide. Results are written into the work item, committed by nobody.
  3. COMMIT (serial, work-list order) — ``_drain_deferred`` writes the training
     rows and spends the thumbnail budget, then atoms are concatenated in
     work-list order.

Output is therefore byte-identical at any concurrency: atom ids are derived from
CONTENT (``stable_id(... region_ref, fact_kind, text)``), never from sequence,
and nothing downstream of the pool ever sees completion order.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core import crop_thumbnail
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
# Closed verdict set for the image head — mirrors headCorrections IMAGE_KINDS.
_IMAGE_KIND_CANDIDATES = [
    "skip", "photo", "diagram", "chart", "table_image", "screenshot",
    "instructions", "label", "map", "logo", "decorative", "signature", "empty",
]

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


# How many images may be in flight at once (``SOWSMITH_PDF_IMAGE_CONCURRENCY``).
#
# WHY 4: the VLM behind this stage is a SHARED Ollama proxy fleet, not a private
# GPU. Ollama serves a model with a small fixed number of parallel slots
# (OLLAMA_NUM_PARALLEL, 4 by default) and queues everything beyond them, so
# width past the slot count buys no throughput — it just deepens a queue that
# other tenants of the same fleet are also waiting in, and pushes individual
# calls toward the 90s per-image timeout that would then LOSE that image's
# atoms. 4 saturates the default slot count without thrashing it. Each in-flight
# image also pins its crop bytes plus a page-text envelope in memory, which is
# another reason not to open this up.
#
# Clamped to [1, 16]: 1 is the exact sequential behaviour (a real setting — it
# is the control arm of the invariance test), and no deployment has a reason to
# fan a shared fleet wider than 16.
_DEFAULT_CONCURRENCY = 4
_MAX_CONCURRENCY = 16


def _concurrency() -> int:
    return max(1, min(_MAX_CONCURRENCY,
                      _int_env("SOWSMITH_PDF_IMAGE_CONCURRENCY", _DEFAULT_CONCURRENCY)))


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


# ── deferred side effects (what makes the parallel phase deterministic) ─
#
# Two side effects inside ``_process_one`` are ORDER-SENSITIVE, so neither may
# run while several images are in flight:
#
#   1. TrainingLog writes (silver / shadow / veto rows). The log is one SQLite
#      connection opened ``check_same_thread=False`` with no lock of its own, so
#      concurrent ``with conn:`` blocks can collide mid-transaction — and this
#      module swallows every logging exception, so a collision would drop a row
#      SILENTLY. That is precisely the loss the repo forbids.
#   2. The per-compile thumbnail budget (``_thumb_budget``), a read-then-
#      increment against ``_thumb_max()``. Racy on its own, but worse than that:
#      even perfectly locked, *which* images win the budget would become
#      completion order, so the same deal could stamp ``crop_thumb`` on
#      different markers run to run.
#
# So both are COLLECTED per work item during the pool and REPLAYED afterwards in
# work-list order (see ``_drain_deferred``) — chosen over a lock because a lock
# fixes only the corruption, not the ordering. Serialising the replay is a free
# side benefit; determinism is the point.
#
# Outside a pool (direct helper calls, and every existing caller) the sinks are
# unset and both effects fire inline exactly as before.

_deferred = threading.local()


@contextmanager
def _deferring(rows: list[Any], thumbs: list[tuple[dict[str, Any], bytes]]):
    """Route this thread's training rows / thumbnail requests into ``rows`` and
    ``thumbs`` instead of committing them."""
    prev_rows = getattr(_deferred, "rows", None)
    prev_thumbs = getattr(_deferred, "thumbs", None)
    _deferred.rows = rows
    _deferred.thumbs = thumbs
    try:
        yield
    finally:
        _deferred.rows = prev_rows
        _deferred.thumbs = prev_thumbs


def _emit_training_rows(rows: list[Any]) -> None:
    """Write training rows, or park them for the deterministic replay."""
    sink = getattr(_deferred, "rows", None)
    if sink is not None:
        sink.extend(rows)
        return
    from app.core.training_log import log_rows
    log_rows(rows)


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


# ``_vision_model`` selects the model by mutating a PROCESS-GLOBAL env var, so
# two overlapping calls could swap it out from under each other. Guarded below.
_VLM_ENV_LOCK = threading.Lock()


def _vlm(image_bytes: bytes, prompt: str, *, model: str | None, max_tokens: int) -> str:
    if _use_ollama_for_pdf_images():
        # No env swap on this path: ``_ollama_vision_direct`` takes the model
        # explicitly and only falls back to OLLAMA_VISION_MODEL when ``model``
        # is None — the exact case where ``_vision_model`` is already a no-op.
        # The wrapper was therefore never load-bearing here, and dropping it is
        # what lets the default (forced-Ollama) path run genuinely in parallel.
        return _ollama_vision_direct(
            image_bytes, prompt, model=model, max_tokens=max_tokens,
        ) or ""
    from app.core.vision_extraction import call_vision_llm
    if not model:
        return call_vision_llm(image_bytes, prompt, max_tokens=max_tokens) or ""
    # Teacher path: ``call_vision_llm`` reads the model from the env, so the
    # swap IS load-bearing and must not overlap. These calls serialise — correct
    # over fast; the default path above is the one that parallelises.
    with _VLM_ENV_LOCK, _vision_model(model):
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


def _log_gate_silver(
    caption: str, ocr: str, image_kind: str, *, via: str,
    attribution: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> None:
    """Silver training row for the CPU gate distillation loop. Never raises.

    Honesty rules (each one closes an observed data-rot pathology):
      * cpu_gate verdicts are NEVER logged — the distilled student teaching
        itself is self-distillation, not new signal (and the gate trainer
        consumes every ``pdf_image_kind`` row, so the exclusion must happen
        here, at the source);
      * a degenerate feature ("no context") is never logged — the verdict
        still applies at runtime, it just teaches nothing;
      * ``teacher`` names who actually decided (store hit vs VLM call);
      * the row id is a content hash (``trn_vlm_<sha16>``, mirroring the
        ``trn_sa_`` audit-import convention) so recompiles of the same deal
        upsert instead of appending duplicates;
      * deal/pdf/page/region attribution rides along — ``deal_id`` as a column
        so the holdout split works, the rest in ``provenance`` for audit.
    """
    if via == "cpu_gate":
        return
    try:
        from app.core.pdf_image_gate import gate_feature_text
        from app.core.training_log import (
            TEACHER_LLM,
            TEACHER_STORE,
            TrainingRow,
        )
        label = "skip" if image_kind in _SKIP_KINDS else image_kind
        feat = gate_feature_text(caption, ocr)
        if not feat.strip() or feat == "no context":
            return
        att = attribution or {}
        deal_id = str(att.get("deal_id") or "")
        pdf_name = str(att.get("pdf") or "")
        region_ref = str(att.get("region_ref") or "")
        image_sha16 = str(att.get("image_sha16") or "")
        row_id = "trn_vlm_" + hashlib.sha256(
            f"pdf_image_kind|{label}|{feat}|{pdf_name}|{region_ref}|{image_sha16}"
            .encode("utf-8")
        ).hexdigest()[:16]
        _emit_training_rows([TrainingRow(
            id=row_id,
            relation="pdf_image_kind",
            label=label,
            raw_text=feat,
            masked_text=feat,
            label_kind="judgment",
            teacher=TEACHER_STORE if via == "store_gate" else TEACHER_LLM,
            confidence=confidence if confidence is not None else 0.7,
            deal_id=deal_id,
            project_id=str(att.get("project_id") or deal_id),
            provenance={
                "stage": "pdf_image_vision_gate",
                "via": via,
                "pdf": pdf_name,
                "page": att.get("page", ""),
                "region_ref": region_ref,
                "image_sha16": image_sha16,
            },
        )])
    except Exception:
        pass


def _log_gate_shadow(
    caption: str,
    ocr: str,
    cpu_shadow: tuple[bool, str, float],
    teacher_kind: str,
    *,
    via: str,
    attribution: dict[str, Any] | None = None,
) -> None:
    """Record a CPU↔teacher (VLM/store) pair without changing routing.

    Relation ``pdf_image_gate_shadow`` is eval/diagnostics only — listed in
    ``NON_TRAINING_RELATIONS`` so nightly retrain never consumes it. Fires only
    when ``SOWSMITH_PDF_IMAGE_GATE_SHADOW=1`` and ``probe()`` answered.
    """
    try:
        from app.core.pdf_image_gate import gate_feature_text
        from app.core.training_log import TrainingRow

        _cpu_meaningful, cpu_kind, cpu_conf = cpu_shadow
        teacher_label = "skip" if teacher_kind in _SKIP_KINDS else (teacher_kind or "skip")
        cpu_label = "skip" if cpu_kind in _SKIP_KINDS or not _cpu_meaningful else cpu_kind
        feat = gate_feature_text(caption, ocr)
        if not feat.strip() or feat == "no context":
            return
        att = attribution or {}
        deal_id = str(att.get("deal_id") or "")
        pdf_name = str(att.get("pdf") or "")
        region_ref = str(att.get("region_ref") or "")
        image_sha16 = str(att.get("image_sha16") or "")
        agree = cpu_label == teacher_label
        row_id = "trn_shadow_" + hashlib.sha256(
            f"pdf_image_gate_shadow|{cpu_label}|{teacher_label}|{feat}|{pdf_name}|{region_ref}|{image_sha16}"
            .encode("utf-8")
        ).hexdigest()[:16]
        _emit_training_rows([TrainingRow(
            id=row_id,
            relation="pdf_image_gate_shadow",
            label=teacher_label,
            raw_text=feat,
            masked_text=feat,
            label_kind="judgment",
            teacher="shadow",
            confidence=float(cpu_conf),
            deal_id=deal_id,
            project_id=str(att.get("project_id") or deal_id),
            provenance={
                "stage": "pdf_image_gate_shadow",
                "via": via,
                "cpu_kind": cpu_label,
                "cpu_conf": float(cpu_conf),
                "teacher_kind": teacher_label,
                "agree": agree,
                "pdf": pdf_name,
                "page": att.get("page", ""),
                "region_ref": region_ref,
                "image_sha16": image_sha16,
            },
        )])
    except Exception:
        pass


def _store_classify_image(
    caption: str, ocr: str, attribution: dict[str, Any] | None = None,
) -> tuple[bool, str, str, float | None] | None:
    """PM correction store-front for image-kind triage (no VLM cost).

    Tries gate feature text, caption, and OCR snippet so corrections committed
    via the image chip can override the classify gate on the next compile.
    Returns (meaningful, kind, via, confidence), or None when the store
    abstains (fall through to VLM gate).
    """
    from app.core.decide import decide
    from app.core.pdf_image_gate import gate_feature_text

    lookups: list[str] = []
    feat = gate_feature_text(caption, ocr)
    if feat.strip():
        lookups.append(feat)
    cap = (caption or "").strip()
    if cap and cap not in lookups:
        lookups.append(cap)
    ocr_s = (ocr or "").strip()[:500]
    if ocr_s and ocr_s not in lookups:
        lookups.append(ocr_s)
    if not lookups:
        return None

    instruction = "Classify the embedded PDF image kind for routing."
    for text in lookups:
        d = decide(
            "pdf_image_kind", text, _IMAGE_KIND_CANDIDATES,
            instruction=instruction, llm=False,
        )
        if d.source != "store" or not d.verdict:
            continue
        kind = str(d.verdict).strip().lower()
        conf = float(getattr(d, "confidence", 0.0) or 0.0) or None
        if kind == "skip" or kind in _SKIP_KINDS:
            _log_gate_silver(caption, ocr, kind or "skip", via="store_gate",
                             attribution=attribution, confidence=conf)
            return False, kind or "skip", "store_gate", conf
        if kind in _IMAGE_KIND_CANDIDATES:
            _log_gate_silver(caption, ocr, kind, via="store_gate",
                             attribution=attribution, confidence=conf)
            return True, kind, "store_gate", conf
    return None


def _classify_image(
    *, crop: bytes, caption: str, saved_path: str,
    attribution: dict[str, Any] | None = None,
) -> tuple[bool, str, str, float | None]:
    """Classify one crop. Returns (meaningful, image_kind, via_tag, confidence).

    ``confidence`` is None when the deciding tier exposes none (cpu/vlm gate).
    cpu_gate verdicts are NOT logged as silver — see :func:`_log_gate_silver`.
    """
    ocr = _ocr_crop(saved_path, crop)  # cheap chain only (no VLM cost on the gate)
    cpu_shadow: tuple[bool, str, float] | None = None
    try:
        from app.core import pdf_image_gate
        # Routing path: GATE_CPU on → may short-circuit to cpu_gate.
        cpu = pdf_image_gate.classify(caption, ocr)
        if cpu is not None:
            meaningful, kind = cpu
            kind = kind if meaningful else "skip"
            return meaningful, kind, "cpu_gate", None
        # Shadow path: score CPU without routing (Phase 3). Never changes
        # the VLM/store decision below — only records the pair.
        if pdf_image_gate.shadow_enabled():
            # Always record argmax (min_conf=0) — shadow is diagnostics, not routing.
            cpu_shadow = pdf_image_gate.probe(caption, ocr, min_conf=0.0)
    except Exception:
        pass
    try:
        store_hit = _store_classify_image(caption, ocr, attribution)
        if store_hit is not None:
            if cpu_shadow is not None:
                _log_gate_shadow(
                    caption, ocr, cpu_shadow, store_hit[1], via="store_gate",
                    attribution=attribution,
                )
            return store_hit
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
    _log_gate_silver(caption, ocr, image_kind or "skip", via="vlm_gate",
                     attribution=attribution)
    if cpu_shadow is not None:
        _log_gate_shadow(
            caption, ocr, cpu_shadow, image_kind or "skip", via="vlm_gate",
            attribution=attribution,
        )
    return meaningful, image_kind, "vlm_gate", None


def _stamp_skip_verdict(
    marker: Any, kind: str, *, via: str, confidence: float | None,
    ocr_preview: str = "",
) -> None:
    """Receipt a skip decision on its image_marker atom so it is traceable.

    Adds RECORDED fields only (``value['gate_verdict']``) — never changes
    which atoms are emitted or how anything routes. ``ocr_preview`` (when
    present) lets the PM culprit surface quote what the crop said without
    re-opening the image. Never raises.
    """
    try:
        val = getattr(marker, "value", None)
        if not isinstance(val, dict):
            return
        verdict: dict[str, Any] = {"kind": kind, "via": via}
        if confidence is not None:
            verdict["confidence"] = confidence
        preview = (ocr_preview or "").strip()
        if preview:
            verdict["ocr_preview"] = preview[:240]
        val["gate_verdict"] = verdict
    except Exception:
        pass


# ── disputed-crop persistence (best-effort, liveness-receipted) ─────
# Only DISPUTED images (a fired hard or soft veto) ever upload — never the
# whole image stream. Same gate as the feedback blob mirror
# (SOWSMITH_FEEDBACK_BLOB + AZURE_STORAGE_CONNECTION_STRING) so one switch
# governs all best-effort blob traffic; the client pattern is replicated
# locally on purpose (this module must not import the feedback/training
# mirrors). Unlike those mirrors, a failure here is NOT silent: the doctrine
# is that best-effort infra must emit a liveness receipt, so an upload that
# dies stamps ``gate_verdict['crop_ref_error']`` instead of nothing.

_TRUTHY = {"1", "true", "yes", "on"}


def _crop_blob_enabled() -> bool:
    """Deliberately OFF (unconfigured) -> no upload and no receipt; that is a
    config choice, not a dead uploader."""
    if os.environ.get("SOWSMITH_FEEDBACK_BLOB", "").strip().lower() not in _TRUTHY:
        return False
    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip())


def _crop_container_client():
    """Blob ContainerClient for disputed crops. Raises on failure — the caller
    turns the exception class into the liveness receipt."""
    from azure.storage.blob import BlobServiceClient
    conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"].strip()
    container = os.environ.get(
        "SOWSMITH_FEEDBACK_BLOB_CONTAINER", "orbitbrief-artifacts"
    ).strip() or "orbitbrief-artifacts"
    return BlobServiceClient.from_connection_string(conn).get_container_client(container)


def _upload_disputed_crop(
    crop: bytes, attribution: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Persist one disputed crop's pixels to blob so a human can SEE what the
    gate skipped. Returns ``(crop_ref, error_class)``:

      * ``(path, None)``  — uploaded; stamp ``crop_ref``.
      * ``(None, name)``  — the uploader is alive-but-failing; stamp
        ``crop_ref_error`` so a dead path shows up in envelopes (never a
        silent zero).
      * ``(None, None)``  — gate off, or nothing attributable to upload.
    """
    if not _crop_blob_enabled():
        return None, None
    att = attribution or {}
    deal_id = str(att.get("deal_id") or "")
    image_sha16 = str(att.get("image_sha16") or "")
    if not crop or not deal_id or not image_sha16:
        return None, None  # unattributable — guess-free, do not invent a path
    path = f"deals/{deal_id}/orbitbrief/disputed_crops/{image_sha16}.png"
    try:
        cc = _crop_container_client()
        cc.upload_blob(name=path, data=crop, overwrite=True)
        return path, None
    except Exception as exc:
        return None, type(exc).__name__


# ── PM thumbnail budget (envelope size discipline) ──────────────────
# ``crop_ref`` is a blob path and no frontend route serves blob images by
# path, so a disputed-skip card today shows the PM nothing. A small JPEG
# data URI stamped as ``gate_verdict['crop_thumb']`` rides the channel the
# verdict already travels (envelope -> core card -> UI) with no new infra.
#
# BUDGET: Orbitbrief-Core caps culprit cards at 20 per envelope, so the worst
# case an envelope can carry is
#     _DEFAULT_THUMB_MAX (20) x SOWSMITH_PDF_IMAGE_THUMB_KB (24KB) ~= 480KB
# of base64. The count cap is global per compile — NOT per card — so a
# pathological deal (hundreds of disputed skips, only 20 of which become
# cards) cannot balloon the envelope with thumbnails nobody will ever see.
# HARD vetoes only: soft vetoes never reach a PM (they feed the review queue),
# so they stay lean and spend none of this budget.
_THUMB_BUDGET_KB = 480  # documented worst case; see arithmetic above
_DEFAULT_THUMB_MAX = 20

_thumb_budget: dict[str, int] = {"used": 0}


def _thumb_max() -> int:
    """Thumbnails embeddable in ONE compile (``SOWSMITH_PDF_IMAGE_THUMB_MAX``).
    0 or negative disables embedding outright."""
    return _int_env("SOWSMITH_PDF_IMAGE_THUMB_MAX", _DEFAULT_THUMB_MAX)


def _reset_thumb_budget() -> None:
    """Called once per compile at the top of :func:`process_image_markers`."""
    _thumb_budget["used"] = 0


def _maybe_thumb(crop: bytes) -> tuple[str | None, str | None]:
    """``(data_uri, error)`` for one card-worthy crop, budget-checked.

    Spends one unit of the per-compile budget on success. Every non-result
    carries a receipt (doctrine: never a silent zero) — the exception class,
    ``'too_large'``, or ``'budget_exhausted'`` when this compile has already
    embedded its allowance. Never raises.
    """
    try:
        if _thumb_budget["used"] >= _thumb_max():
            return None, "budget_exhausted"
        uri, err = crop_thumbnail.make_thumb_data_uri_receipted(crop)
        if uri:
            _thumb_budget["used"] += 1
        return uri, err
    except Exception as exc:
        return None, type(exc).__name__


def _apply_thumb(gv: dict[str, Any], crop: bytes) -> None:
    """Spend budget on one hard-veto crop and stamp the result (or its receipt)
    onto an already-built ``gate_verdict``."""
    thumb, thumb_err = _maybe_thumb(crop)
    if thumb:
        gv["crop_thumb"] = thumb
    elif thumb_err:
        gv["crop_thumb_error"] = thumb_err


def _request_thumb(gv: dict[str, Any], crop: bytes) -> None:
    """Ask for a thumbnail. Inside the pool this only RECORDS the request — the
    budget is spent later, in work-list order, so which images win it never
    depends on which VLM call happened to return first."""
    sink = getattr(_deferred, "thumbs", None)
    if sink is not None:
        sink.append((gv, crop))
        return
    _apply_thumb(gv, crop)


def _log_veto_row(
    caption: str, ocr: str, kind: str, *, via: str, prob: float,
    attribution: dict[str, Any] | None,
    band: str = "hard", crop_ref: str | None = None,
) -> None:
    """Review-queue feed row for a fired skip veto. Never raises.

    This is NOT training silver: ``relation='pdf_image_veto'`` exists so the
    parser review queue can surface skips the veto head disagreed with (wrong
    skips are evidence loss). No trainer consumes this relation — the type /
    span / multitask trainers select their relations by whitelist, and the
    eval-gated registry retrain excludes it explicitly
    (``app.learning.retrain.NON_TRAINING_RELATIONS``) because every row here
    carries the same label ('meaningful') by construction.

    Same honesty rules as :func:`_log_gate_silver`: content-hash id
    (``trn_veto_<sha16>``) so recompiles upsert instead of duplicating, and
    full deal/pdf/page/region attribution for the queue.

    ``band`` records which zone fired: ``'hard'`` (confident disagreement,
    also stamped as ``gate_verdict.veto``) or ``'soft'`` (uncertain zone —
    active-learning harvest only, never a PM card). The id hash deliberately
    excludes the band: one disputed image is ONE queue row, and a recompile
    whose prob drifts across the bar updates that row in place. ``crop_ref``
    (when the disputed crop uploaded) rides in provenance so the queue can
    link the pixels.
    """
    try:
        from app.core.pdf_image_gate import gate_feature_text
        from app.core.training_log import TrainingRow
        feat = gate_feature_text(caption, ocr)
        if not feat.strip() or feat == "no context":
            return
        att = attribution or {}
        deal_id = str(att.get("deal_id") or "")
        pdf_name = str(att.get("pdf") or "")
        region_ref = str(att.get("region_ref") or "")
        image_sha16 = str(att.get("image_sha16") or "")
        row_id = "trn_veto_" + hashlib.sha256(
            f"pdf_image_veto|meaningful|{feat}|{pdf_name}|{region_ref}|{image_sha16}"
            .encode("utf-8")
        ).hexdigest()[:16]
        provenance: dict[str, Any] = {
            "stage": "pdf_image_veto",
            "via": via,
            "band": band,
            "gate_kind": kind,
            "pdf": pdf_name,
            "page": att.get("page", ""),
            "region_ref": region_ref,
            "image_sha16": image_sha16,
            "model": "pdf_image_veto",
        }
        if crop_ref:
            provenance["crop_ref"] = crop_ref
        _emit_training_rows([TrainingRow(
            id=row_id,
            relation="pdf_image_veto",
            label="meaningful",
            raw_text=feat,
            masked_text=feat,
            label_kind="judgment",
            teacher="veto",
            confidence=prob,
            deal_id=deal_id,
            project_id=str(att.get("project_id") or deal_id),
            provenance=provenance,
        )])
    except Exception:
        pass


def _maybe_veto_skip(
    marker: Any, *, caption: str, saved_path: str, crop: bytes,
    kind: str, via: str, attribution: dict[str, Any] | None,
    ocr: str | None = None,
) -> None:
    """Second opinion on a skip verdict — RECORDED ONLY, routing untouched.

    When the vlm/store gate rules skip and the trained veto head confidently
    says meaningful (HARD band, prob >= hard bar), the disagreement is
    (a) stamped into the marker's ``value['gate_verdict']['veto']`` and
    (b) logged as a ``relation='pdf_image_veto'`` row with provenance
    ``band='hard'`` (see :func:`_log_veto_row`). The image still skips either
    way — the veto is a flag for the review queue, never a re-route.

    SOFT band (soft bar <= prob < hard bar): the head is UNCERTAIN — an
    active-learning harvest signal, not a disagreement. Stamped as
    ``gate_verdict['veto_soft']`` and logged with ``band='soft'``; the
    ``'veto'`` key is NEVER set for a soft veto, so Orbitbrief-Core (which
    builds PM culprit cards only from ``gate_verdict.veto``) never bothers a
    PM with the model's own uncertainty — it feeds only the review queue.

    Either band makes the image DISPUTED: its crop pixels are persisted to
    blob (best-effort, gated — see :func:`_upload_disputed_crop`) so a human
    grading the queue can see the image, not just its OCR. Success stamps
    ``gate_verdict['crop_ref']``; an upload failure stamps
    ``gate_verdict['crop_ref_error']`` (liveness receipt, never silence).

    HARD vetoes additionally carry a small inline JPEG data URI in
    ``gate_verdict['crop_thumb']`` so the PM culprit card can RENDER the image
    (``crop_ref`` is a blob path nothing serves). Failure or an exhausted
    per-compile budget stamps ``gate_verdict['crop_thumb_error']`` — same
    never-silent doctrine. Soft vetoes get no thumbnail: they never reach a PM.

    cpu_gate verdicts are NEVER veto-checked: once the distilled gate serves,
    the veto second-guessing its sibling student (trained on the same feature
    text) is self-review, not independent signal. Never raises.
    """
    if via == "cpu_gate":
        return
    try:
        from app.core import pdf_image_veto
        if not pdf_image_veto.enabled():
            return
        if ocr is None:
            ocr = _ocr_crop(saved_path, crop)  # cheap chain only, same as the gate
        prob = pdf_image_veto.veto(caption, ocr or "")
        band = "hard"
        if prob is None:
            prob = pdf_image_veto.soft_veto(caption, ocr or "")
            band = "soft"
        if prob is None:
            return
        crop_ref, crop_err = _upload_disputed_crop(crop, attribution)
        val = getattr(marker, "value", None)
        if isinstance(val, dict) and isinstance(val.get("gate_verdict"), dict):
            gv = val["gate_verdict"]
            if band == "hard":
                gv["veto"] = {"meaningful_prob": prob, "model": "pdf_image_veto"}
                # Card-worthy: carry the pixels inline so the PM sees the image
                # without going looking. Hard band only — see the budget block.
                _request_thumb(gv, crop)
            else:
                gv["veto_soft"] = {"meaningful_prob": prob}
            if crop_ref:
                gv["crop_ref"] = crop_ref
            elif crop_err:
                gv["crop_ref_error"] = crop_err
        _log_veto_row(
            caption, ocr or "", kind, via=via, prob=prob,
            attribution=attribution, band=band, crop_ref=crop_ref,
        )
    except Exception:
        pass


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


@dataclass
class _ImageWork:
    """One image selected for VLM work, plus everywhere its results land.

    Selection (which images, in what order) is decided SEQUENTIALLY before any
    VLM call; only ``_process_one`` fans out. The result slots are per-item, so
    two workers never touch the same object.
    """
    marker: Any
    pdf_name: str
    page_index: int
    region_ref: str
    saved_path: str
    caption: str
    crop: bytes
    crop_hash: str
    atoms: list[EvidenceAtom] = field(default_factory=list)
    rows: list[Any] = field(default_factory=list)
    thumbs: list[tuple[dict[str, Any], bytes]] = field(default_factory=list)


def _build_work_list(
    atoms: list[Any], *, max_images: int, min_bytes: int,
) -> list[_ImageWork]:
    """Phase 1 — pick the images, sequentially and with NO VLM calls.

    Dedup (identical crops by ``sha256[:16]``), the min-bytes filter and the
    ``max_images`` cap all live here, walking ``_iter_image_markers`` in its
    natural order. Keeping this phase serial is what makes the parallel phase
    safe to reason about: the cap and the dedup set are the two pieces of shared
    state that would otherwise race, and here they cannot — the same deal picks
    the same images in the same order at any concurrency.
    """
    work: list[_ImageWork] = []
    seen_hashes: set[str] = set()
    for marker, pdf_name, page_index, region_ref, saved_path, caption in _iter_image_markers(atoms):
        if len(work) >= max_images:
            break
        crop = _load_crop(saved_path)
        if len(crop) < min_bytes:
            continue
        crop_hash = hashlib.sha256(crop).hexdigest()[:16]
        if crop_hash in seen_hashes:
            continue
        seen_hashes.add(crop_hash)
        work.append(_ImageWork(
            marker=marker, pdf_name=pdf_name, page_index=page_index,
            region_ref=region_ref, saved_path=saved_path, caption=caption,
            crop=crop, crop_hash=crop_hash,
        ))
    return work


def _run_work_item(w: _ImageWork, cfg: dict[str, Any]) -> None:
    """Phase 2 body — the expensive part, run once per image, possibly in
    parallel. Writes ONLY into ``w``; commits nothing. A failure here is
    contained to this one image (abstain), exactly as before."""
    try:
        with _deferring(w.rows, w.thumbs):
            w.atoms = _process_one(
                marker=w.marker, pdf_name=w.pdf_name, page_index=w.page_index,
                region_ref=w.region_ref, saved_path=w.saved_path,
                caption=w.caption, crop=w.crop, crop_hash=w.crop_hash,
                neighbor_chars=cfg["neighbor_chars"],
                max_page_chars=cfg["max_page_chars"],
                guard_min=cfg["guard_min"], caption_min=cfg["caption_min"],
            )
    except Exception as exc:  # one bad image never breaks the compile
        logger.warning("pdf_image_vision: %s %s failed: %s", w.pdf_name, w.region_ref, exc)
        w.atoms = []
        # w.rows / w.thumbs keep whatever was collected before the failure, so a
        # half-processed image still receipts what it learned (never a silent
        # zero); the surviving images are untouched.


def _drain_deferred(work: list[_ImageWork]) -> None:
    """Phase 3 — replay the collected side effects in WORK-LIST order.

    Single-threaded by construction, so the SQLite writes cannot collide, and
    the thumbnail budget is spent by the same images it would have been spent by
    sequentially. Each item is isolated: a bad row never costs the next one.
    """
    # ``TrainingLog.rows()`` reads back ``ORDER BY created_at``, and a row's
    # created_at is stamped when the TrainingRow is CONSTRUCTED — inside the
    # worker. Left alone, the log would read back in completion order, which is
    # a real (if quiet) regression: today it reads back in image order. Restamp
    # at commit time, strictly increasing, so created_at means what it meant
    # sequentially — when the row was written. The 1us spacing is not cosmetic:
    # time.time() ties easily across a tight loop, and ties make the ORDER BY
    # arbitrary again. Nothing else in the repo reads this field's value.
    stamp = time.time()
    for w in work:
        if w.rows:
            try:
                from app.core.training_log import log_rows
                for row in w.rows:
                    row.created_at = stamp
                    stamp += 1e-6
                log_rows(w.rows)
            except Exception:
                pass
        for gv, crop in w.thumbs:
            _apply_thumb(gv, crop)


def process_image_markers(atoms: list[Any]) -> list[EvidenceAtom]:
    """Describe / transcribe embedded PDF images into NEW atoms.

    Three phases (see ``_build_work_list`` / ``_run_work_item`` /
    ``_drain_deferred``): pick the images serially, do the VLM work with a small
    bounded pool, then commit the order-sensitive side effects in the order the
    images were picked. Output is byte-identical at any concurrency — atom ids
    are content-derived, and ``out`` is assembled from the work list, never from
    completion order.

    Returns [] when disabled, no vision endpoint, or nothing qualifies. Never
    raises and never removes or rewrites input atoms — the only mutation is
    recording the gate verdict on skipped image markers (a receipt, not a
    routing change)."""
    if not enabled() or not atoms:
        return []
    _reset_thumb_budget()  # the thumbnail allowance is PER COMPILE
    try:
        if not _vision_reachable():
            logger.info("pdf_image_vision: no vision endpoint; abstaining")
            return []
    except Exception:
        return []

    max_images = _int_env("SOWSMITH_PDF_IMAGE_MAX", 40)
    min_bytes = _int_env("SOWSMITH_PDF_IMAGE_MIN_BYTES", 3000)
    cfg = {
        "neighbor_chars": _int_env("SOWSMITH_PDF_IMAGE_NEIGHBOR_CHARS", 600),
        "max_page_chars": _int_env("SOWSMITH_PDF_IMAGE_PAGE_CHARS", 4000),
        "guard_min": _float_env("SOWSMITH_PDF_IMAGE_GUARD_MIN", 0.25),
        "caption_min": _float_env("SOWSMITH_PDF_IMAGE_CAPTION_MIN", 0.2),
    }

    work = _build_work_list(atoms, max_images=max_images, min_bytes=min_bytes)
    if not work:
        return []

    width = min(_concurrency(), len(work))
    if width <= 1:
        for w in work:
            _run_work_item(w, cfg)
    else:
        with ThreadPoolExecutor(
            max_workers=width, thread_name_prefix="pdf_image_vision",
        ) as pool:
            # list() drains the map so every image is done before we commit.
            list(pool.map(lambda w: _run_work_item(w, cfg), work))

    _drain_deferred(work)

    out: list[EvidenceAtom] = []
    for w in work:  # work-list order, NEVER completion order
        out.extend(w.atoms)
    if out:
        logger.info(
            "pdf_image_vision: %d atoms from %d images (concurrency %d)",
            len(out), len(work), width,
        )
    return out


def _process_one(
    *, marker: Any, pdf_name: str, page_index: int, region_ref: str,
    saved_path: str, caption: str, crop: bytes,
    neighbor_chars: int, max_page_chars: int, guard_min: float,
    caption_min: float, crop_hash: str = "",
) -> list[EvidenceAtom]:
    project_id = str(getattr(marker, "project_id", "") or "")
    attribution = {
        "deal_id": project_id,  # the compiler uses project_id as deal_id
        "project_id": project_id,
        "pdf": pdf_name,
        "page": page_index,
        "region_ref": region_ref,
        "image_sha16": crop_hash,
    }
    meaningful, image_kind, via, gate_conf = _classify_image(
        crop=crop, caption=caption, saved_path=saved_path,
        attribution=attribution,
    )
    if not meaningful or image_kind in _SKIP_KINDS or not image_kind:
        # One OCR pass for the receipt + veto (both recorded-only).
        ocr_preview = _ocr_crop(saved_path, crop)
        _stamp_skip_verdict(
            marker, image_kind or "skip", via=via, confidence=gate_conf,
            ocr_preview=ocr_preview,
        )
        _maybe_veto_skip(
            marker, caption=caption, saved_path=saved_path, crop=crop,
            kind=image_kind or "skip", via=via, attribution=attribution,
            ocr=ocr_preview,
        )
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
    if atoms and via in ("cpu_gate", "store_gate"):
        for a in atoms:
            a.value["gate_via"] = via
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
