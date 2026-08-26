"""Harvest pdf_image_kind silver CANDIDATES from real deal PDFs.

The CPU image gate (``app/core/pdf_image_gate.py``) is supposed to distill the
VLM triage step, but the live TrainingLog holds almost no ``pdf_image_kind``
rows. This tool grows the candidate pool WITHOUT any VLM/teacher call:

  1. enumerate deal PDFs on blob (``az`` CLI subprocess; degrades with a clear
     message when auth/CLI is unavailable) — or read local PDFs (``--local-dir``),
  2. extract every embedded raster image per page (PyMuPDF),
  3. build the SAME feature text the gate sees at runtime
     (``gate_feature_text(caption, ocr)``), with a nearest-text-above heuristic
     as the caption fallback and the PDF text layer near the image as the OCR
     fallback when no OCR engine is installed,
  4. flag candidates whose feature text hits PM-critical vocabulary
     (``zero_miss.PM_CRITICAL_TERMS``) — the cheap "a skip here would be a
     culprit" signal,
  5. emit JSONL (+ CSV) of candidates for later human / teacher labeling.

Guess-free: rows record where each field came from (``caption_source``,
``ocr_source``); a field the tool cannot determine is empty, never invented.
NO labels are assigned here — label acquisition is a separate step.

Outputs default under the session scratch dir and must NEVER be committed
(repo rule: code + tests only, no datasets).

Usage:
  python tools/harvest_image_silver.py --deal 005a4c6b-7e26-4c94-a8b2-aee86afb4f7b
  python tools/harvest_image_silver.py --max-deals 5 --out /tmp/scratch/cands.jsonl
  python tools/harvest_image_silver.py --local-dir real_data_cases  # no blob
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo imports (tool runs from the repo root; pyproject pythonpath is ".").
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.pdf_image_gate import gate_feature_text  # noqa: E402
from app.core.zero_miss import PM_CRITICAL_TERMS  # noqa: E402

STORAGE_ACCOUNT = "purpulsedevstg01"
DEALS_CONTAINER = "orbitbrief-artifacts"

# Mirrors the pipeline's floor (SOWSMITH_PDF_IMAGE_MIN_BYTES default 3000):
# smaller embedded images are borders / spacers / icons, never content.
MIN_IMAGE_BYTES = 3000
# Caption heuristic: a text block qualifies as "above the image" when its
# bottom edge sits within this many points of the image top.
CAPTION_MAX_GAP_PTS = 60.0
CAPTION_MAX_CHARS = 300
OCR_SNIPPET_CHARS = 500

_COMPILED_PM_TERMS = [
    (re.compile(t["pat"], re.IGNORECASE), t["pat"], t["kind"])
    for t in PM_CRITICAL_TERMS
]


# ── pure functions (unit-tested) ────────────────────────────────────


def build_feature_text(caption: str, ocr: str) -> str:
    """EXACTLY what the gate sees at runtime — one source of truth."""
    return gate_feature_text(caption, ocr)


def suspicious_hits(feature_text: str) -> list[dict[str, str]]:
    """PM-critical vocabulary hits in the feature text.

    Any hit means: if the gate (or the VLM) says "skip" for this image, it may
    be skipping PM-critical content — a culprit candidate for review.
    """
    hits: list[dict[str, str]] = []
    text = feature_text or ""
    for rx, pat, kind in _COMPILED_PM_TERMS:
        m = rx.search(text)
        if m:
            hits.append({"pattern": pat, "kind": kind, "matched": m.group(0)})
    return hits


def image_dedup_key(image_bytes: bytes) -> str:
    """Same identity the pipeline uses (sha256[:16] of crop bytes)."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


@dataclass
class Block:
    """One PDF text block: bbox in page points + its text."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def _h_overlap(b: Block, rect: tuple[float, float, float, float]) -> float:
    rx0, _, rx1, _ = rect
    return max(0.0, min(b.x1, rx1) - max(b.x0, rx0))


def nearest_caption_above(
    blocks: list[Block],
    rect: tuple[float, float, float, float],
    *,
    max_gap: float = CAPTION_MAX_GAP_PTS,
) -> str:
    """Nearest text block ABOVE the image with horizontal overlap.

    Fallback caption heuristic (the pipeline's markers carry a richer
    ``expected_content`` caption; standalone harvest has only the text layer).
    Returns "" when no block qualifies — no guessing.
    """
    _, ry0, _, _ = rect
    best: Block | None = None
    best_gap = max_gap
    for b in blocks:
        if not b.text.strip():
            continue
        gap = ry0 - b.y1
        if gap < 0 or gap > best_gap:
            continue
        if _h_overlap(b, rect) <= 0:
            continue
        if best is None or gap < best_gap or (gap == best_gap):
            best, best_gap = b, gap
    if best is None:
        return ""
    return " ".join(best.text.split())[:CAPTION_MAX_CHARS]


def text_layer_near_rect(
    blocks: list[Block],
    rect: tuple[float, float, float, float],
    *,
    pad: float = 8.0,
) -> str:
    """PDF text-layer content overlapping the (padded) image rect.

    OCR degradation path: many "images" in deal PDFs sit under/over selectable
    text; when no OCR engine is installed this is the honest cheap substitute.
    """
    rx0, ry0, rx1, ry1 = rect
    rx0, ry0, rx1, ry1 = rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad
    parts: list[str] = []
    for b in blocks:
        if not b.text.strip():
            continue
        if b.x1 <= rx0 or b.x0 >= rx1 or b.y1 <= ry0 or b.y0 >= ry1:
            continue
        parts.append(" ".join(b.text.split()))
    return "\n".join(parts)


def dedup_candidates(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeat images by content hash (first occurrence wins) — the same
    letterhead/logo recurring on every page must not flood the pool."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in cands:
        key = c.get("image_sha16") or ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(c)
    return out


# ── OCR (cheap chain; degrade honestly) ─────────────────────────────


def _ocr_engine():
    """Return (fn(bytes)->str, source_name) for the cheapest local OCR, or
    (None, reason). Tries pytesseract only — no network engines, no keys."""
    try:
        import io

        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]

        pytesseract.get_tesseract_version()  # raises when binary is absent

        def _run(data: bytes) -> str:
            try:
                img = Image.open(io.BytesIO(data))
                return (pytesseract.image_to_string(img) or "").strip()
            except Exception:
                return ""

        return _run, "tesseract"
    except Exception as exc:
        return None, f"unavailable ({type(exc).__name__})"


# ── PDF extraction ──────────────────────────────────────────────────


@dataclass
class HarvestStats:
    pdfs: int = 0
    images_seen: int = 0
    images_kept: int = 0
    dupes_dropped: int = 0
    too_small: int = 0
    ocr_source_counts: dict[str, int] = field(default_factory=dict)


def _page_blocks(page) -> list[Block]:
    out: list[Block] = []
    try:
        for blk in page.get_text("blocks"):
            x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
            if isinstance(text, str) and text.strip():
                out.append(Block(float(x0), float(y0), float(x1), float(y1), text))
    except Exception:
        pass
    return out


def harvest_pdf(
    pdf_path: Path,
    *,
    deal_id: str,
    ocr_fn,
    ocr_source: str,
    stats: HarvestStats,
    min_bytes: int = MIN_IMAGE_BYTES,
) -> list[dict[str, Any]]:
    """Extract embedded-image candidates from one PDF. Never raises."""
    try:
        import pymupdf  # type: ignore[import-not-found]
    except Exception:
        import fitz as pymupdf  # type: ignore[import-not-found, no-redef]
    cands: list[dict[str, Any]] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        print(f"  ! cannot open {pdf_path.name}: {exc}", file=sys.stderr)
        return cands
    try:
        stats.pdfs += 1
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            blocks = _page_blocks(page)
            try:
                infos = page.get_image_info(xrefs=True)
            except Exception:
                infos = []
            for info in infos:
                stats.images_seen += 1
                xref = int(info.get("xref") or 0)
                rect = tuple(float(v) for v in (info.get("bbox") or (0, 0, 0, 0)))
                image_bytes = b""
                if xref > 0:
                    try:
                        image_bytes = doc.extract_image(xref).get("image") or b""
                    except Exception:
                        image_bytes = b""
                if len(image_bytes) < min_bytes:
                    stats.too_small += 1
                    continue
                caption = nearest_caption_above(blocks, rect)
                caption_source = "text_above" if caption else "none"
                ocr_text = ""
                used_ocr_source = "none"
                if ocr_fn is not None:
                    ocr_text = ocr_fn(image_bytes)
                    if ocr_text:
                        used_ocr_source = ocr_source
                if not ocr_text:
                    ocr_text = text_layer_near_rect(blocks, rect)
                    if ocr_text:
                        used_ocr_source = "text_layer"
                stats.ocr_source_counts[used_ocr_source] = (
                    stats.ocr_source_counts.get(used_ocr_source, 0) + 1
                )
                feat = build_feature_text(caption, ocr_text)
                hits = suspicious_hits(feat)
                # region_ref mirrors the pipeline's marker naming.
                region_ref = f"page{page_index}/image{xref}"
                cands.append({
                    "deal_id": deal_id,
                    "pdf": pdf_path.name,
                    "page": page_index,
                    "image_ref": region_ref,
                    "image_sha16": image_dedup_key(image_bytes),
                    "image_bytes": len(image_bytes),
                    "caption": caption,
                    "caption_source": caption_source,
                    "ocr_snippet": ocr_text[:OCR_SNIPPET_CHARS],
                    "ocr_source": used_ocr_source,
                    "feature_text": feat,
                    "suspicious_skip": bool(hits),
                    "pm_term_hits": hits,
                })
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return cands


# ── blob enumeration (az CLI subprocess) ────────────────────────────


def _az(args: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            ["az", *args], capture_output=True, text=True, timeout=300,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "az CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", "az CLI timed out"


def list_deal_pdfs(deal_id: str) -> list[str]:
    """Blob names of PDFs under deals/<deal_id>/artifacts/. [] on auth failure
    (with a clear message) — the caller decides whether to fall back."""
    code, out, err = _az([
        "storage", "blob", "list",
        "--account-name", STORAGE_ACCOUNT,
        "--container-name", DEALS_CONTAINER,
        "--prefix", f"deals/{deal_id}/artifacts/",
        "--auth-mode", "login",
        "--query", "[].name", "-o", "json",
    ])
    if code != 0:
        print(
            f"  ! blob list failed for deal {deal_id} (az exit {code}): "
            f"{err.strip()[:200] or 'no stderr'} — is `az login` current?",
            file=sys.stderr,
        )
        return []
    try:
        names = json.loads(out)
    except Exception:
        return []
    return [n for n in names if isinstance(n, str) and n.lower().endswith(".pdf")]


def list_deal_ids(limit: int) -> list[str]:
    """First `limit` distinct deal UUIDs that have blobs. [] on failure."""
    code, out, err = _az([
        "storage", "blob", "list",
        "--account-name", STORAGE_ACCOUNT,
        "--container-name", DEALS_CONTAINER,
        "--prefix", "deals/",
        "--num-results", "5000",
        "--auth-mode", "login",
        "--query", "[].name", "-o", "json",
    ])
    if code != 0:
        print(
            f"  ! blob enumeration failed (az exit {code}): "
            f"{err.strip()[:200] or 'no stderr'} — is `az login` current?",
            file=sys.stderr,
        )
        return []
    ids: list[str] = []
    try:
        for name in json.loads(out):
            parts = str(name).split("/")
            if len(parts) >= 2 and parts[0] == "deals":
                if parts[1] not in ids:
                    ids.append(parts[1])
            if len(ids) >= limit:
                break
    except Exception:
        return []
    return ids


def download_blob(name: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True  # cached from a previous run
    code, _, err = _az([
        "storage", "blob", "download",
        "--account-name", STORAGE_ACCOUNT,
        "--container-name", DEALS_CONTAINER,
        "--name", name,
        "--file", str(dest),
        "--auth-mode", "login",
        "--no-progress",
    ])
    if code != 0:
        print(f"  ! download failed {name}: {err.strip()[:200]}", file=sys.stderr)
        return False
    return True


# ── output ──────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "deal_id", "pdf", "page", "image_ref", "image_sha16", "image_bytes",
    "caption", "caption_source", "ocr_snippet", "ocr_source",
    "feature_text", "suspicious_skip", "pm_term_kinds",
]


def write_outputs(cands: list[dict[str, Any]], out_jsonl: Path) -> Path:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    out_csv = out_jsonl.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in cands:
            row = dict(c)
            row["pm_term_kinds"] = ";".join(
                sorted({h["kind"] for h in c.get("pm_term_hits", [])})
            )
            w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
    return out_csv


# ── main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    default_scratch = Path(
        "/private/tmp/claude-501/-Users-purtera/"
        "688e4b18-55b8-4411-9d01-a00049d5ca4f/scratchpad/image-silver-data"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deal", action="append", default=[],
                    help="deal UUID to harvest (repeatable)")
    ap.add_argument("--max-deals", type=int, default=0,
                    help="also enumerate the first N deals from blob")
    ap.add_argument("--local-dir", type=Path, default=None,
                    help="harvest local PDFs under this dir instead of blob")
    ap.add_argument("--scratch", type=Path, default=default_scratch,
                    help="download/cache dir (NEVER inside the repo)")
    ap.add_argument("--out", type=Path, default=None,
                    help="candidates JSONL path (default under --scratch)")
    ap.add_argument("--min-bytes", type=int, default=MIN_IMAGE_BYTES)
    args = ap.parse_args(argv)

    out_jsonl = args.out or (args.scratch / "image_silver_candidates.jsonl")
    ocr_fn, ocr_source = _ocr_engine()
    if ocr_fn is None:
        print(
            f"OCR engine {ocr_source}; degrading to PDF text-layer near each "
            "image (ocr_source=text_layer in output).",
        )

    stats = HarvestStats()
    cands: list[dict[str, Any]] = []

    pdf_jobs: list[tuple[str, Path]] = []  # (deal_id, local pdf path)
    if args.local_dir:
        for p in sorted(args.local_dir.rglob("*.pdf")):
            pdf_jobs.append(("local:" + p.parent.name, p))
        if not pdf_jobs:
            print(f"No PDFs under {args.local_dir}", file=sys.stderr)
            return 1
    else:
        deal_ids = list(dict.fromkeys(args.deal))
        if args.max_deals:
            for d in list_deal_ids(args.max_deals + len(deal_ids)):
                if d not in deal_ids:
                    deal_ids.append(d)
                if len(deal_ids) >= args.max_deals + len(args.deal):
                    break
        if not deal_ids:
            print("No deals given (--deal/--max-deals) and no --local-dir; "
                  "nothing to do.", file=sys.stderr)
            return 1
        any_blob = False
        for deal_id in deal_ids:
            names = list_deal_pdfs(deal_id)
            if names:
                any_blob = True
            print(f"deal {deal_id}: {len(names)} PDFs on blob")
            for name in names:
                dest = args.scratch / "pdfs" / name.replace("/", "__")
                if download_blob(name, dest):
                    pdf_jobs.append((deal_id, dest))
        if not any_blob:
            print(
                "Blob access unavailable or no PDFs found. Re-run with "
                "--local-dir real_data_cases to use the in-repo corpus.",
                file=sys.stderr,
            )
            return 1

    for deal_id, pdf_path in pdf_jobs:
        got = harvest_pdf(
            pdf_path, deal_id=deal_id, ocr_fn=ocr_fn, ocr_source=ocr_source,
            stats=stats, min_bytes=args.min_bytes,
        )
        cands.extend(got)

    before = len(cands)
    cands = dedup_candidates(cands)
    stats.dupes_dropped = before - len(cands)
    stats.images_kept = len(cands)

    out_csv = write_outputs(cands, out_jsonl)
    n_feat = sum(1 for c in cands if c["feature_text"] != "no context")
    n_susp = sum(1 for c in cands if c["suspicious_skip"])
    print(
        f"\nHarvest: {stats.pdfs} PDFs, {stats.images_seen} embedded images seen, "
        f"{stats.too_small} below {args.min_bytes}B, {stats.dupes_dropped} dupes "
        f"dropped, {stats.images_kept} candidates kept."
    )
    if stats.images_kept:
        print(
            f"Feature text non-empty: {n_feat}/{stats.images_kept} "
            f"({100.0 * n_feat / stats.images_kept:.0f}%); suspicious "
            f"(PM-critical vocab): {n_susp}/{stats.images_kept} "
            f"({100.0 * n_susp / stats.images_kept:.0f}%)."
        )
        print(f"OCR sources: {stats.ocr_source_counts}")
    print(f"Wrote {out_jsonl}\n      {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
