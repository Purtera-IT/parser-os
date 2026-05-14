"""OrbitBrief PDF parser — wires the color-driven page OS extractor into Parser OS.

Pipeline per artifact::

    PDF -> orbitbrief_page_os.detect (per page)
        -> overlay JSON payload
        -> structured_extract.extract_structured (per page)
        -> orbitbrief.pdf.structured.v1 document (all pages, hierarchical)
        -> structured.md projection (LLM-friendly mirror of the JSON)
        -> EvidenceAtom stream (one atom per content block, typed by
           section context so OrbitBrief knows what each chunk *means*)

The structured JSON + markdown pair is the "perfect compressible
OrbitBrief input format" for a single PDF.  See
``app.core.orbitbrief_envelope`` for the project-level envelope that
fuses every parser's structured projection into a single
``orbitbrief.input.v1`` payload an open-source LLM can swallow in one
prompt.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserCapability,
    ParserMatch,
    ParserOutput,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser

PARSER_NAME = "orbitbrief_pdf"
PARSER_VERSION = "orbitbrief_pdf_v3"
STRUCTURED_SCHEMA_VERSION = "orbitbrief.pdf.structured.v1"
DERIVED_DIR_SUFFIX = ".derived"
STRUCTURED_FILENAME = "structured.json"
STRUCTURED_MARKDOWN_FILENAME = "structured.md"
EXTRACTION_METHOD = "orbitbrief_pdf_color_driven_v1"

DEFAULT_BLOCK_CONFIDENCE = 0.88
DEFAULT_NOTE_CONFIDENCE = 0.78
TABLE_ROW_CONFIDENCE = 0.92  # tables are the most-trustworthy structure on a page

PDF_MAGIC = b"%PDF-"


# ─── PRODUCTION_GAPS P1.1: Q&A-aware paragraph segmentation ───
# When PDF text extraction collapses an entire pre-proposal Q&A
# transcript into a single paragraph block, downstream packet anchors
# become unusable 2,400-character keys.  This regex splits a paragraph
# at every Q-marker / A-marker boundary so each Q-pair becomes its own
# atom (with its own entity_keys, qa:qN markers, etc.).
_QA_BOUNDARY_REGEX = re.compile(r"(?=(?:^|\s)[QA]\d{1,3}\.\s)")
_QA_PAIR_PROBE = re.compile(r"\b[QA]\d{1,3}\.\s")


def _split_qa_blob(text: str) -> list[str]:
    """Split a paragraph at Q\\d. / A\\d. boundaries.

    Returns the original text as a singleton if no boundaries are
    found, or fewer than 2 distinct Q-or-A markers are present.
    """
    if not text:
        return []
    markers = _QA_PAIR_PROBE.findall(text)
    if len(markers) < 2:
        return [text]
    parts = [p.strip() for p in _QA_BOUNDARY_REGEX.split(text) if p.strip()]
    # Coalesce consecutive Q-then-A into one chunk so downstream packet
    # anchors get the full Q+A context but two pairs don't get fused.
    merged: list[str] = []
    pending: str | None = None
    for part in parts:
        if pending is None:
            pending = part
            continue
        # If pending starts with Q\d. and this one starts with A\d. with
        # the *same* number, merge them; otherwise flush pending and
        # start fresh.
        m_pending = re.match(r"\b([QA])(\d{1,3})\.", pending)
        m_part = re.match(r"\b([QA])(\d{1,3})\.", part)
        if (
            m_pending
            and m_part
            and m_pending.group(1) == "Q"
            and m_part.group(1) == "A"
            and m_pending.group(2) == m_part.group(2)
        ):
            pending = f"{pending} {part}".strip()
        else:
            merged.append(pending)
            pending = part
    if pending:
        merged.append(pending)
    return merged or [text]


# ─── PRODUCTION_GAPS P1.2: form-field template detection ───
# Vendor-info forms ("FULL LEGAL NAME (PRINT) ...", "Federal Taxpayer
# Number (ID#)", "col_4: DATE") add atom-count noise without scope
# value.  We detect these by counting form-field markers and skip
# atom emission entirely when the paragraph is dominated by them.
#
# Strong markers — fingerprints unique to vendor-info templates.
_FORM_FIELD_STRONG_MARKERS = (
    "(print)",
    "(in ink)",
    "(if applicable)",
    "(if different",
    "id#",
    "fein",
    "duns",
    "spin",
    "frn",
    "ein number",
    "ssn number",
    "tin number",
    "______",
)
# Weak markers — placeholder column-names produced by the structured
# table extractor when the source row had no proper header.  They show
# up in legitimate tables (NATOMAS school list) too, so we only let
# them count *when paired with a strong marker*.  See Week 6 P6.6 —
# without this distinction the school list (5 placeholder columns) was
# blanket-rejected as a form-field template.
_FORM_FIELD_WEAK_MARKERS = (
    "col_1:",
    "col_2:",
    "col_3:",
    "col_4:",
    "col_5:",
    "col_6:",
    "col_7:",
    "col_8:",
)
_FORM_FIELD_MARKERS = _FORM_FIELD_STRONG_MARKERS + _FORM_FIELD_WEAK_MARKERS
_FORM_FIELD_KEYWORDS = (
    "full legal name",
    "federal taxpayer number",
    "billing name",
    "purchase order address",
    "payment address",
    "business name",
    "dba name",
    "authorized representative",
    "contact name/title",
    "name (print",
    "address:",
    "telephone:",
    "fax:",
    "fax number",
    "tax id",
    "tax id#",
    "tax id number",
    "tax identification number",
    "duns number",
    "fein number",
)


def _looks_like_form_field(text: str) -> bool:
    """Detect vendor-info form-field templates.

    Decision rules (any one is sufficient):
      * ≥1 strong marker AND ≥1 other marker (strong or weak)
      * ≥3 form-field keywords ("Full Legal Name", "FEIN Number", …)
      * Long underscore run (blank form line) plus any marker

    Weak markers alone (the placeholder ``col_N:`` column names) are
    NOT enough — they appear in legitimate tables (NATOMAS school
    list) when the structured extractor couldn't infer headers.

    Tuned against the VT-CAM "FULL LEGAL NAME (PRINT) (Company name as
    it appears with your Federal Taxpayer Number): ..." templates that
    were emitting at 0.92 confidence with 0 entity keys.
    """
    if not text:
        return False
    text_lower = text.lower()
    strong_hits = sum(1 for m in _FORM_FIELD_STRONG_MARKERS if m in text_lower)
    weak_hits = sum(1 for m in _FORM_FIELD_WEAK_MARKERS if m in text_lower)
    if strong_hits >= 2:
        return True
    if strong_hits >= 1 and (strong_hits + weak_hits) >= 2:
        return True
    keyword_hits = sum(1 for kw in _FORM_FIELD_KEYWORDS if kw in text_lower)
    if keyword_hits >= 3:
        return True
    if "____" in text and (strong_hits + weak_hits) >= 1:
        return True
    return False


# ─── PRODUCTION_GAPS P1.3: page-footer / page-header detection ───
# Example: "RFP 25-107 Wireless Equipment November 20, 2024 Technology
# Services Department Page 17 of 25".  These appear once per page (often
# as both a footer and a redundant header band) and contribute pure
# noise — they're the same string with only the page number changing,
# so they pollute the atom set with N copies per N-page PDF.
_PAGE_NUMBER_PATTERN = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
_PAGE_FOOTER_HINTS = (
    "rfp ",
    "rfp#",
    "rfp:",
    "request for proposal",
    "purchase order",
    "po #",
    "section ",
    "exhibit ",
    "addendum",
    "all rights reserved",
    "copyright",
    "confidential",
    "proprietary",
)


def _looks_like_page_footer(text: str) -> bool:
    """Detect repeating page-footer / page-header band text.

    Two complementary signals:
    1. The literal "Page N of M" pattern (very high precision) — by
       itself enough when text is short.
    2. A "Page N" suffix on a short line (≤ 220 chars) — common for
       footers that omit the "of M" half.

    A short line containing "Page N of M" but no Q\\d./A\\d. or
    sentence-shaped scope content is treated as a footer.  Q&A
    paragraphs slip past because the splitter handles them earlier.
    """
    if not text:
        return False
    if len(text) > 240:
        return False  # Real footers are short; long blocks are scope.
    if _PAGE_NUMBER_PATTERN.search(text):
        return True
    # "Page 17" alone (no "of M") on a short line that also carries an
    # RFP/footer hint is also a footer.
    if re.search(r"\bpage\s+\d+\b", text, re.IGNORECASE):
        text_lower = text.lower()
        if any(hint in text_lower for hint in _PAGE_FOOTER_HINTS):
            # Make sure it doesn't carry quantitative info that scope
            # atoms care about.
            has_money = bool(re.search(r"\$\s*\d", text))
            has_qty = bool(re.search(r"\b\d+(?:,\d{3})*\s*(?:cameras?|aps?|drops?|outlets?|jacks?|users?|licenses?|installations?)\b", text, re.IGNORECASE))
            if not (has_money or has_qty):
                return True
    return False


# Page-footer band prefix detector.  When PDF text extraction folds the
# header/footer band into the start of a real paragraph (Natomas: every
# page yielded one mega-atom of "RFP 25-107 ... Page N of 25 <real
# scope content>"), we want to *strip* the band, not drop the atom.
# The pattern: a short prefix ending in "Page N of M" (or "Page N").
_PAGE_BAND_PREFIX = re.compile(
    r"^[^.\n]{1,220}?\bPage\s+\d+(?:\s+of\s+\d+)?\b\s*",
    re.IGNORECASE,
)


def _strip_page_band_prefix(text: str) -> str:
    """Remove a page-footer/header band prefix from the start of ``text``.

    Returns the original ``text`` unchanged when no clean band prefix
    is detectable, or the band itself looks like real content (e.g.
    contains a sentence ending before the "Page N").  Always preserves
    the substantive paragraph that follows.

    See PRODUCTION_GAPS.md P1.3.  This is the prefix-stripping
    counterpart to ``_looks_like_page_footer`` — short stand-alone
    bands get filtered entirely; embedded bands at the start of long
    paragraphs get cleaned in place.
    """
    if not text or len(text) <= 240:
        # Short atoms are either a real footer (handled by
        # _looks_like_page_footer) or short scope text we shouldn't
        # touch.
        return text
    match = _PAGE_BAND_PREFIX.match(text)
    if not match:
        return text
    prefix = match.group(0)
    # Safety: only strip when the prefix doesn't itself contain a
    # complete sentence (no period inside) and includes RFP-style
    # footer hints.
    prefix_lower = prefix.lower()
    if not any(hint in prefix_lower for hint in _PAGE_FOOTER_HINTS):
        return text
    if "." in prefix.rstrip():
        return text  # Has a sentence — don't strip.
    remainder = text[match.end():].lstrip()
    if len(remainder) < 30:
        return text  # Nothing left worth keeping.
    return remainder


# ─── PRODUCTION_GAPS P1.4: title-case fragment / bullet-noise filter ───
# Example: bullet items like "Cost Proposal", "Project Description",
# "Equipment/Service Installed" emit as standalone atoms because the
# proposal-format checklist gets exploded one-bullet-per-atom.  These
# carry no scope info — they're just labels for what the vendor's
# proposal must include.  A real scope atom either has a verb, a
# number, or names a real device/site.
_FRAGMENT_DEVICE_HINTS = (
    "camera", "controller", "panel", " ap ", "switch", "router",
    "cable", "drop", "jack", "speaker", "antenna",
    "horn", "strobe", "detector", "reader", "sensor", "monitor",
    "display", "projector", "rack", "ups", "battery",
    "fiber", "voltage", "amp", "watt", "ghz", "mhz", "mbps", "gbps",
    "psi", "bbe", "btu", "cfm",
)
# Verbs in modal/imperative form that signal scope sentences.  We use
# more specific patterns than "install" alone (which matches the
# noun "Installed" in proposal-format checklists like
# "Equipment/Service Installed").
_FRAGMENT_SENTENCE_VERBS = re.compile(
    r"\b(shall|will|must|may|should)\s+(?:provide|install|supply|furnish|"
    r"deliver|coordinate|configure|test|commission|warrant|comply|maintain|"
    r"submit|describe|confirm|include|require|offer|design|review)\b"
    r"|"
    r"\b(?:provided|installed|furnished|configured|tested|commissioned|delivered|"
    r"submitted|warranted|maintained)\s+by\s+\w+",
    re.IGNORECASE,
)


def _looks_like_fragment(text: str) -> bool:
    """Drop bullet-list-fragment-noise atoms like "Cost Proposal".

    Conservative: only drops atoms that
    - are short (≤ 45 chars),
    - have no digits or pricing,
    - have no scope-sentence verb (modal "shall provide"-type pattern),
    - have no device/contract keyword,
    - have ≤ 4 tokens,
    - and read as a noun-only label (every non-stop token starts with
      an uppercase letter).

    Real short scope atoms ("100 Mbps wireless", "Cisco Catalyst 9166I",
    "Provide all conduits") pass because they carry digits, device
    hints, or modal verbs.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 45:
        return False
    # Numbers usually indicate quantitative scope.
    if re.search(r"\d", stripped):
        return False
    text_lower = stripped.lower()
    # Modal-verb scope sentences ("shall provide ...") never look like
    # bullet-fragment labels.
    if _FRAGMENT_SENTENCE_VERBS.search(stripped):
        return False
    # Has a device / contract keyword?
    if any(h in text_lower for h in _FRAGMENT_DEVICE_HINTS):
        return False
    # Token check.  We want short noun-phrase labels, not full sentences.
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", stripped)
    if len(tokens) > 6:
        return False
    # Stop words that don't count toward "all tokens are Title-Case"
    # (so phrases like "Cost & Schedule" don't get rejected for the
    # lowercase "and").
    stop = {"of", "and", "the", "for", "to", "in", "on", "at", "or", "an", "a"}
    significant = [t for t in tokens if t.lower() not in stop]
    if not significant:
        return False
    # Bullet-list label heuristic: every significant token starts with
    # an uppercase letter (Title Case or ALL CAPS).
    if all(t[0].isupper() for t in significant) and len(tokens) <= 6:
        return True
    # Single-/two-word atoms with no info = noise.
    if len(tokens) <= 2 and len(stripped) <= 25:
        return True
    return False


class OrbitBriefPdfParser(BaseParser):
    """Parses ``.pdf`` artifacts into the OrbitBrief structured schema and EvidenceAtoms."""

    parser_name = PARSER_NAME
    parser_version = PARSER_VERSION
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".pdf"],
        supported_artifact_types=[ArtifactType.pdf],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.assumption,
            AtomType.constraint,
        ],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(
        self,
        path: Path,
        sample_text: str | None,
        domain_pack: DomainPack | None,
    ) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        reasons: list[str] = []
        confidence = 0.0
        if suffix == ".pdf":
            reasons.append("pdf_extension")
            confidence = 0.95
        # Magic-byte sniff so a PDF dropped with the wrong extension still
        # routes here.  Cheap (5 bytes) and unambiguous.
        try:
            with path.open("rb") as fh:
                head = fh.read(len(PDF_MAGIC))
            if head == PDF_MAGIC:
                reasons.append("pdf_magic_bytes")
                confidence = max(confidence, 0.90)
        except OSError:
            pass
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.pdf,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        structured_doc = build_structured_document(path)
        write_structured_doc(path, structured_doc)
        write_structured_markdown(path, structured_doc)
        atoms = list(
            atoms_from_structured_doc(
                structured_doc=structured_doc,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                parser_version=self.parser_version,
            )
        )
        # PR7 — checkbox states, NOC/SOC workflow steps, and review
        # markers for low-text visual pages. These are extracted from
        # raw PDF text in a single fitz pass; opening fitz here avoids
        # adding a second pipeline dependency.
        try:
            atoms.extend(
                _scan_pdf_for_extras(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    path=path,
                    parser_version=self.parser_version,
                )
            )
        except Exception:  # pragma: no cover — never fail the parse
            pass
        # Surface the derived artifacts in the parser output so the
        # compiler-level cache captures them and replays them on every
        # cache hit.  This guarantees ``<stem>.derived/structured.json``
        # and ``structured.md`` are always present after a compile, even
        # for cache-hot artifacts.
        derived = derived_dir_for(path)
        return ParserOutput(
            atoms=atoms,
            derived_files=[
                {
                    "relative_path": f"{derived.name}/{STRUCTURED_FILENAME}",
                    "content_kind": "json",
                    "content_json": structured_doc,
                },
                {
                    "relative_path": f"{derived.name}/{STRUCTURED_MARKDOWN_FILENAME}",
                    "content_kind": "markdown",
                    "content_text": structured_doc_to_markdown(structured_doc),
                },
            ],
        )


# ──────────────────────── public helpers ─────────────────────────────────


def build_structured_document(pdf_path: Path) -> dict[str, Any]:
    """Build the full multi-page OrbitBrief structured document for a PDF.

    Output schema is ``orbitbrief.pdf.structured.v1``:

        {
          "schema_version": "orbitbrief.pdf.structured.v1",
          "source": {"filename", "page_count"},
          "document": {"title", "metadata"},
          "pages": [
              {
                "page": int,
                "title": str | None,
                "metadata": [str, ...],
                "outline": [{"level", "heading", "block_count"}],
                "sections": [
                    {
                      "heading", "level",
                      "blocks": [
                          {"id", "kind": "paragraph", "text"},
                          {"id", "kind": "bullet_list", "intro"?, "items": [...]},
                          {"id", "kind": "table", "columns", "rows"},
                          {"id", "kind": "note", "text"},
                      ],
                      "subsections": [...]
                    },
                    ...
                ],
              },
              ...
          ],
        }
    """
    from orbitbrief_page_os.segmentation.core.config import Cfg
    from orbitbrief_page_os.segmentation.core.pipeline import build_pipeline
    from orbitbrief_page_os.segmentation.detect_standalone import _box_to_dict
    from orbitbrief_page_os.segmentation.structured_extract import extract_structured

    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - env-specific
        raise RuntimeError(
            "PyMuPDF (fitz) is required for the OrbitBrief PDF parser"
        ) from exc

    pdf_path = Path(pdf_path)
    cfg = Cfg()
    pipeline = build_pipeline()

    pages: list[dict[str, Any]] = []
    document_title: str | None = None
    document_metadata: list[str] = []
    seen_metadata: set[str] = set()

    # P2.1: pre-scan the PDF for per-page text length so we can fast-path
    # low-text pages (scanned drawings, image-only floor plans) without
    # running the heavyweight layout-detection pipeline on them.  These
    # pages contribute 0 atoms either way — we pay 5-10s/page for nothing
    # otherwise.  See PRODUCTION_GAPS.md P2.1.
    page_text_lengths: list[int] = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for page_idx in range(page_count):
            try:
                page_text = doc[page_idx].get_text("text") or ""
            except Exception:  # pragma: no cover — bad page shouldn't kill compile
                page_text = ""
            page_text_lengths.append(len(page_text.strip()))

    # Threshold: a page with <80 characters of extractable text after
    # stripping is almost certainly a scanned drawing or blank.  Real
    # Q&A / scope text exceeds this on every page we've measured.
    LOW_TEXT_PAGE_THRESHOLD = 80

    for page_index in range(page_count):
        if page_text_lengths[page_index] < LOW_TEXT_PAGE_THRESHOLD:
            # Fast path: skip the heavyweight pipeline; emit a marker
            # page so the structured doc still records the page exists
            # (with metadata noting why it was skipped).
            pages.append(
                {
                    "page": page_index,
                    "title": None,
                    "metadata": [
                        f"[low-text page (≤{LOW_TEXT_PAGE_THRESHOLD} chars) "
                        f"— likely scanned image; layout pipeline skipped for perf]"
                    ],
                    "outline": [],
                    "sections": [],
                }
            )
            continue
        state = pipeline.run(str(pdf_path), page_index=page_index, cfg=cfg)
        result = state.result
        assert result is not None, "overlay pipeline produced no result"
        payload = {
            "pdf": str(pdf_path),
            "page": page_index,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "debug_stats": result.debug_stats,
            "boxes": [_box_to_dict(b) for b in result.boxes],
        }
        struct = extract_structured(payload, pdf_path=pdf_path)
        page_doc = (struct.get("document") or {})
        page_title = page_doc.get("title")
        page_meta = list(page_doc.get("metadata") or [])

        if not document_title and page_title:
            document_title = page_title
        for entry in page_meta:
            if not entry:
                continue
            key = normalize_text(entry)
            if not key or key in seen_metadata:
                continue
            seen_metadata.add(key)
            document_metadata.append(entry)

        sections = list(struct.get("sections") or [])
        _stamp_section_and_block_ids(sections, page_index)

        pages.append(
            {
                "page": page_index,
                "title": page_title,
                "metadata": page_meta,
                "outline": list(struct.get("outline") or []),
                "sections": sections,
            }
        )

    return {
        "schema_version": STRUCTURED_SCHEMA_VERSION,
        "source": {
            "filename": pdf_path.name,
            "page_count": page_count,
        },
        "document": {
            "title": document_title,
            "metadata": document_metadata,
        },
        "pages": pages,
    }


def write_structured_doc(pdf_path: Path, structured_doc: dict[str, Any]) -> Path:
    """Persist the structured doc to ``<pdf>.derived/structured.json``."""
    derived_dir = derived_dir_for(pdf_path)
    derived_dir.mkdir(parents=True, exist_ok=True)
    out = derived_dir / STRUCTURED_FILENAME
    out.write_text(
        json.dumps(structured_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def write_structured_markdown(pdf_path: Path, structured_doc: dict[str, Any]) -> Path:
    """Persist the LLM-friendly markdown projection next to the JSON.

    The markdown mirrors the JSON structure 1:1 with stable HTML anchors
    (``<a id="blk_..."></a>`` / ``<a id="sec_..."></a>``) so an LLM can
    cite a region by anchor and a UI can scroll to the same place.
    """
    derived_dir = derived_dir_for(pdf_path)
    derived_dir.mkdir(parents=True, exist_ok=True)
    out = derived_dir / STRUCTURED_MARKDOWN_FILENAME
    out.write_text(structured_doc_to_markdown(structured_doc), encoding="utf-8")
    return out


def derived_dir_for(pdf_path: Path) -> Path:
    """Return the canonical derived-artifact directory for ``pdf_path``.

    Convention: a sibling directory named ``<stem>.derived`` next to the
    PDF.  For uploaded artifacts that lands under the project's
    ``.purtera_artifacts/<project>/<sha>.derived/``; for ad-hoc files it
    sits next to the source PDF.
    """
    pdf_path = Path(pdf_path)
    return pdf_path.with_name(f"{pdf_path.stem}{DERIVED_DIR_SUFFIX}")


def overlay_payload_and_extraction(
    pdf_path: str | Path,
    *,
    page_index: int = 0,
    overlay_dir: Path | None = None,
    file_stem: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Run overlay detection + legacy text extraction for a single page.

    Mirrors what ``detect_standalone --json-out --extraction-out`` does
    so callers and tests can compute or persist the same overlay JSON +
    text extraction artifacts (PNG, ``.overlay.json``,
    ``.extraction.json``, ``.extraction.md``).

    Returns ``(overlay_payload, extraction_doc, written_paths)``.  When
    ``overlay_dir`` is ``None`` no files are written; when supplied,
    the four artifacts above are written under ``overlay_dir`` using
    ``file_stem`` (or ``"<pdf-stem>_p{NNNN}"``) as the filename root.
    """
    from orbitbrief_page_os.segmentation.core.config import Cfg
    from orbitbrief_page_os.segmentation.core.pipeline import (
        build_pipeline,
        render_overlay,
    )
    from orbitbrief_page_os.segmentation.detect_standalone import _box_to_dict
    from orbitbrief_page_os.segmentation.extract_overlay_text import (
        extract_from_overlay_json,
        write_extraction_artifacts,
    )

    pdf_path = Path(pdf_path).resolve()
    cfg = Cfg()
    pipeline = build_pipeline()
    state = pipeline.run(str(pdf_path), page_index=page_index, cfg=cfg)
    result = state.result
    rgb = state.rgb
    assert result is not None and rgb is not None, "overlay pipeline produced no result"

    payload = {
        "pdf": str(pdf_path),
        "page": page_index,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "debug_stats": result.debug_stats,
        "boxes": [_box_to_dict(b) for b in result.boxes],
    }
    doc = extract_from_overlay_json(payload, pdf_path=pdf_path)

    written: list[Path] = []
    if overlay_dir is not None:
        overlay_dir = Path(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or f"{pdf_path.stem}_p{page_index:04d}"
        png = overlay_dir / f"{stem}.png"
        ov_js = overlay_dir / f"{stem}.overlay.json"
        # write_extraction_artifacts always appends ``.extraction.json`` /
        # ``.extraction.md`` to its base path, so pass the bare stem to
        # avoid producing ``stem.extraction.extraction.json``.
        ex_base = overlay_dir / stem
        render_overlay(rgb, result, png, draw_labels=False)
        ov_js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ex_paths = write_extraction_artifacts(ex_base, doc)
        written = [png.resolve(), ov_js.resolve()] + sorted(
            Path(p).resolve() for p in ex_paths.values()
        )

    return payload, doc, written


# ──────────────────────── atom emission ──────────────────────────────────


def atoms_from_structured_doc(
    *,
    structured_doc: dict[str, Any],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    """Stream EvidenceAtoms from a structured document.

    One atom per content block (paragraph, bullet item, table row, or
    note).  Headings are not atoms — they become locator context
    (``section_path``) on the atoms beneath them so OrbitBrief can
    re-classify or re-aggregate without re-parsing.
    """
    for page in structured_doc.get("pages", []):
        page_index = int(page.get("page", 0))
        yield from _atoms_for_sections(
            sections=page.get("sections", []) or [],
            section_path=[],
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        )


def _atoms_for_sections(
    *,
    sections: Iterable[dict[str, Any]],
    section_path: list[str],
    page_index: int,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    for section in sections:
        heading = section.get("heading")
        path = section_path + ([heading] if heading else [])
        for block in section.get("blocks", []) or []:
            yield from _atoms_for_block(
                block=block,
                section_path=path,
                page_index=page_index,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
            )
        yield from _atoms_for_sections(
            sections=section.get("subsections", []) or [],
            section_path=path,
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        )


def _atoms_for_block(
    *,
    block: dict[str, Any],
    section_path: list[str],
    page_index: int,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    kind = block.get("kind")
    block_id = block.get("id") or stable_id("blk", page_index, kind or "?", id(block))
    base_locator: dict[str, Any] = {
        "page": page_index,
        "block_id": block_id,
        "block_kind": kind,
        "section_path": section_path,
    }

    if kind == "paragraph":
        text = (block.get("text") or "").strip()
        if not text:
            return
        # P1.2: skip vendor-info form-field templates entirely — they
        # carry no scope content and pollute downstream anchors.
        if _looks_like_form_field(text):
            return
        # P1.3: skip page-footer / page-header band text (e.g. "RFP
        # 25-107 Wireless Equipment ... Page 17 of 25").  These appear
        # once per page and bloat the atom set N-fold for an N-page PDF.
        if _looks_like_page_footer(text):
            return
        # P1.3 (band-prefix variant): when PDF extraction folded the
        # header/footer band into the *start* of a real paragraph,
        # strip the band rather than drop the atom.
        text = _strip_page_band_prefix(text)
        if not text or len(text) < 10:
            return
        # P1.4: skip pure-title-case bullet-fragment labels like "Cost
        # Proposal", "Project Description", "Addendums".  These come
        # from proposal-format checklists and carry no scope data.
        if _looks_like_fragment(text):
            return
        # P1.1: when a paragraph contains ≥2 Q\d. / A\d. markers, split
        # it into one atom per Q&A pair so packet anchors don't end up
        # as 2,400-char transcripts.  Single-Q paragraphs and
        # paragraphs without Q&A markers fall through to the original
        # single-atom path below.
        qa_chunks = _split_qa_blob(text)
        if len(qa_chunks) >= 2:
            for chunk_idx, chunk in enumerate(qa_chunks):
                atom_type, authority = _classify_text_block(
                    text=chunk, section_path=section_path, kind="paragraph"
                )
                yield _make_atom(
                    text=chunk,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    parser_version=parser_version,
                    atom_type=atom_type,
                    authority_class=authority,
                    confidence=DEFAULT_BLOCK_CONFIDENCE,
                    locator={
                        **base_locator,
                        "qa_chunk_index": chunk_idx,
                        "qa_chunk_count": len(qa_chunks),
                    },
                    value={"kind": "paragraph", "qa_split": True},
                )
            return
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="paragraph")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_BLOCK_CONFIDENCE,
            locator=base_locator,
            value={"kind": "paragraph"},
        )
        return

    if kind == "bullet_list":
        intro = (block.get("intro") or "").strip()
        if intro:
            intro_type, intro_auth = _classify_text_block(
                text=intro, section_path=section_path, kind="bullet_intro"
            )
            yield _make_atom(
                text=intro,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=intro_type,
                authority_class=intro_auth,
                confidence=DEFAULT_BLOCK_CONFIDENCE,
                locator={**base_locator, "bullet_role": "intro"},
                value={"kind": "bullet_intro"},
            )
        # Bullets inherit the intro's classification context; for many docs
        # the intro line ("Partner(s) must:") is what colors every child.
        bullet_section_path = section_path + ([intro] if intro else [])
        for index, item in enumerate(block.get("items", []) or []):
            yield from _atoms_for_bullet(
                item=item,
                depth=1,
                path_indices=[index],
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                base_locator=base_locator,
                section_path=bullet_section_path,
            )
        return

    if kind == "table":
        columns = list(block.get("columns") or [])
        rows = list(block.get("rows") or [])
        sample_cells: list[str] = []
        for row in rows[:5]:
            if isinstance(row, dict):
                for value in row.values():
                    if value is None:
                        continue
                    s = str(value).strip()
                    if s:
                        sample_cells.append(s)
        atom_type, authority = _classify_table(
            section_path=section_path,
            columns=columns,
            sample_cells=sample_cells,
        )
        for row_index, row in enumerate(rows):
            row_text = _row_to_text(row)
            if not row_text:
                continue
            # P1.2: skip table rows that are obviously vendor-info form
            # templates (the VT-CAM "FULL LEGAL NAME (PRINT) ... |
            # CONTACT NAME/TITLE | FEDERAL TAXPAYER NUMBER (ID#)"
            # rows).  A table row is a form template when its text
            # would qualify as one if it appeared as a paragraph.
            if _looks_like_form_field(row_text):
                continue
            # P1.3: skip table rows that are repeated page-footer
            # bands (some PDF extractors fold multi-line footers into
            # a single-row table).
            if _looks_like_page_footer(row_text):
                continue
            # P1.7: skip fused multi-row cells where the "column name"
            # is actually data from a previous row (e.g. "AIR-DNA-E:
            # AIR-DNA-E-T-5Y | ... | 500: 500").  These produce noise
            # part_number entities and confuse the quantity_conflict rule.
            if isinstance(row, dict) and _looks_like_fused_table_row(row):
                continue
            yield _make_atom(
                text=row_text,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=atom_type,
                authority_class=authority,
                confidence=TABLE_ROW_CONFIDENCE,
                locator={**base_locator, "row_index": row_index},
                value={
                    "kind": "table_row",
                    "columns": columns,
                    "cells": dict(row),
                },
            )
        return

    if kind == "note":
        text = (block.get("text") or "").strip()
        if not text:
            return
        text = _strip_page_band_prefix(text)
        if not text or len(text) < 10:
            return
        # P1.3 / P1.2 / P1.4: notes also catch page-footer text and
        # form-field templates on some layouts; same filters as paragraph.
        if _looks_like_form_field(text) or _looks_like_page_footer(text) or _looks_like_fragment(text):
            return
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="note")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_NOTE_CONFIDENCE,
            locator=base_locator,
            value={"kind": "note"},
        )
        return


def _atoms_for_bullet(
    *,
    item: dict[str, Any],
    depth: int,
    path_indices: list[int],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    base_locator: dict[str, Any],
    section_path: list[str],
) -> Iterator[EvidenceAtom]:
    text = (item.get("text") or "").strip()
    if text:
        # Strip page-band prefix that some extractors fold into bullet text.
        text = _strip_page_band_prefix(text)
    if text and len(text) >= 10 and not _looks_like_form_field(text) and not _looks_like_page_footer(text) and not _looks_like_fragment(text):
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="bullet")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_BLOCK_CONFIDENCE,
            locator={
                **base_locator,
                "bullet_path": list(path_indices),
                "bullet_depth": depth,
            },
            value={"kind": "bullet", "depth": depth},
        )
    for child_index, child in enumerate(item.get("children", []) or []):
        yield from _atoms_for_bullet(
            item=child,
            depth=depth + 1,
            path_indices=path_indices + [child_index],
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            base_locator=base_locator,
            section_path=section_path,
        )


def _row_to_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for col, val in row.items():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        parts.append(f"{col}: {s}")
    return " | ".join(parts)


# ─── PRODUCTION_GAPS P1.7: fused table-row detection ───
# When OrbitBrief PDF table extraction confuses a 2-row vertical fold
# for a 1-row horizontal fold, we get atoms like:
#   "AIR-DNA-E: AIR-DNA-E-T-5Y | Wireless Cisco DNA On-Prem Essential,
#    Term Lic: Wireless Cisco DNA On-Prem Essential, 5Y Term, ... | 500: 500"
# The "column names" are actually data values from a previous row.
# Detection signals (any one is sufficient):
#   1. ≥2 columns whose name == value (e.g. ``500: 500``).
#   2. ≥2 columns whose name looks like a SKU (uppercase + digits,
#      length 3-30, with ``-`` or ``_``).
#   3. ≥1 column whose name is a multi-word phrase containing
#      vendor/product keywords ("Cisco DNA"-type strings).
_SKU_SHAPED_COLUMN = re.compile(r"^[A-Z][A-Z0-9_]{1,8}(?:[-/][A-Z0-9_]{1,12}){1,4}$")
_DATA_SHAPED_HEADER_PHRASES = (
    "wireless cisco dna",
    "ceiling grid clip",
    "low profile mounting",
    "universal mounting bracket",
    "single pack option",
    "dna on-prem",
    "dna on prem",
    "perpetual network stack",
    "essentials",
)


def _looks_like_fused_table_row(row: dict[str, Any]) -> bool:
    """Detect rows where the "column name" was actually data from a
    previous row in the source PDF.

    Returns True iff at least 2 strong signals fire (so a single
    coincidence — e.g. an actual column literally named "500" with
    value "500" — doesn't trigger).  The caller can drop or downgrade
    such atoms to keep them from polluting entity_keys.
    """
    if not row:
        return False
    same_value_cells = 0
    sku_columns = 0
    data_phrase_columns = 0
    for col, val in row.items():
        if val is None:
            continue
        col_str = str(col).strip()
        val_str = str(val).strip()
        if not col_str or not val_str:
            continue
        # Signal 1: col == val (e.g. "500: 500"). One match is rare in
        # legitimate tables (a column literally named "500" with value
        # "500" would be an extreme oddity), so any single hit counts.
        if col_str == val_str and re.search(r"[A-Z0-9]", col_str):
            same_value_cells += 1
        # Signal 2: column name looks like a SKU.
        if _SKU_SHAPED_COLUMN.match(col_str):
            sku_columns += 1
        # Signal 3: column name is a long Cisco DNA / vendor phrase.
        col_lower = col_str.lower()
        if len(col_str) > 25 and any(p in col_lower for p in _DATA_SHAPED_HEADER_PHRASES):
            data_phrase_columns += 1
    signals = (
        (1 if same_value_cells >= 1 else 0)
        + (1 if sku_columns >= 1 else 0)
        + (1 if data_phrase_columns >= 1 else 0)
    )
    # Two independent signals → confidently fused.  One signal alone is
    # ambiguous (could be a real table that happens to have a SKU
    # column heading or a "500: 500" coincidence).
    return signals >= 2


def _make_atom(
    *,
    text: str,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    atom_type: AtomType,
    authority_class: AuthorityClass,
    confidence: float,
    locator: dict[str, Any],
    value: dict[str, Any],
) -> EvidenceAtom:
    src_id = stable_id(
        "src",
        artifact_id,
        locator.get("page"),
        locator.get("block_id"),
        locator.get("bullet_path"),
        locator.get("row_index"),
    )
    source_ref = SourceRef(
        id=src_id,
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator=dict(locator),
        extraction_method=EXTRACTION_METHOD,
        parser_version=parser_version,
    )
    atom_id = stable_id(
        "atm",
        project_id,
        artifact_id,
        atom_type.value,
        text,
        locator.get("page"),
        locator.get("block_id"),
        locator.get("bullet_path"),
        locator.get("row_index"),
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=normalize_text(text),
        value=value,
        entity_keys=[],
        source_refs=[source_ref],
        receipts=[],
        authority_class=authority_class,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version=parser_version,
    )


# ──────────────────────── classification ─────────────────────────────────

# Each rule is a (regex, AtomType, AuthorityClass) tuple.  First match wins.
# Section-path matching is done case-insensitively against the joined path.
_SECTION_RULES: list[tuple[re.Pattern[str], AtomType, AuthorityClass]] = [
    (
        re.compile(r"\b(out\s*of\s*scope|exclusion(s)?|excluded|not\s+included)\b", re.I),
        AtomType.exclusion,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\bassumption(s)?\b", re.I),
        AtomType.assumption,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(constraint(s)?|requirement(s)?|prerequisite(s)?)\b", re.I),
        AtomType.constraint,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(open\s+question(s)?|tbd|to\s+be\s+determined|outstanding)\b", re.I),
        AtomType.open_question,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(decision(s)?|approved|approval(s)?)\b", re.I),
        AtomType.decision,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(
            r"\b(action\s*item(s)?|task(s)?|deliverable(s)?|to[-\s]?do(s)?|next\s+steps?)\b",
            re.I,
        ),
        AtomType.action_item,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(pricing|price|cost|quote(d)?|line\s+items?|sow\s+pricing|fees?)\b", re.I),
        AtomType.vendor_line_item,
        AuthorityClass.vendor_quote,
    ),
    (
        re.compile(
            r"\b(customer\s+(instruction|request|requirement)|client\s+(instruction|request|requirement))\b",
            re.I,
        ),
        AtomType.customer_instruction,
        AuthorityClass.customer_current_authored,
    ),
    (
        re.compile(
            r"\b(scope\s+of\s+work|sow|scope|kitting\s+requirement|partner\s+requirement|operational\s+expectation)\b",
            re.I,
        ),
        AtomType.scope_item,
        AuthorityClass.contractual_scope,
    ),
]

# Block-text overrides — applied after section rules, only when the text
# itself is unambiguous (modal verbs / question marks).  These let a
# constraint sentence in a "scope" section still be tagged as a constraint.
#
# Order matters within a list: more-specific patterns first.  Each rule
# returns its own AtomType regardless of section context.
_TEXT_OVERRIDES: list[tuple[re.Pattern[str], AtomType]] = [
    # Open question shapes — vendor-asked clarification (Q\d., trailing ?)
    (re.compile(r"^\s*Q\s*\d+\.\s"), AtomType.open_question),
    (re.compile(r"\?\s*$"), AtomType.open_question),
    # Strong exclusion shapes that the original list missed.  The
    # VT-CAM addendum carries many of these ("would not be needed",
    # "no plans for", "not at this time", "is not currently") that
    # used to default to scope_item.  See PRODUCTION_GAPS / Week 5.
    (re.compile(r"\b(would\s+not\s+be\s+(?:needed|needing|required|requiring)|likely\s+not\s+be\s+needed)\b", re.I), AtomType.exclusion),
    (re.compile(r"\b(no\s+plans?\s+for|not\s+at\s+this\s+time|not\s+currently|is\s+not\s+currently|do\s+not\s+(?:plan|intend|expect)\s+to)\b", re.I), AtomType.exclusion),
    (re.compile(r"\b(not\s+a\s+part\s+of|not\s+included|not\s+in\s+scope|out\s+of\s+scope)\b", re.I), AtomType.exclusion),
    (re.compile(r"\b(by\s+(?:others|gc|owner|customer|vendor)|n\.?i\.?c\.?|provided\s+by\s+(?:others|owner))\b", re.I), AtomType.exclusion),
    (re.compile(r"^\s*(do not|may not|cannot|must not|shall not|will not)\b", re.I), AtomType.exclusion),
    # ─── Compliance clauses (Week 6 P6.1) ───
    # These cite an external standard / code / regulation and live as a
    # separate atom_type so OrbitBrief can render a "Compliance" tab.
    # Order matters: compliance patterns fire BEFORE generic constraint
    # patterns so "must comply with NFPA 72" isn't first matched as a
    # constraint.
    #
    # "comply with X" / "in accordance with X" / "per X" / "X-compliant"
    # — the X must look like a standard (ALLCAPS acronym, "Section X",
    # numbered code reference) so a bare "comply with the project
    # schedule" doesn't get pulled in.
    (
        re.compile(
            r"\b(?:must\s+comply\s+with|shall\s+comply\s+with|complies?\s+with|compliant\s+with|in\s+(?:full\s+)?accordance\s+with|in\s+conformance\s+with|conforms?\s+to|per\s+the\s+requirements\s+of|as\s+required\s+by)\s+"
            r"(?:[A-Z]{2,8}(?:\s*\d|\s+[A-Z][a-z])|"
            r"(?:national|international|federal|state)\s+\w+|"
            r"section\s+\d+|"
            r"(?:nfpa|ieee|ada|osha|nec|ul|csi|iso|en|tia|eia|fcc|niem|fips|hipaa|gdpr|sox|ccpa|sox|pci|nist|fips)\b)",
            re.I,
        ),
        AtomType.compliance,
    ),
    # Trailing-form: "X-compliant" / "X-listed" / "X-rated" / "X-approved"
    (
        re.compile(
            r"\b(?:UL|ETL|FCC|CE|RoHS|ADA|FIPS|NIST|HIPAA|PCI|SOX|GDPR|CCPA|NDAA|TAA)\s*[-–]?\s*(?:listed|certified|compliant|approved|rated|tested)\b",
            re.I,
        ),
        AtomType.compliance,
    ),
    # Code-cite shapes: "per NFPA 72", "per NEC 250.122",
    # "per IEEE 802.3bt", "per Section 27 32 26".
    (
        re.compile(
            r"\b(?:per|under|pursuant\s+to|in\s+accordance\s+with)\s+"
            r"(?:nfpa|ieee|ada|osha|nec|nfpa\d+|ul\d+|csi|iso|en\s*\d|tia|eia|fcc|fips|hipaa|nist|niem)\b",
            re.I,
        ),
        AtomType.compliance,
    ),
    # E-rate / federal-grant compliance (Universal Service Fund, Schools
    # and Libraries, Section 508, ANSI/TIA, …).
    (
        re.compile(
            r"\b(?:e-?rate(?:\s+eligible|\s+eligibility|\s+compliance|\s+funded)?|usf\s+eligible|section\s+508\s+compliant|secure\s+networks\s+act|davis[-–\s]bacon|buy\s+america(?:n)?\s+act|taa\s+compliant|ndaa\s+compliant)\b",
            re.I,
        ),
        AtomType.compliance,
    ),
    # Constraint shapes — modal verbs at the start of a clause.
    (re.compile(r"^\s*(must|shall|required to|is required to|will be required to)\b", re.I), AtomType.constraint),
    (re.compile(r"\b(must\s+(?:comply|conform|support|meet|include)|is\s+required|shall\s+comply)\b", re.I), AtomType.constraint),
    # Decision shapes — "will be", "is to be", "centralized at",
    # "decided to", "approved to".  These are the meeting-decision
    # cues that used to fall through to scope_item.
    (re.compile(r"\b(centralized\s+at|will\s+be\s+(?:provided|managed|operated|housed|located)\s+(?:by|at))\b", re.I), AtomType.decision),
    (re.compile(r"\b(decid(?:ed|ing)\s+to|approved\s+to|approved\s+for|is\s+to\s+be)\b", re.I), AtomType.decision),
    (re.compile(r"\b((?:we|the\s+(?:university|district|college|customer|client|owner))\s+will\s+(?:not\s+)?(?:provide|manage|use|select|host|run|own))\b", re.I), AtomType.decision),
    # Action item shapes — vendor-or-owner commitments.
    (re.compile(r"\b(vendor\s+(?:must|shall|will|is\s+required\s+to)\s+(?:describe|provide|submit|deliver|coordinate|confirm|train|certify))\b", re.I), AtomType.action_item),
    (re.compile(r"\b((?:successful|awarded)\s+(?:offeror|bidder|respondent|firm|contractor)\s+(?:must|shall|will))\b", re.I), AtomType.action_item),
    (re.compile(r"\b(to\s+(?:identify\s+priorit|provide\s+letter|submit\s+the|register\s+with))\b", re.I), AtomType.action_item),
    # Assumption shapes.
    (re.compile(r"^\s*(assume(s|d)?|assuming)\b", re.I), AtomType.assumption),
]


# Authority-class overrides.  Atoms whose text matches one of these
# patterns are flagged as ``customer_current_authored`` so the
# packetizer's customer_override rule can fire.
#
# Three pattern families:
#  1. PRODUCTION_GAPS / Week 5 — Q&A answer markers ("A12.", "A47.").
#     These appear in pre-proposal-conference transcripts where the
#     customer's blue-text answer is the authoritative source.
#  2. Week 6 P6.3 — explicit customer/owner attribution ("Owner-furnished",
#     "Owner Preferred:", "Customer Notes:", "Owner shall provide").
#     These show up in addenda, customer overlays, and owner-side
#     mark-ups.
#  3. Week 6 P6.3 — first-person customer voice ("VT will manage",
#     "the District has selected", "we have decided").  When the
#     customer is the speaker, the atom is customer-authored.  Tight
#     enough to avoid catching every "we" pronoun in vendor-authored
#     text; requires a customer/owner subject + commitment verb.
_AUTHORITY_OVERRIDES: list[tuple[re.Pattern[str], AuthorityClass]] = [
    # 1) Q&A answer markers
    (re.compile(r"^\s*A\s*\d+\.\s"), AuthorityClass.customer_current_authored),
    (re.compile(r"\bA\s*\d+\.\s"), AuthorityClass.customer_current_authored),
    # 2) Explicit owner/customer attribution.  Allows possessive
    # ("Owner's Notes:") and bare-noun-phrase ("Customer Notes:") forms.
    (
        re.compile(
            r"\b(?:owner[-\s]?(?:furnished|preferred|provided|approved|directed)|owner\s+shall|owner\s+will|"
            r"owner(?:['’]s)?\s+(?:notes?|comments?|requirements?|preferences?|direction)|"
            r"customer[-\s]?(?:furnished|preferred|provided|approved|directed)|customer\s+(?:shall|will|requires|prefers)|"
            r"customer(?:['’]s)?\s+(?:notes?|comments?|requirements?|preferences?|direction|response))\b",
            re.I,
        ),
        AuthorityClass.customer_current_authored,
    ),
    # 3) Customer-side first-person commitment / decision
    (
        re.compile(
            r"\b(?:the\s+(?:university|district|college|school|agency|customer|client|owner|board|department|hospital|authority|county|city)\s+"
            r"(?:will|has|have|shall|does|does\s+not|do|do\s+not|requires|prefers|selected|approved|decided|provided|manages))\b",
            re.I,
        ),
        AuthorityClass.customer_current_authored,
    ),
    # 4) Addendum / customer-response markup ("RESPONSE:", "CUSTOMER:",
    #    "ANSWER:" headers used in column-style RFP responses).
    (
        re.compile(
            r"^\s*(?:RESPONSE|ANSWER|CUSTOMER\s+RESPONSE|OWNER\s+RESPONSE|DISTRICT\s+RESPONSE|UNIVERSITY\s+RESPONSE)\s*:",
            re.I,
        ),
        AuthorityClass.customer_current_authored,
    ),
]

# Tight column-header regex: only fires on unambiguously-pricing words.
# We deliberately do NOT match bare "unit" / "total" / "amount" because
# those appear in scope-checklist tables ("Unit AP Installation",
# "Total Devices") that have nothing to do with pricing.  When in doubt,
# fall back to the section classifier.
_PRICING_COLUMN_HINTS = re.compile(
    r"\b("
    r"unit\s+(price|cost|rate)"
    r"|line\s+item(s)?"
    r"|extended\s+(price|cost|amount)"
    r"|subtotal"
    r"|hourly\s+rate"
    r"|price"
    r"|cost"
    r"|invoice"
    r"|fee(s)?"
    r"|rate\s+card"
    r")\b",
    re.I,
)
_CURRENCY_PATTERN = re.compile(r"(?:\$|£|€|usd|gbp|eur)\s*\d", re.I)


# Splits a coalesced "Q4. ...? A4. ..." chunk into (question_part,
# answer_part).  When the chunk has an A-marker we want to classify
# atom_type from the *answer* body — that's the substantive customer
# content; the question is a contractual-scope template line.
_QA_ANSWER_SPLIT = re.compile(r"\bA\s*\d+\.\s")


def _split_question_and_answer(text: str) -> tuple[str, str]:
    """Return ``(question_part, answer_part)``.

    If no A-marker is found, ``answer_part`` is empty and the original
    text is returned in ``question_part``.  When the marker IS present
    the question is everything up to (and including) the marker, and
    the answer is everything after.
    """
    if not text:
        return "", ""
    match = _QA_ANSWER_SPLIT.search(text)
    if not match:
        return text, ""
    return text[: match.start()].strip(), text[match.end() :].strip()


_PROMOTABLE_ATOMS_FROM_QA: frozenset[AtomType] = frozenset(
    {AtomType.scope_item, AtomType.open_question}
)


def _classify_text_block(
    *,
    text: str,
    section_path: list[str],
    kind: str,
) -> tuple[AtomType, AuthorityClass]:
    """Pick (AtomType, AuthorityClass) from section context + the block text.

    ``kind`` is the structural kind (``paragraph`` / ``bullet`` / ``note``).
    Notes always default to ``assumption / meeting_note`` unless the
    section path screams something different (e.g. a red callout under a
    pricing section is still a vendor signal, not a meeting note).
    """
    section_blob = " ".join(section_path or [])

    section_atom: AtomType | None = None
    section_auth: AuthorityClass | None = None
    for pattern, atom_type, auth in _SECTION_RULES:
        if pattern.search(section_blob):
            section_atom = atom_type
            section_auth = auth
            break

    # Week 5: when the chunk is a coalesced Q+A pair (Q4. ... A4. ...),
    # the *answer* body carries the customer's substantive position, so
    # classify atom_type from the answer body and only fall back to the
    # full text if the answer body doesn't yield a definite signal.  This
    # is what lets "A43. The lighting plan is attached." classify as a
    # decision rather than the open_question its leading "Q43." would
    # have implied.
    _question_part, answer_part = _split_question_and_answer(text)
    classify_text = answer_part if answer_part and len(answer_part) >= 10 else text

    text_atom: AtomType | None = None
    for pattern, atom_type in _TEXT_OVERRIDES:
        if pattern.search(classify_text):
            text_atom = atom_type
            break
    # If we tried the answer body and got nothing, retry against the full
    # text so the original Q-marker-only / "?-suffix" signals can still
    # fire (e.g. a pure question with no useful answer body).
    if text_atom is None and classify_text is not text:
        for pattern, atom_type in _TEXT_OVERRIDES:
            if pattern.search(text):
                text_atom = atom_type
                break

    # Authority override (Week 5).  Q&A answer markers ("A12.") signal
    # customer-authored content.  When an atom carries an answer
    # *and* its content reads as an instruction, promote the atom_type
    # to customer_instruction so the packetizer's customer_override
    # rule can fire.
    text_authority: AuthorityClass | None = None
    for pattern, authority in _AUTHORITY_OVERRIDES:
        if pattern.search(text):
            text_authority = authority
            break

    if kind == "note":
        if section_atom is not None:
            # A red callout under a typed section keeps the section's authority
            # but stays an assumption (it's a callout, not a primary clause).
            return AtomType.assumption, section_auth or AuthorityClass.meeting_note
        return AtomType.assumption, AuthorityClass.meeting_note

    if text_atom is not None:
        # Text override wins over section default for definite signals.
        authority = text_authority or section_auth or AuthorityClass.contractual_scope
        # When the atom is customer-authored AND it reads like a scope
        # statement / open question (default), promote it to
        # customer_instruction so the packetizer's customer_override
        # rule can fire.  Decisions / action_items / exclusions /
        # constraints surface as themselves — those are STRONGER signals
        # than customer_instruction and the packetizer wants them
        # un-merged for meeting_decision / action_item / scope_exclusion
        # families.
        if (
            text_authority == AuthorityClass.customer_current_authored
            and text_atom in _PROMOTABLE_ATOMS_FROM_QA
        ):
            return AtomType.customer_instruction, authority
        return text_atom, authority

    if section_atom is not None:
        authority = text_authority or section_auth or AuthorityClass.contractual_scope
        if (
            text_authority == AuthorityClass.customer_current_authored
            and section_atom in _PROMOTABLE_ATOMS_FROM_QA
        ):
            return AtomType.customer_instruction, authority
        return section_atom, authority

    # Default: a customer-authored answer is a customer_instruction;
    # everything else is a scope_item.
    if text_authority == AuthorityClass.customer_current_authored:
        return AtomType.customer_instruction, text_authority
    return AtomType.scope_item, AuthorityClass.contractual_scope


def _classify_table(
    *,
    section_path: list[str],
    columns: list[str],
    sample_cells: list[str] | None = None,
) -> tuple[AtomType, AuthorityClass]:
    """Pick (AtomType, AuthorityClass) for every row in a table.

    Strategy (in priority order):
      1. Section path screams a non-default role (exclusion / pricing /
         decision / etc) → use that.
      2. Column headers contain unambiguous pricing words ("unit price",
         "subtotal", "fee", etc.) → vendor_line_item / vendor_quote.
      3. Sample cell values contain currency markers ($, £, €, USD …) →
         vendor_line_item / vendor_quote.
      4. Default to scope_item / contractual_scope.

    We deliberately put the section classifier FIRST so a "SCOPE OF
    WORK" matrix doesn't get hijacked by a column whose name happens to
    contain the substring "unit".
    """
    section_blob = " ".join(section_path or [])
    for pattern, atom_type, auth in _SECTION_RULES:
        if pattern.search(section_blob):
            return atom_type, auth

    column_blob = " ".join(str(c) for c in columns)
    if _PRICING_COLUMN_HINTS.search(column_blob):
        return AtomType.vendor_line_item, AuthorityClass.vendor_quote

    if sample_cells:
        joined = " ".join(sample_cells[:20])
        if _CURRENCY_PATTERN.search(joined):
            return AtomType.vendor_line_item, AuthorityClass.vendor_quote

    return AtomType.scope_item, AuthorityClass.contractual_scope


# ──────────────────────── markdown projection ────────────────────────────


def structured_doc_to_markdown(structured_doc: dict[str, Any]) -> str:
    """Render the structured doc as LLM-friendly markdown with anchors.

    Output shape::

        ---
        schema: orbitbrief.pdf.structured.v1
        filename: <name>
        page_count: N
        ---

        # <document title>

        > **Metadata**
        > - line 1
        > - line 2

        <!-- page 0 -->

        ## <section>  <a id="sec_..."></a>

        <a id="blk_..."></a>
        body paragraph text

        <a id="blk_..."></a>
        **Intro:** intro line for bullet list
        - bullet
          - sub-bullet

        <a id="blk_..."></a>
        | col a | col b |
        |-------|-------|
        | v1    | v2    |

        > **Note:** note text  <a id="blk_..."></a>
    """
    lines: list[str] = []
    source = structured_doc.get("source") or {}
    document = structured_doc.get("document") or {}

    lines.append("---")
    lines.append(f"schema: {structured_doc.get('schema_version', STRUCTURED_SCHEMA_VERSION)}")
    if source.get("filename"):
        lines.append(f"filename: {source['filename']}")
    if source.get("page_count") is not None:
        lines.append(f"page_count: {source['page_count']}")
    lines.append("---")
    lines.append("")

    title = document.get("title")
    if title:
        lines.append(f"# {title}")
        lines.append("")

    metadata = document.get("metadata") or []
    if metadata:
        lines.append("> **Metadata**")
        for entry in metadata:
            lines.append(f"> - {entry}")
        lines.append("")

    for page in structured_doc.get("pages", []) or []:
        page_index = page.get("page", 0)
        lines.append(f"<!-- page {page_index} -->")
        lines.append("")
        page_meta = [m for m in (page.get("metadata") or []) if m and m not in metadata]
        if page_meta:
            for entry in page_meta:
                lines.append(f"_{entry}_")
            lines.append("")
        for section in page.get("sections", []) or []:
            _render_section_md(lines, section, depth=2)

    text = "\n".join(lines).rstrip() + "\n"
    return text


def _render_section_md(lines: list[str], section: dict[str, Any], *, depth: int) -> None:
    heading = (section.get("heading") or "").strip()
    section_id = section.get("id")
    if heading:
        prefix = "#" * min(max(depth, 1), 6)
        anchor = f'  <a id="{section_id}"></a>' if section_id else ""
        lines.append(f"{prefix} {heading}{anchor}")
        lines.append("")

    for block in section.get("blocks", []) or []:
        _render_block_md(lines, block)

    for child in section.get("subsections", []) or []:
        _render_section_md(lines, child, depth=depth + 1)


def _render_block_md(lines: list[str], block: dict[str, Any]) -> None:
    kind = block.get("kind")
    block_id = block.get("id")
    anchor = f'<a id="{block_id}"></a>' if block_id else ""

    if kind == "paragraph":
        text = (block.get("text") or "").strip()
        if not text:
            return
        if anchor:
            lines.append(anchor)
        lines.append(text)
        lines.append("")
        return

    if kind == "bullet_list":
        if anchor:
            lines.append(anchor)
        intro = (block.get("intro") or "").strip()
        if intro:
            lines.append(f"**Intro:** {intro}")
        for item in block.get("items", []) or []:
            _render_bullet_md(lines, item, depth=0)
        lines.append("")
        return

    if kind == "table":
        if anchor:
            lines.append(anchor)
        columns = list(block.get("columns") or [])
        rows = list(block.get("rows") or [])
        if not columns and rows:
            # Synthesize column names from the first row's keys to keep
            # markdown valid.
            columns = list(rows[0].keys())
        if not columns:
            raw = (block.get("raw_text") or "").strip()
            if raw:
                lines.append(raw)
                lines.append("")
            return
        lines.append("| " + " | ".join(_md_cell(c) for c in columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(_md_cell(row.get(col, "")) for col in columns)
                + " |"
            )
        lines.append("")
        return

    if kind == "note":
        text = (block.get("text") or "").strip()
        if not text:
            return
        suffix = f"  {anchor}" if anchor else ""
        lines.append(f"> **Note:** {text}{suffix}")
        lines.append("")
        return


def _render_bullet_md(lines: list[str], item: dict[str, Any], *, depth: int) -> None:
    text = (item.get("text") or "").strip()
    indent = "  " * depth
    if text:
        lines.append(f"{indent}- {text}")
    for child in item.get("children", []) or []:
        _render_bullet_md(lines, child, depth=depth + 1)


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Escape pipes so we never break the markdown table.
    return text.replace("|", "\\|").replace("\n", " ")


# ──────────────────────── internals ──────────────────────────────────────


def _stamp_section_and_block_ids(sections: list[dict[str, Any]], page_index: int) -> None:
    """Stamp every section and block in ``sections`` with a stable ``id``.

    IDs are deterministic strings derived from page + walk counter + kind
    (``sec_<digest>`` for sections, ``blk_<digest>`` for blocks), so a
    re-run on the same PDF produces the same ids without depending on
    object identity.
    """
    section_counter = [0]
    block_counter = [0]

    def visit(nodes: list[dict[str, Any]]) -> None:
        for section in nodes:
            section["id"] = stable_id(
                "sec", page_index, section_counter[0], section.get("level") or 1
            )
            section_counter[0] += 1
            for block in section.get("blocks", []) or []:
                block["id"] = stable_id(
                    "blk", page_index, block_counter[0], block.get("kind") or "?"
                )
                block_counter[0] += 1
            visit(section.get("subsections", []) or [])

    visit(sections)


# ─────────────────── PR7: checkbox / workflow / visual-page atoms ────


_CHECKBOX_RE = re.compile(
    r"(?P<mark>☒|☑|✓|✔|\[x\]|\[X\]|\(x\)|\(X\)|☐|□|\[\s\]|\(\s\))"
    r"\s*(?P<label>[^|;\n]+)"
)
_WORKFLOW_STEP_RE = re.compile(
    r"\b(detect|triage|contain|escalate|recover|remediate|notify|"
    r"dispatch|close|improve)\b",
    re.I,
)
_LOW_TEXT_VISUAL_THRESHOLD = 80


def _checkbox_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Extract checked / unchecked checkbox state from page text.

    Checked boxes (☒, ☑, ✓, ✔, [x], (X)) emit a ``scope_item`` atom
    with ``value.checked=true``. Unchecked boxes (☐, □, [ ], ( ))
    emit an ``exclusion`` atom with ``value.checked=false`` and the
    review flag ``unchecked_checkbox_not_scope`` so the calibrator
    flags it for human review — unchecked is *evidence of exclusion*,
    not silent absence.
    """
    atoms: list[EvidenceAtom] = []
    for idx, m in enumerate(_CHECKBOX_RE.finditer(text)):
        mark = m.group("mark")
        label = m.group("label").strip(" :-\t")
        if not label:
            continue
        checked = mark in {"☒", "☑", "✓", "✔", "[x]", "[X]", "(x)", "(X)"}
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "checkbox", idx),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={"page": page_number, "checkbox_index": idx},
            extraction_method="pdf_checkbox_state_v1",
            parser_version=parser_version,
        )
        atoms.append(
            EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    "checkbox",
                    page_number,
                    idx,
                    checked,
                    label,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.scope_item if checked else AtomType.exclusion,
                raw_text=f"{'Selected' if checked else 'Not selected'} checkbox: {label}",
                normalized_text=normalize_text(label),
                value={
                    "kind": "checkbox",
                    "label": label,
                    "checked": checked,
                    "page": page_number,
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.90 if checked else 0.72,
                review_status=ReviewStatus.auto_accepted
                if checked
                else ReviewStatus.needs_review,
                review_flags=[] if checked else ["unchecked_checkbox_not_scope"],
                parser_version=parser_version,
            )
        )
    return atoms


def _workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit one ``action_item`` atom per workflow step on a page that
    contains 3+ workflow verbs (detect / triage / contain / escalate /
    recover / remediate / notify / dispatch / close / improve).

    Page text is split on common arrow / pipe glyphs (→ -> › > / |)
    so ``Detect → Triage → Contain → Recover`` becomes 4 atoms."""
    if len(_WORKFLOW_STEP_RE.findall(text)) < 3:
        return []
    chunks = re.split(r"\s*(?:→|->|›|>|/|\|)\s*", text)
    atoms: list[EvidenceAtom] = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk or not _WORKFLOW_STEP_RE.search(chunk):
            continue
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "workflow", idx),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={"page": page_number, "workflow_step_index": idx},
            extraction_method="pdf_workflow_step_v1",
            parser_version=parser_version,
        )
        atoms.append(
            EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    "workflow",
                    page_number,
                    idx,
                    chunk,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.action_item,
                raw_text=chunk,
                normalized_text=normalize_text(chunk),
                value={
                    "kind": "workflow_step",
                    "step_index": idx,
                    "page": page_number,
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.86,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version=parser_version,
            )
        )
    return atoms


def _visual_review_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    parser_version: str,
    reason: str,
) -> EvidenceAtom:
    """Mark a low-text page as carrying visual evidence the structured
    pipeline could not extract (rack diagrams, floor plans, OCR-only
    pages). Surfaces as ``open_question`` with
    ``review_flags=['visual_evidence_not_fully_extracted']`` so the
    review UI surfaces the page instead of letting it disappear."""
    source_ref = SourceRef(
        id=stable_id("src", artifact_id, "pdf", page_number, "visual_review"),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator={"page": page_number},
        extraction_method="pdf_visual_page_marker_v1",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=stable_id(
            "atm", project_id, artifact_id, "visual_review", page_number, reason
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.open_question,
        raw_text=(
            f"PDF page {page_number} appears to contain visual / table / "
            "diagram evidence that was not fully extracted."
        ),
        normalized_text="visual evidence requires review",
        value={
            "kind": "visual_page_marker",
            "page": page_number,
            "reason": reason,
        },
        entity_keys=[],
        source_refs=[source_ref],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.60,
        review_status=ReviewStatus.needs_review,
        review_flags=["visual_evidence_not_fully_extracted"],
        parser_version=parser_version,
    )


def _scan_pdf_for_extras(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Single fitz pass — emit checkbox / workflow / visual atoms.

    Errors are swallowed by the caller so a malformed PDF can't kill
    the structured pipeline; this whole pass is best-effort enrichment.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return []

    out: list[EvidenceAtom] = []
    with fitz.open(str(path)) as doc:
        for page_idx in range(len(doc)):
            try:
                page_text = doc[page_idx].get_text("text") or ""
            except Exception:
                page_text = ""
            stripped = page_text.strip()
            if len(stripped) < _LOW_TEXT_VISUAL_THRESHOLD:
                out.append(
                    _visual_review_atom(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        page_number=page_idx + 1,
                        parser_version=parser_version,
                        reason=f"low_text_page_{len(stripped)}_chars",
                    )
                )
                continue
            out.extend(
                _checkbox_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            out.extend(
                _workflow_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
    return out


__all__ = [
    "OrbitBriefPdfParser",
    "PARSER_NAME",
    "PARSER_VERSION",
    "STRUCTURED_SCHEMA_VERSION",
    "STRUCTURED_FILENAME",
    "STRUCTURED_MARKDOWN_FILENAME",
    "DERIVED_DIR_SUFFIX",
    "build_structured_document",
    "write_structured_doc",
    "write_structured_markdown",
    "structured_doc_to_markdown",
    "derived_dir_for",
    "overlay_payload_and_extraction",
    "atoms_from_structured_doc",
]
