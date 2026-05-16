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
import os
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
        # Low-voltage takeoff layer — additive only. A failure here
        # NEVER fails the parse; it is captured as a warning instead.
        # The existing structured.json / structured.md / atom stream
        # above is untouched whether or not this succeeds.
        derived = derived_dir_for(path)
        derived_files: list[dict[str, Any]] = [
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
        ]
        warnings: list[str] = []
        try:
            from app.takeoff.exports import (
                TAKEOFF_FILENAME,
                TAKEOFF_MARKDOWN_FILENAME,
                takeoff_doc_to_markdown,
                takeoff_to_atoms,
                write_takeoff_doc,
                write_takeoff_markdown,
            )
            from app.takeoff.pipeline import build_low_voltage_takeoff

            takeoff_doc = build_low_voltage_takeoff(path)
            write_takeoff_doc(path, takeoff_doc)
            write_takeoff_markdown(path, takeoff_doc)
            # QA overlays are an expensive offline aid (one pixmap render per
            # page). Skip them on the hot parse path unless the operator opts
            # in via env flag. Truthy values: 1, true, yes, on (case-insensitive).
            if os.environ.get("PARSER_OS_WRITE_TAKEOFF_QA", "").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    from app.takeoff.qa_overlay import write_qa_overlays

                    write_qa_overlays(pdf_path=path, takeoff=takeoff_doc)
                except Exception:  # pragma: no cover - QA overlays are best-effort
                    pass
            atoms.extend(
                takeoff_to_atoms(
                    takeoff=takeoff_doc,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=self.parser_version,
                )
            )
            derived_files.append(
                {
                    "relative_path": f"{derived.name}/{TAKEOFF_FILENAME}",
                    "content_kind": "json",
                    "content_json": takeoff_doc.model_dump(mode="json"),
                }
            )
            derived_files.append(
                {
                    "relative_path": f"{derived.name}/{TAKEOFF_MARKDOWN_FILENAME}",
                    "content_kind": "markdown",
                    "content_text": takeoff_doc_to_markdown(takeoff_doc),
                }
            )
        except Exception as exc:  # pragma: no cover — never fail the parse
            warnings.append(f"low_voltage_takeoff_failed: {exc!r}")

        # Surface the derived artifacts in the parser output so the
        # compiler-level cache captures them and replays them on every
        # cache hit.  This guarantees ``<stem>.derived/structured.json``
        # and ``structured.md`` are always present after a compile, even
        # for cache-hot artifacts.
        return ParserOutput(
            atoms=atoms,
            warnings=warnings,
            derived_files=derived_files,
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

    document_title: str | None = None
    document_metadata: list[str] = []
    seen_metadata: set[str] = set()

    # P2.1: pre-scan the PDF for per-page text length so we can fast-path
    # low-text pages (scanned drawings, image-only floor plans) without
    # running the heavyweight layout-detection pipeline on them.
    # Insanity-perf: ALSO collect the actual page text so a text-rich
    # page can be parsed via a lightweight prose splitter without ever
    # touching the layout pipeline (which costs 5–10 s/page).
    page_text_lengths: list[int] = []
    page_texts: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for page_idx in range(page_count):
            try:
                page_text = doc[page_idx].get_text("text") or ""
            except Exception:  # pragma: no cover — bad page shouldn't kill compile
                page_text = ""
            page_texts.append(page_text)
            page_text_lengths.append(len(page_text.strip()))

    # Page bucketing thresholds:
    #   < LOW_TEXT_PAGE_THRESHOLD       → marker page only (scanned)
    #   >= TEXT_RICH_PAGE_THRESHOLD     → text-only fast path
    #   else                             → heavyweight layout pipeline
    LOW_TEXT_PAGE_THRESHOLD = 80
    TEXT_RICH_PAGE_THRESHOLD = 1200

    def _build_low_text_page(page_index: int) -> dict[str, Any]:
        return {
            "page": page_index,
            "title": None,
            "metadata": [
                f"[low-text page (≤{LOW_TEXT_PAGE_THRESHOLD} chars) "
                "— likely scanned image; layout pipeline skipped for perf]"
            ],
            "outline": [],
            "sections": [],
        }

    def _build_text_rich_page(page_index: int) -> dict[str, Any]:
        sections = _text_rich_sections(page_texts[page_index])
        _stamp_section_and_block_ids(sections, page_index)
        return {
            "page": page_index,
            "title": None,
            "metadata": [
                "[text-rich page — heavyweight layout pipeline skipped; "
                "prose extracted via lightweight text splitter]"
            ],
            "outline": [
                {"level": s.get("level", 2), "heading": s.get("heading"),
                 "block_count": len(s.get("blocks") or [])}
                for s in sections
            ],
            "sections": sections,
        }

    def _build_heavyweight_page(page_index: int) -> dict[str, Any]:
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
        sections = list(struct.get("sections") or [])
        _stamp_section_and_block_ids(sections, page_index)
        return {
            "page": page_index,
            "title": page_doc.get("title"),
            "metadata": list(page_doc.get("metadata") or []),
            "outline": list(struct.get("outline") or []),
            "sections": sections,
        }

    def _build_one_page(page_index: int) -> dict[str, Any]:
        if page_text_lengths[page_index] < LOW_TEXT_PAGE_THRESHOLD:
            return _build_low_text_page(page_index)
        if page_text_lengths[page_index] >= TEXT_RICH_PAGE_THRESHOLD:
            return _build_text_rich_page(page_index)
        return _build_heavyweight_page(page_index)

    # NOTE: PyMuPDF is NOT thread-safe — running the page loop on a
    # ThreadPoolExecutor crashes with SIGSEGV inside libmupdf. The
    # text-rich fast path (above) is the dominant speedup; pages
    # that still hit the heavyweight pipeline run serially. A future
    # optimization could spawn a process per page (multiprocessing
    # with each worker opening its own fitz doc), at the cost of
    # ~2 s per-fork startup on macOS.
    pages: list[dict[str, Any]] = [_build_one_page(i) for i in range(page_count)]

    # Aggregate document title + metadata across pages (in order).
    for p in pages:
        page_title = p.get("title")
        if not document_title and page_title:
            document_title = page_title
        for entry in p.get("metadata") or []:
            if not entry:
                continue
            key = normalize_text(entry)
            if not key or key in seen_metadata:
                continue
            seen_metadata.add(key)
            document_metadata.append(entry)

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
    # Boss-review v9 C002-F3 — dropped bare ``by vendor`` from the
    # exclusion list. ``blocked by vendor`` in MSP acceptance
    # checklists is a STATUS field, not a contractual exclusion.
    # Legitimate "by vendor" exclusions use ``performed by vendor``
    # / ``furnished by vendor`` which we don't classify as exclusion
    # either (those are RACI assignments).
    (re.compile(r"\b(by\s+(?:others|gc|owner|customer)|n\.?i\.?c\.?|provided\s+by\s+(?:others|owner))\b", re.I), AtomType.exclusion),
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


_BULLET_LINE_RE = re.compile(r"^\s*([-*•·\u2022]|\d+[.)])\s+(.+?)\s*$")
_HEADING_LINE_RE = re.compile(
    r"^\s*((?:[A-Z0-9][A-Z0-9 &/\-,()]{2,80})|(?:#{1,6}\s+.{2,80}))\s*$"
)


def _text_rich_sections(page_text: str) -> list[dict[str, Any]]:
    """Lightweight prose splitter for text-rich PDF pages.

    The heavyweight layout pipeline costs 5–10 s/page; on a
    text-rich page (≥ 1200 chars of clean text — NOC playbook,
    scope brief, terms-and-conditions) the layout boxes don't
    actually buy us anything beyond paragraph + bullet ordering.
    This function produces a structured ``sections`` list that
    matches the same shape ``extract_structured`` would, so the
    downstream atom emitter doesn't need to know which path
    produced the page.

    Heuristics:
      * blank line → end of paragraph
      * leading bullet glyph or "1." style → bullet item
      * an all-caps line (or markdown ``#``-prefixed) → heading;
        starts a new section, prior content flushed
      * otherwise → paragraph line, accumulated then joined.
    """
    if not page_text or not page_text.strip():
        return []

    lines = page_text.splitlines()
    sections: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []
    bullet_buffer: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = " ".join(x.strip() for x in paragraph_lines if x.strip()).strip()
        if text:
            current_blocks.append({"kind": "paragraph", "text": text})
        paragraph_lines = []

    def flush_bullets() -> None:
        nonlocal bullet_buffer
        if not bullet_buffer:
            return
        items = [{"text": x} for x in bullet_buffer if x.strip()]
        if items:
            current_blocks.append({"kind": "bullet_list", "items": items})
        bullet_buffer = []

    def flush_section() -> None:
        nonlocal current_heading, current_blocks
        flush_paragraph()
        flush_bullets()
        if current_blocks or current_heading:
            sections.append(
                {
                    "heading": current_heading or "",
                    "level": 2,
                    "blocks": current_blocks,
                    "subsections": [],
                }
            )
        current_heading = None
        current_blocks = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            continue

        bullet_m = _BULLET_LINE_RE.match(line)
        if bullet_m:
            flush_paragraph()
            bullet_buffer.append(bullet_m.group(2).strip())
            continue

        # heading guess (all caps or markdown-style #)
        stripped = line.strip()
        if (
            len(stripped) <= 80
            and (stripped.startswith("#") or (stripped.isupper() and len(stripped) >= 3))
        ):
            flush_section()
            current_heading = stripped.lstrip("# ").strip()
            continue

        # Paragraph continuation. Flush any pending bullets first so a
        # paragraph doesn't get glued onto a bullet list.
        flush_bullets()
        paragraph_lines.append(line)

    flush_section()
    # Drop empty sections that may have been created by trailing
    # whitespace.
    return [s for s in sections if s.get("blocks") or s.get("heading")]


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
# RF2 — literal "x Foo" / "X Foo" line-prefix detection. Many PDFs
# strip the unicode glyphs on text extraction, leaving sequences like
# ``x LogicMonitor x Microsoft Sentinel ServiceNow Event Mgmt x Aruba``
# where "x" prefixes the CHECKED option and unmarked words are the
# UNCHECKED alternatives. We scan a candidate line for the
# ``x <Word>`` literal pattern and emit one form_option_state atom
# per option, with ``checked=True`` for items preceded by literal
# x/X and ``checked=False`` for the unmarked siblings.
_LITERAL_X_OPTION_RE = re.compile(
    r"(?P<mark>\bx\b|\bX\b)\s+(?P<label>[A-Z][A-Za-z][A-Za-z0-9 \-/&._']{1,80}?)"
    r"(?=(?:\s+\bx\b|\s+\bX\b|\s*$|\s*[|;]|\s*[A-Z][A-Z]))",
    re.UNICODE,
)
# Heuristic: a "checkbox cluster" line has ≥2 capitalized labels and
# ≥1 literal x prefix.
_CHECKBOX_LITERAL_LINE_RE = re.compile(r"\bx\s+[A-Z]", re.UNICODE)
_WORKFLOW_STEP_RE = re.compile(
    r"\b(detect|triage|contain|escalate|recover|remediate|notify|"
    r"dispatch|close|improve)\b",
    re.I,
)
_LOW_TEXT_VISUAL_THRESHOLD = 80


# ───────────────── PR5 (post-v3) — PDF v2 supplements ─────────────────


_PDF_HEADER_LABELS_RE = re.compile(
    r"\bCUSTOMER\b.*\bSERVICE\s+LINE\b.*\bTARGET\s+GO[-\s]?LIVE\b",
    re.I,
)
_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


_HEADER_LABEL_LINE_RE = re.compile(
    r"^\s*(CUSTOMER|SERVICE\s+LINE|TARGET\s+GO[-\s]?LIVE)\s*:?\s*$",
    re.I,
)
_HEADER_LABEL_TO_FIELD = {
    "customer": "customer",
    "service line": "service_line",
    "target go-live": "target_go_live",
    "target go live": "target_go_live",
}


def _pdf_header_field_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """5A — extract CUSTOMER / SERVICE LINE / TARGET GO-LIVE header
    fields from a PDF page. Handles two layouts:

    1. Combined-line: ``CUSTOMER  SERVICE LINE  TARGET GO-LIVE`` on
       one line followed by 2-3 value lines.
    2. Separate-line: each label on its own line, followed by 1-3
       value lines until the next label or 3 lines pass.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[EvidenceAtom] = []

    # Try the separate-line layout first by finding any of the three
    # label lines.
    field_values: dict[str, str] = {}
    i = 0
    while i < min(len(lines), 80):
        m = _HEADER_LABEL_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        label_norm = re.sub(r"\s+", " ", m.group(1).lower()).replace("-", " ")
        field = _HEADER_LABEL_TO_FIELD.get(label_norm) or _HEADER_LABEL_TO_FIELD.get(
            label_norm.replace("go live", "go-live")
        )
        if not field:
            i += 1
            continue
        # Consume value lines until next label or 3 lines.
        value_parts: list[str] = []
        j = i + 1
        while j < len(lines) and j - i <= 3:
            if _HEADER_LABEL_LINE_RE.match(lines[j]):
                break
            value_parts.append(lines[j])
            j += 1
        value = " ".join(value_parts).strip()
        if field == "target_go_live":
            date_match = _DATE_RE.search(value)
            if date_match:
                value = date_match.group(0)
        if value:
            field_values.setdefault(field, value)
        i = j

    if field_values:
        for field, value in field_values.items():
            atom_type = (
                AtomType.project_metadata if field == "customer"
                else AtomType.scope_item if field == "service_line"
                else AtomType.constraint
            )
            kind = field
            source_ref = SourceRef(
                id=stable_id("src", artifact_id, "pdf", page_number, "header", field),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={"page": page_number, "header_field": field},
                extraction_method="pdf_header_kv_v2",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "pdf_header",
                        page_number, field, value,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=f"{field.replace('_', ' ').title()}: {value}",
                    normalized_text=normalize_text(value),
                    value={
                        "kind": kind,
                        "field": field,
                        "value": value,
                        "page": page_number,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.92,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
        return out

    # Combined-line fallback (rare but kept for compatibility).
    for i, line in enumerate(lines[:25]):
        if not _PDF_HEADER_LABELS_RE.search(line):
            continue

        customer = lines[i + 1] if i + 1 < len(lines) else ""
        service_line = ""
        target_go_live = ""

        if i + 2 < len(lines):
            candidate = lines[i + 2]
            date_match = _DATE_RE.search(candidate)
            if date_match:
                target_go_live = date_match.group(0)
                service_line = candidate[: date_match.start()].strip()
            else:
                service_line = candidate
                if i + 3 < len(lines):
                    date_match2 = _DATE_RE.search(lines[i + 3])
                    if date_match2:
                        target_go_live = date_match2.group(0)

        fields = [
            ("customer", customer, AtomType.project_metadata, "customer"),
            ("service_line", service_line, AtomType.scope_item, "service_line"),
            ("target_go_live", target_go_live, AtomType.constraint, "target_go_live"),
        ]
        for field, value, atom_type, kind in fields:
            if not value:
                continue
            source_ref = SourceRef(
                id=stable_id("src", artifact_id, "pdf", page_number, "header", field),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={"page": page_number, "header_field": field},
                extraction_method="pdf_header_kv_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "pdf_header",
                        page_number, field, value,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=f"{field.replace('_', ' ').title()}: {value}",
                    normalized_text=normalize_text(value),
                    value={
                        "kind": kind,
                        "field": field,
                        "value": value,
                        "page": page_number,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.92,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
        break
    return out


# 5B — Form grid (multi-line / multi-column "x Foo" tables).
_FORM_GROUP_HEADINGS: dict[str, dict[str, frozenset[str]]] = {
    "monitoring tool intake": {
        "known_options": frozenset(
            {
                "LogicMonitor",
                "Microsoft Sentinel",
                "ServiceNow Event Mgmt",
                "Aruba Central",
                "Meraki Dashboard",
                "Genetec Security Center",
                "PRTG",
                "Datadog",
            }
        )
    },
}


def _split_form_grid_line(line: str) -> list[tuple[str, bool]]:
    cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
    out: list[tuple[str, bool]] = []
    for cell in cells:
        selected = bool(re.match(r"^[xX]\s+", cell))
        label = re.sub(r"^[xX]\s+", "", cell).strip()
        if label:
            out.append((label, selected))
    return out


def _form_grid_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """5B — when a line names a known form-group heading (e.g.
    "Monitoring Tool Intake"), scan the next ~12 lines for option
    labels. Emit one ``form_option_state`` atom per known option,
    with ``selected=True`` if the cell starts with literal ``x ``,
    else ``selected=False``."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[EvidenceAtom] = []
    for i, line in enumerate(lines):
        group_name = normalize_text(line)
        group_config = _FORM_GROUP_HEADINGS.get(group_name)
        if group_config is None:
            continue
        known_options = group_config["known_options"]
        option_index = 0
        for j in range(i + 1, min(i + 12, len(lines))):
            candidate = lines[j].strip()
            if not candidate:
                break
            for label, selected in _split_form_grid_line(candidate):
                if label not in known_options:
                    continue
                source_ref = SourceRef(
                    id=stable_id(
                        "src", artifact_id, "pdf", page_number, "form_grid",
                        group_name, option_index,
                    ),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.pdf,
                    filename=filename,
                    locator={
                        "page": page_number,
                        "group": group_name,
                        "line_index": j,
                        "option_index": option_index,
                    },
                    extraction_method="pdf_form_grid_v1",
                    parser_version=parser_version,
                )
                out.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm", project_id, artifact_id, "form_grid",
                            page_number, group_name, option_index,
                            selected, label,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.form_option_state,
                        raw_text=(
                            f"{'Selected' if selected else 'Not selected'} "
                            f"{group_name}: {label}"
                        ),
                        normalized_text=normalize_text(label),
                        value={
                            "kind": "form_option_state",
                            "group": group_name,
                            "label": label,
                            "selected": selected,
                            "page": page_number,
                        },
                        entity_keys=[],
                        source_refs=[source_ref],
                        receipts=[],
                        authority_class=AuthorityClass.customer_current_authored,
                        confidence=0.90 if selected else 0.70,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[]
                        if selected
                        else ["form_option_unselected", "do_not_certify_as_exclusion"],
                        parser_version=parser_version,
                    )
                )
                option_index += 1
        break
    return out


# 5C — fix the "blocked by vendor" / "by vendor" false-positive.
_EXPLICIT_BY_OTHERS_RE = re.compile(
    r"\b("
    r"by\s+(?:others|gc)\b|"
    r"n\.?i\.?c\.?|"
    r"provided\s+by\s+(?:others|owner|customer)|"
    r"performed\s+by\s+(?:others|owner|customer)|"
    r"furnished\s+by\s+(?:others|owner|customer)|"
    r"owner[-\s]?provided|customer[-\s]?provided"
    r")\b",
    re.I,
)


# 5D — field checklist row.
_FIELD_CHECKLIST_ROW_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s{2,}"
    r"(?P<item>.+?)\s{2,}"
    r"(?P<status>OPEN|N/A|NA|PASS|FAIL|BLOCKED|CLOSED|PENDING)\s{2,}"
    r"(?P<area>[A-Za-z0-9 /_-]{2,60})\s{2,}"
    r"(?P<note>.+?)\s*$",
    re.I,
)


def _field_checklist_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """5D — emit one atom per field-checklist row when a page
    contains the literal phrase 'field checklist'."""
    if "field checklist" not in normalize_text(text):
        return []
    out: list[EvidenceAtom] = []
    for line_idx, line in enumerate(text.splitlines()):
        m = _FIELD_CHECKLIST_ROW_RE.match(line)
        if not m:
            continue
        item_no = m.group("num")
        item = m.group("item").strip()
        status = m.group("status").strip()
        area = m.group("area").strip()
        note = m.group("note").strip()
        atom_type = (
            AtomType.constraint
            if status.upper() in {"OPEN", "BLOCKED", "PENDING"}
            else AtomType.scope_item
        )
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "field_check", item_no),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "line_index": line_idx,
                "field_check_item": item_no,
            },
            extraction_method="pdf_field_checklist_row_v1",
            parser_version=parser_version,
        )
        out.append(
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "field_checklist",
                    page_number, item_no, item, status,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=f"Field checklist {item_no}: {item} | {status} | {area} | {note}",
                normalized_text=normalize_text(item),
                value={
                    "kind": "field_checklist_row",
                    "item_no": item_no,
                    "item": item,
                    "status": status,
                    "area": area,
                    "note": note,
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
    return out


# 5E — horizontal workflow (Detect | Triage | Contain | Escalate | Recover | Improve).
_WORKFLOW_ORDER = ["Detect", "Triage", "Contain", "Escalate", "Recover", "Improve"]


def _horizontal_workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[EvidenceAtom] = []
    heading_idx = None
    for i, line in enumerate(lines):
        if all(step.lower() in line.lower() for step in _WORKFLOW_ORDER):
            heading_idx = i
            break
    if heading_idx is None:
        return out
    # PR5 — descriptions can be:
    #   (a) one line per step (already array-aligned), OR
    #   (b) ONE line with all step descriptions separated by ≥2 spaces.
    # Try (b) first when the very next line splits into N pieces.
    raw_descs: list[str] = []
    if heading_idx + 1 < len(lines):
        candidate = lines[heading_idx + 1]
        cells = [c.strip() for c in re.split(r"\s{2,}", candidate.strip()) if c.strip()]
        if len(cells) == len(_WORKFLOW_ORDER):
            raw_descs = cells
    if not raw_descs:
        raw_descs = lines[heading_idx + 1 : heading_idx + 1 + len(_WORKFLOW_ORDER)]
    for idx, step in enumerate(_WORKFLOW_ORDER):
        desc = raw_descs[idx] if idx < len(raw_descs) else ""
        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "pdf", page_number, "workflow_horizontal", idx,
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "workflow_step_index": idx,
                "layout": "horizontal",
            },
            extraction_method="pdf_horizontal_workflow_v1",
            parser_version=parser_version,
        )
        out.append(
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "workflow_horizontal",
                    page_number, idx, step, desc,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.action_item,
                raw_text=f"Workflow step {idx + 1} {step}: {desc}".strip(),
                normalized_text=normalize_text(f"{step} {desc}"),
                value={
                    "kind": "workflow_step",
                    "step_index": idx,
                    "step_name": step,
                    "description": desc,
                    "page": page_number,
                    "layout": "horizontal",
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.82,
                review_status=ReviewStatus.needs_review,
                review_flags=["layout_derived_workflow"],
                parser_version=parser_version,
            )
        )
    return out


# 5C support — aggregate paragraph that lists all monitoring tool
# names but lost the per-option state. Detect + suppress so the
# brain doesn't see the ambiguous string.
_MONITORING_TOOL_NAMES = frozenset(
    {
        "logicmonitor",
        "microsoft sentinel",
        "servicenow event mgmt",
        "aruba central",
        "meraki dashboard",
        "genetec security center",
        "prtg",
        "datadog",
    }
)


def _looks_like_form_option_aggregate(text: str) -> bool:
    low = normalize_text(text)
    hits = sum(1 for name in _MONITORING_TOOL_NAMES if name in low)
    return hits >= 4 and "selected" not in low and "not selected" not in low


_SINGLE_LINE_X_RE = re.compile(
    r"^\s*([xX])\s+(?P<label>[A-Z][A-Za-z][A-Za-z0-9 \-/&._']{1,80}?)\s*$"
)


def _literal_x_checkbox_atoms_from_line(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    line: str,
    line_index: int,
    parser_version: str,
) -> list[EvidenceAtom]:
    """RF2 — emit one ``form_option_state`` atom per option on a
    line like ``"x LogicMonitor x Microsoft Sentinel ServiceNow x Aruba"``.

    Two modes:

    1. SINGLE-LINE: a line that is exactly ``"x SomeLabel"`` is one
       checked option. (PDFs frequently render each option on its
       own line.)

    2. MULTI-OPTION: a line with 2+ literal-x markers gets split
       into per-option atoms; the first label after each marker is
       CHECKED and any sibling Title-Case clusters between markers
       are UNCHECKED.
    """
    # ── single-line "x Label" → one checked option ──
    m = _SINGLE_LINE_X_RE.match(line)
    if m:
        label = m.group("label").strip()
        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "pdf", page_number,
                "literal_x_checkbox", line_index, 0,
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "line_index": line_index,
                "checkbox_index": 0,
            },
            extraction_method="pdf_literal_x_checkbox_v1",
            parser_version=parser_version,
        )
        return [
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "literal_x_checkbox",
                    page_number, line_index, 0, True, label,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.scope_item,
                raw_text=f"Selected option: {label}",
                normalized_text=normalize_text(label),
                value={
                    "kind": "checkbox",
                    "label": label,
                    "checked": True,
                    "page": page_number,
                    "extraction": "literal_x_marker",
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.85,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version=parser_version,
            )
        ]

    # ── multi-option same-line ──
    if not _CHECKBOX_LITERAL_LINE_RE.search(line):
        return []

    # Tokenize: split on whitespace, walk tokens, accumulate labels
    # until the next "x" / "X" sentinel or another capitalized
    # standalone word.
    tokens = line.split()
    if len(tokens) < 4:
        return []

    def _split_into_labels(words: list[str]) -> list[str]:
        """A label is 1-3 consecutive Title Case / ALL CAPS words.
        Lower-case connector words ("of", "and", "the") within a
        ≤3-word group are kept; everything else starts a new label.
        """
        out_labels: list[str] = []
        cur: list[str] = []
        for w in words:
            looks_like_label_word = (
                w[:1].isupper() if w else False
            ) or w.isupper()
            connector = w.lower() in {"of", "and", "the", "for", "to"}
            if looks_like_label_word and len(cur) >= 3:
                out_labels.append(" ".join(cur))
                cur = [w]
            elif looks_like_label_word:
                cur.append(w)
            elif connector and cur:
                cur.append(w)
            elif cur:
                out_labels.append(" ".join(cur))
                cur = []
        if cur:
            out_labels.append(" ".join(cur))
        return out_labels

    options: list[tuple[str, bool]] = []  # (label, checked)
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        is_marker = tok in ("x", "X")
        if is_marker:
            # Consume words until the next marker; first label is
            # checked, any subsequent labels in the same run are
            # unchecked siblings.
            i += 1
            run: list[str] = []
            while i < n and tokens[i] not in ("x", "X"):
                run.append(tokens[i])
                i += 1
            labels = _split_into_labels(run)
            for j, label in enumerate(labels):
                options.append((label, j == 0))
        else:
            # Pre-marker run — all unchecked.
            run = []
            while i < n and tokens[i] not in ("x", "X"):
                run.append(tokens[i])
                i += 1
            for label in _split_into_labels(run):
                options.append((label, False))

    out: list[EvidenceAtom] = []
    for opt_idx, (label, checked) in enumerate(options):
        atom_type = AtomType.scope_item if checked else AtomType.form_option_state
        confidence = 0.85 if checked else 0.55
        review_status = (
            ReviewStatus.auto_accepted if checked else ReviewStatus.needs_review
        )
        review_flags: list[str] = (
            []
            if checked
            else ["unchecked_checkbox_ambiguous", "do_not_certify_as_exclusion"]
        )
        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "pdf", page_number, "literal_x_checkbox",
                line_index, opt_idx,
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "line_index": line_index,
                "checkbox_index": opt_idx,
            },
            extraction_method="pdf_literal_x_checkbox_v1",
            parser_version=parser_version,
        )
        out.append(
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "literal_x_checkbox",
                    page_number, line_index, opt_idx, checked, label,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=f"{'Selected' if checked else 'Not selected'} option: {label}",
                normalized_text=normalize_text(label),
                value={
                    "kind": "checkbox",
                    "label": label,
                    "checked": checked,
                    "page": page_number,
                    "extraction": "literal_x_marker",
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=confidence,
                review_status=review_status,
                review_flags=review_flags,
                parser_version=parser_version,
            )
        )
    return out


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
        # Revised checkbox semantics (post-PR7 review). Checked
        # boxes are evidence the option WAS selected → scope_item.
        # Unchecked boxes are AMBIGUOUS — they can mean "not selected",
        # "not applicable", "blank option", or "not answered". So
        # unchecked emits ``form_option_state`` (a neutral marker) and
        # is left for the packetizer to combine with explicit
        # exclusion language elsewhere if appropriate. Never auto-
        # certify an unchecked box as a contractual exclusion.
        atom_type = AtomType.scope_item if checked else AtomType.form_option_state
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
                atom_type=atom_type,
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
                confidence=0.90 if checked else 0.55,
                review_status=ReviewStatus.auto_accepted
                if checked
                else ReviewStatus.needs_review,
                review_flags=[]
                if checked
                else [
                    "unchecked_checkbox_ambiguous",
                    "do_not_certify_as_exclusion",
                ],
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


# =====================================================================
# Boss-review (post-2-case) PDF v3 — vertical-listed tables, vertical
# workflow, and group-aware form-option states.
# =====================================================================

# Each profile is a dict so we can attach optional per-profile guards
# without touching every existing entry.  Required keys:
#   header:          tuple[str, ...]   — header tokens, lower
#   atom_kind:       str               — value.kind tag
#   atom_type:       str               — AtomType enum value
#   field_names:     tuple[str, ...]   — value-dict keys for each cell
#   locator_label:   str               — short tag in source_ref.locator
# Optional keys (post-v8 boss review hardening):
#   first_cell_re:   compiled regex    — every row's first cell MUST match
#                                        or the table parser stops early
#   row_stop_re:     compiled regex    — when the FIRST cell matches, stop
#                                        (e.g., page-2 measurement table
#                                        below field-checklist).
_PORT_TOKEN_RE = re.compile(r"^(gi|fa|te|xe|et|eth|mgmt)\d+/\d+(/\d+)?$", re.I)
_RFI_ID_RE = re.compile(r"^rfi-\d{2,4}$", re.I)
_RB_ID_RE = re.compile(r"^rb-\d{2,4}$", re.I)
_MEAS_ID_RE = re.compile(r"^m-\d{2,4}$", re.I)
_FCHK_NUM_RE = re.compile(r"^\d{1,3}$")
# Anything that looks like the start of a NEW table / section header
# terminates the previous table early. Boss-review v8 follow-up:
# applied ONLY at row boundaries (i.e., when the first cell of a new
# row is being read), never mid-row. We also exclude single nouns like
# "port" / "patch" / "vlan" that legitimately appear as data cells
# inside other tables (e.g. "patch field" in the measurement table).
_NEW_TABLE_HEADER_RE = re.compile(
    r"^("
    r"working\s+measurements|nonconforming\s+items?|"
    r"open\s+rfis?|acceptance\s+exceptions?|"
    r"required\s+signatures?|signature/date|customer\s+it\s+signature|"
    r"facilities\s+signature|msp\s+pm\s+signature|field\s+lead\s+signature|"
    r"layout\s+reference|reference\s+urls?|"
    r"hand\s+correction|mark[- ]?up|"
    r"synthetic\s+planning|"
    r"incident\s+and\s+vulnerability"
    r")\b",
    re.I,
)
# Workflow-specific stop tokens — applied only by the vertical-workflow
# extractor when collecting the description for the FINAL step
# ("Improve"). These are bare single nouns that legitimately appear as
# data cells inside other tables, so we never use them in
# _NEW_TABLE_HEADER_RE.
_WORKFLOW_STOP_RE = re.compile(
    r"^(runbook|trigger|owner|status|evidence|"
    r"cyber\s*/\s*logging\s+notes|notes?)\s*$",
    re.I,
)


_VERTICAL_TABLE_PROFILES: list[dict] = [
    {
        "header": ("#", "survey item", "status", "area", "note"),
        "atom_kind": "field_checklist_row_v2",
        "atom_type": "scope_item",
        "field_names": ("item_no", "item", "status", "area", "note"),
        "locator_label": "field_check",
        "first_cell_re": _FCHK_NUM_RE,  # F2 — only digits
    },
    # Boss-review v9 C002-F3 — Managed Services Acceptance Checklist
    # ("# / Acceptance Item / Status / Owner / Due"). Status values
    # like "Customer Pending" / "Exception" / "blocked by vendor"
    # belong here as ``open_question`` / ``action_item`` atoms, NOT
    # as scope_exclusion atoms.
    {
        "header": ("#", "acceptance item", "status", "owner", "due"),
        "atom_kind": "managed_services_acceptance_checklist_row",
        "atom_type": "open_question",
        "field_names": ("item_no", "item", "status", "owner", "due"),
        "locator_label": "msp_acceptance_checklist",
        "first_cell_re": _FCHK_NUM_RE,
    },
    {
        "header": ("rfi", "issue", "owner", "status", "needed by"),
        "atom_kind": "rfi_row",
        "atom_type": "open_question",
        "field_names": ("rfi_id", "issue", "owner", "status", "needed_by"),
        "locator_label": "rfi",
        "first_cell_re": _RFI_ID_RE,  # F3 — strictly RFI-### only
    },
    {
        "header": ("ref", "measurement", "value", "field note"),
        "atom_kind": "working_measurement_row",
        "atom_type": "quantity",
        "field_names": ("ref", "measurement", "value", "field_note"),
        "locator_label": "measurement",
        "first_cell_re": _MEAS_ID_RE,  # F2 — strictly M-### only
    },
    {
        "header": ("port", "patch", "vlan/use", "note"),
        "atom_kind": "port_vlan_assignment",
        "atom_type": "port_vlan_assignment",
        "field_names": ("port", "patch", "vlan_use", "note"),
        "locator_label": "port_vlan",
        "first_cell_re": _PORT_TOKEN_RE,  # F4 — must be Gi/Fa/Te/etc switch port
    },
    {
        "header": ("runbook", "trigger", "owner", "status", "evidence"),
        "atom_kind": "runbook_row",
        "atom_type": "action_item",
        "field_names": ("runbook_id", "trigger", "owner", "status", "evidence"),
        "locator_label": "runbook",
        "first_cell_re": _RB_ID_RE,  # only RB-### rows
    },
]


def _vertical_table_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Detect vertical-listed tables on a PDF page.

    Boss-review F3+F4: the original ``_field_checklist_atoms_from_text``
    required all 5 cells on one line. PyMuPDF on hand-form/scanned-feel
    PDFs returns each cell on its OWN line. We detect headers on
    consecutive lines, then chunk subsequent lines into N-row groups.
    """
    out: list[EvidenceAtom] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    norm = [normalize_text(ln).strip() for ln in lines]
    i = 0
    while i < len(lines):
        # Skip empties cheaply.
        if not norm[i]:
            i += 1
            continue
        for profile in _VERTICAL_TABLE_PROFILES:
            header_tokens = profile["header"]
            atom_kind = profile["atom_kind"]
            atom_type_str = profile["atom_type"]
            field_names = profile["field_names"]
            locator_label = profile["locator_label"]
            first_cell_re = profile.get("first_cell_re")
            n = len(header_tokens)
            # Try to align the next n non-empty lines to header_tokens.
            cand: list[int] = []
            j = i
            while j < len(lines) and len(cand) < n:
                if norm[j]:
                    cand.append(j)
                j += 1
            if len(cand) < n:
                continue
            if not all(norm[cand[k]] == header_tokens[k] for k in range(n)):
                continue
            # Header matched. Read row groups.
            row_idx = 0
            cursor = cand[-1] + 1
            while cursor < len(lines):
                row_lines: list[int] = []
                while cursor < len(lines) and len(row_lines) < n:
                    if norm[cursor]:
                        # Boss-review v8 F2/F3/F4/F5 — STOP if the
                        # FIRST cell of a new row matches a known new-
                        # table header. We only apply this check at
                        # row boundaries (len(row_lines)==0) so we
                        # don't accidentally cut a row mid-way when a
                        # data cell happens to share a header word.
                        if len(row_lines) == 0 and _NEW_TABLE_HEADER_RE.match(lines[cursor].strip()):
                            break
                        row_lines.append(cursor)
                    cursor += 1
                if len(row_lines) < n:
                    break
                row_values = [lines[ix].strip() for ix in row_lines]
                first = row_values[0]
                # Heuristic + per-profile guard — the first cell must
                # match the profile's expected pattern (Gi…, RFI-###,
                # M-###, RB-###, or an integer for field-checklist).
                if not first or first.isupper() and len(first.split()) > 4:
                    break
                if first_cell_re is not None and not first_cell_re.match(first):
                    # Stop the table; the row that failed the guard is
                    # likely the start of a different section.
                    break
                row_dict = dict(zip(field_names, row_values))
                # Determine atom type — "OPEN" status → constraint not scope_item.
                atype = atom_type_str
                status = row_dict.get("status", "")
                if atom_kind == "field_checklist_row_v2" and status.upper() in {"OPEN", "BLOCKED", "PENDING", "EXCEPTION", "RFI"}:
                    atype = "constraint"
                # Build atom.
                try:
                    resolved_atom_type = AtomType(atype)
                except ValueError:
                    resolved_atom_type = AtomType.scope_item
                row_id = row_values[0] or f"row_{row_idx}"
                source_ref = SourceRef(
                    id=stable_id("src", artifact_id, "pdf", page_number, locator_label, row_id),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.pdf,
                    filename=filename,
                    locator={
                        "page": page_number,
                        "vertical_table": locator_label,
                        "row_index": row_idx,
                        "row_id": row_id,
                    },
                    extraction_method=f"pdf_vertical_table_v1::{atom_kind}",
                    parser_version=parser_version,
                )
                pretty = " | ".join(f"{fn}: {row_values[k]}" for k, fn in enumerate(field_names))
                out.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm", project_id, artifact_id, atom_kind,
                            page_number, row_idx, *row_values,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=resolved_atom_type,
                        raw_text=pretty,
                        normalized_text=normalize_text(pretty),
                        value={
                            "kind": atom_kind,
                            "page": page_number,
                            "row_index": row_idx,
                            **row_dict,
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
                row_idx += 1
            # After processing the table, advance i past it.
            i = cursor
            break
        else:
            i += 1
    return out


# =====================================================================
# Boss-review F6 — vertical workflow steps.
# =====================================================================
def _vertical_workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit one ``action_item`` atom per workflow step when steps are
    listed vertically (each step name on its own line followed by a
    short description that may span 1-2 lines).

    Trigger phrase: ``Incident and Vulnerability Response Workflow`` or
    a sequence where ``Detect`` and ``Triage`` appear on consecutive
    non-empty lines (a strong vertical signal).
    """
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[EvidenceAtom] = []
    n = len(lines)
    # Locate the first occurrence of "Detect" on its own line where
    # "Triage" appears within the next 3 non-empty lines.
    for i in range(n):
        if lines[i].lower() != "detect":
            continue
        # Confirm Triage appears within the next ~6 non-empty lines.
        seen: list[int] = []
        j = i + 1
        while j < n and len(seen) < 6:
            if lines[j]:
                seen.append(j)
            j += 1
        if not any(lines[k].lower() == "triage" for k in seen):
            continue
        # Collect step boundaries by scanning forward.
        steps_lower = ["detect", "triage", "contain", "escalate", "recover", "improve"]
        anchor_indices: dict[str, int] = {}
        cursor = i
        for step in steps_lower:
            while cursor < n and lines[cursor].lower() != step:
                cursor += 1
            if cursor >= n:
                break
            anchor_indices[step] = cursor
            cursor += 1
        if len(anchor_indices) < 4:
            return out
        # For each step, the description is everything between its
        # anchor and the next anchor (or up to 4 lines).
        step_keys = [s for s in steps_lower if s in anchor_indices]
        anchors_ordered = [anchor_indices[s] for s in step_keys]
        anchors_ordered.append(min(n, anchors_ordered[-1] + 6))
        for idx, step in enumerate(step_keys):
            start = anchors_ordered[idx] + 1
            end = anchors_ordered[idx + 1]
            desc_lines: list[str] = []
            for k in range(start, end):
                ln = lines[k]
                if not ln:
                    continue
                # Boss-review v8 F5 — stop description collection when
                # the next table header begins (Runbook | Trigger |
                # Owner | Status | Evidence on noc_soc page 2 was
                # bleeding into "Improve").
                if _NEW_TABLE_HEADER_RE.match(ln) or _WORKFLOW_STOP_RE.match(ln):
                    break
                desc_lines.append(ln)
            desc = " ".join(desc_lines).strip()
            step_name = step.title()
            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, "pdf", page_number, "workflow_vertical", idx,
                ),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={
                    "page": page_number,
                    "workflow_step_index": idx,
                    "layout": "vertical",
                },
                extraction_method="pdf_vertical_workflow_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "workflow_vertical",
                        page_number, idx, step_name, desc,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.action_item,
                    raw_text=f"Workflow step {idx + 1} {step_name}: {desc}".strip(),
                    normalized_text=normalize_text(f"{step_name} {desc}"),
                    value={
                        "kind": "workflow_step",
                        "step_index": idx,
                        "step_name": step_name,
                        "description": desc,
                        "page": page_number,
                        "layout": "vertical",
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.84,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
        return out
    return out


# =====================================================================
# Boss-review F5 — group-aware unchecked form-option detection.
# =====================================================================
_FORM_OPTION_GROUP_HEADERS: tuple[str, ...] = (
    "connection availability / field checks",
    "connection availability",
    "field checks",
    "site survey - access checklist",
    "site survey access checklist",
)
_FORM_OPTION_END_MARKERS: tuple[str, ...] = (
    # Boss-review v9 C001-F2/C002-F2 — substring matchers that ALWAYS
    # indicate a real section break. We removed bare single nouns
    # like "port" / "table" because they appeared inside legitimate
    # option labels ("Network port available", "Patch panel
    # accessible") and were stopping the parser at row 5 of 8.
    "margin note",
    "synthetic planning",
    "field checklist - pathway",
    "rack elevation",
    "open rfis",
    "open rfi",
    "working measurements",
    "as-built exception",
    "required signatures",
    "page 1",
    "page 2",
    "incident workflow",
)


def _group_form_option_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit ``form_option_state`` atoms for a known checkbox group.

    Boss-review F5: the parser already emits checked options from
    lines starting with ``x`` (via _SINGLE_LINE_X_RE), but unchecked
    options have no leading sentinel. We anchor on a known group
    header (e.g., 'Connection Availability / Field Checks') and treat
    the next contiguous run of single-line items as form options,
    selecting=true if the line starts with ``x``.
    """
    out: list[EvidenceAtom] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        norm = normalize_text(line).strip()
        if norm not in _FORM_OPTION_GROUP_HEADERS:
            continue
        # Collect up to 12 following non-empty lines as candidate options.
        opts: list[tuple[int, str]] = []
        j = i + 1
        while j < n and len(opts) < 12:
            ln = lines[j].strip()
            if not ln:
                j += 1
                continue
            normln = normalize_text(ln)
            if any(end in normln for end in _FORM_OPTION_END_MARKERS):
                break
            # Skip pure section labels.
            if ln.endswith(":") or len(ln.split()) > 12:
                break
            opts.append((j, ln))
            j += 1
        if not opts:
            continue
        for idx, (line_idx, raw) in enumerate(opts):
            selected = bool(re.match(r"^\s*x\s+\S", raw, re.I))
            label = re.sub(r"^\s*x\s+", "", raw, flags=re.I).strip()
            if not label:
                continue
            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, "pdf", page_number, "form_option_group", idx,
                ),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={
                    "page": page_number,
                    "line_index": line_idx,
                    "form_group": norm,
                    "option_index": idx,
                },
                extraction_method="pdf_group_form_option_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "form_option_grouped",
                        page_number, idx, label, "selected" if selected else "unselected",
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.form_option_state,
                    raw_text=("Selected option: " if selected else "Unselected option: ") + label,
                    normalized_text=normalize_text(label),
                    value={
                        "kind": "form_option_state",
                        "group": norm,
                        "label": label,
                        "selected": selected,
                        "page": page_number,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.84,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
    return out


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
            # RF2 — literal "x Foo x Bar" line scan for PDFs whose
            # text extraction lost the unicode checkbox glyphs.
            for line_idx, line in enumerate(page_text.splitlines()):
                out.extend(
                    _literal_x_checkbox_atoms_from_line(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        page_number=page_idx + 1,
                        line=line,
                        line_index=line_idx,
                        parser_version=parser_version,
                    )
                )
            # PR5 (post-v3) — header KV / form grid / field checklist /
            # horizontal workflow.
            out.extend(
                _pdf_header_field_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            out.extend(
                _form_grid_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            out.extend(
                _field_checklist_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Boss-review F3+F4 — vertical-listed table v2 (each cell
            # on its own line, common with hand-form PDFs).
            out.extend(
                _vertical_table_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Boss-review F5 — group-aware form options (selected=true
            # AND selected=false) under known group headers.
            out.extend(
                _group_form_option_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Prefer the horizontal six-step workflow if the page has
            # one; otherwise try the vertical workflow (each step name
            # on its own line); fall back to the original verb-density
            # workflow extractor.
            horizontal = _horizontal_workflow_atoms_from_text(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                page_number=page_idx + 1,
                text=page_text,
                parser_version=parser_version,
            )
            vertical_workflow: list[EvidenceAtom] = []
            if not horizontal:
                vertical_workflow = _vertical_workflow_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            if horizontal:
                out.extend(horizontal)
            elif vertical_workflow:
                out.extend(vertical_workflow)
            else:
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
