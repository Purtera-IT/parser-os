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
from app.parsers.binary_markers import region_marker

# Moved to app.parsers.pdf._shared. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf._shared import (  # noqa: E402
    EXTRACTION_METHOD,
    TABLE_ROW_CONFIDENCE,
    _BARE_URL_RE,
    _COPYRIGHT_PATTERN,
    _CURRENCY_PATTERN,
    _FIELD_LABEL_RULE,
    _FIELD_LABEL_WORDS,
    _FILLED_FIELD_RE,
    _FORM_FIELD_KEYWORDS,
    _FORM_FIELD_STRONG_MARKERS,
    _FORM_FIELD_WEAK_MARKERS,
    _FORM_INSTRUCTION_RE,
    _FORM_INTERROG_RE,
    _FORM_PAGE_RULE,
    _IMAGE_FIELD_LABEL_RULE,
    _NEW_TABLE_HEADER_RE,
    _PAGE_FOOTER_HINTS,
    _PAGE_NUMBER_LOOSE_PATTERN,
    _PAGE_NUMBER_PATTERN,
    _PHOTO_REQUEST_RE,
    _PHOTO_REQUEST_RULE,
    _PRICING_COLUMN_HINTS,
    _SECTION_RULES,
    _SIGNOFF_RE,
    _classify_table,
    _field_label_lexical,
    _field_label_rule,
    _form_page_rule,
    _image_field_label_rule,
    _is_image_field_label,
    _is_photo_request,
    _is_value_field_label,
    _looks_like_form_field,
    _looks_like_page_footer,
    _make_atom,
    _page_is_form,
    _page_is_form_lexical,
    _photo_request_lexical,
    _photo_request_rule,
    _table_rows_repaired,
)

# Moved to app.parsers.pdf.schematic_pre_pass. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf.schematic_pre_pass import (  # noqa: E402
    _LEGEND_TOKEN_BLOCKLIST,
    _UNKNOWN_TOKEN_IGNORES,
    _augment_legend_with_orphan_tokens,
    _bbox_intersects,
    _run_schematic_pre_pass,
    _unknown_symbol_warnings,
)

# Moved to app.parsers.pdf.site_roster. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf.site_roster import (  # noqa: E402
    _fitz_site_roster_fallback,
    _text_based_site_roster_extract,
)

# Moved to app.parsers.pdf.tables. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf.tables import (  # noqa: E402
    _FCHK_NUM_RE,
    _MEAS_ID_RE,
    _PORT_TOKEN_RE,
    _RB_ID_RE,
    _REVISION_ROW_RE,
    _RFI_ID_RE,
    _TOOL_ROW_RE,
    _VERTICAL_TABLE_PROFILES,
    _extract_column_tables,
    _fitz_generic_table_fallback,
    _looks_like_document_control_row,
    _structured_doc_has_tables,
    _vertical_table_atoms_from_text,
)

# Moved to app.parsers.pdf.forms. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf.forms import (  # noqa: E402
    _CHECKBOX_LITERAL_LINE_RE,
    _CONNECTOR_WORDS,
    _DATE_RE,
    _HEADER_LABEL_LINE_RE,
    _HEADER_LABEL_TO_FIELD,
    _PDF_HEADER_LABELS_RE,
    _SINGLE_LINE_X_RE,
    _is_discrete_answer,
    _literal_x_checkbox_atoms_from_line,
    _pdf_header_field_atoms_from_text,
    _regroup_form_qa,
)

# Moved to app.parsers.pdf.images. Re-exported so every existing import keeps working;
# this module stays the single public entry point for PDF parsing.
from app.parsers.pdf.images import (  # noqa: E402
    _CHECKBOX_RE,
    _FIELD_CHECKLIST_ROW_RE,
    _FORM_GROUP_HEADINGS,
    _FORM_OPTION_END_MARKERS,
    _FORM_OPTION_GROUP_HEADERS,
    _LOW_TEXT_VISUAL_THRESHOLD,
    _WORKFLOW_ORDER,
    _WORKFLOW_STEP_RE,
    _WORKFLOW_STOP_RE,
    _checkbox_atoms_from_text,
    _field_checklist_atoms_from_text,
    _form_grid_atoms_from_text,
    _group_form_option_atoms_from_text,
    _horizontal_workflow_atoms_from_text,
    _ocr_fallback_atoms,
    _pdf_image_markers,
    _safe_stem,
    _scan_pdf_for_extras,
    _split_form_grid_line,
    _vertical_workflow_atoms_from_text,
    _visual_review_atom,
    _workflow_atoms_from_text,
)







PARSER_NAME = "orbitbrief_pdf"
PARSER_VERSION = "orbitbrief_pdf_v3"
STRUCTURED_SCHEMA_VERSION = "orbitbrief.pdf.structured.v1"
DERIVED_DIR_SUFFIX = ".derived"
STRUCTURED_FILENAME = "structured.json"
STRUCTURED_MARKDOWN_FILENAME = "structured.md"

DEFAULT_BLOCK_CONFIDENCE = 0.88
DEFAULT_NOTE_CONFIDENCE = 0.78

PDF_MAGIC = b"%PDF-"


# ─── PRODUCTION_GAPS P1.1: Q&A-aware paragraph segmentation ───
# When PDF text extraction collapses an entire pre-proposal Q&A
# transcript into a single paragraph block, downstream packet anchors
# become unusable 2,400-character keys.  This regex splits a paragraph
# at every Q-marker / A-marker boundary so each Q-pair becomes its own
# atom (with its own entity_keys, qa:qN markers, etc.).
_QA_BOUNDARY_REGEX = re.compile(r"(?=(?:^|\s)[QA]\d{1,3}\.\s)")
_QA_PAIR_PROBE = re.compile(r"\b[QA]\d{1,3}\.\s")

# ─── v57 P1.1b: form-style Q&A (no Q1./A1. markers) ───
# Discovery notes / pre-proposal interview transcripts often use a
# free-form pattern like:
#   "City/state for this location? – location Santa Fe, NM 87506
#    What size TVs? – LG part 65UN570H0UD – 65"
#    Will techs need to perform an inventory count?  Yes
#    Property has 23 dwellings, approx 8 have second story..."
#
# Each question ends in "?" and is followed by an answer (possibly
# prefixed with an em-dash "–"). The whole transcript is one PDF
# paragraph, so without splitting we get one giant blob.
_FORM_QA_SEGMENT = re.compile(
    # Split AFTER a "?" + following whitespace. Each resulting segment
    # (except possibly the last) ends in "?" and has the shape
    # "<answer-to-previous-question> <next-question?>". The >=3 "?" gate
    # at the top of ``_split_form_qa_blob`` prevents firing on normal
    # prose with one or two rhetorical questions.
    r"(?<=\?)\s+"
)
# A question almost always opens with a Wh-word or an auxiliary/modal
# verb. We use that to find where the *next* question starts inside a
# segment, so the preceding text (the previous answer) attaches to the
# question it actually answers -- not glued onto the next question.
_QUESTION_START = re.compile(
    r"\b(?:what|where|when|who|whom|whose|which|why|how|"
    r"is|are|am|was|were|do|does|did|will|would|can|could|"
    r"should|shall|has|have|had|may|might|must)\b",
    re.IGNORECASE,
)
# Strip a leading dash answer prefix off a segment head.
_ANSWER_PREFIX = re.compile(r"^[–—-]\s*")
# After splitting at "?" boundaries, the final chunk may be a trailing
# statement (no "?" in it) — like "Property has 23 dwellings..." after
# the last question. We want to KEEP that as a separate atom because
# it carries scope.
_DECLARATIVE_TAIL_BOUNDARY = re.compile(
    # Split at sentence-final "." followed by "  " or start of capital
    # word that begins a new declarative fact. Conservative — requires
    # ≥2 whitespace OR newline so we don't split mid-paragraph prose.
    r"(?<=[.!)])[\s\xa0]+(?=[A-Z][a-z])"
)
_NBSP_REWRITE = re.compile(r"(?:&nbsp;|&#160;| )+")










def _decode_html_entities(text: str) -> str:
    """Replace common HTML entities + non-breaking-space runs with a
    single regular space.

    The PDF text extractor occasionally surfaces &nbsp; / \\u00A0 when
    a copy-paste source preserved literal HTML; downstream tokenizers
    then split tokens on the wrong boundary or fail to lower-case
    them correctly. Normalize at the splitter boundary so every
    downstream pipeline sees clean prose.
    """
    if not text:
        return text
    return _NBSP_REWRITE.sub(" ", text)


def _peel_next_question(segment: str) -> tuple[str, str]:
    """Split a "<answer> <next-question?>" segment into (answer, question).

    The next question is taken to begin at the first Wh-word / auxiliary
    verb; everything before it is the previous question's answer. When no
    opener is found (or it sits at the head) the whole segment is the
    question and the answer is empty.
    """
    m = _QUESTION_START.search(segment)
    if m is None or m.start() == 0:
        return "", segment.strip()
    return segment[: m.start()].strip(), segment[m.start() :].strip()




def _looks_like_form_answer(line: str) -> bool:
    """A short value that answers a form question — 'Yes', 'No', 'New Tablet',
    '8' — not itself a question, instruction, or long sentence."""
    s = (line or "").strip()
    if not s or len(s.split()) > 6 or s.endswith("?"):
        return False
    return not _FORM_INTERROG_RE.match(s) and not _FORM_INSTRUCTION_RE.match(s)
























def _split_form_qa_blob(text: str) -> list[str]:
    """Split a free-form Q&A interview transcript into per-question atoms.

    Triggers when the paragraph contains ≥3 "?" marks AND no formal
    ``Q\\d./A\\d.`` markers (those go through ``_split_qa_blob``).

    Each emitted chunk is ``"<question?> <answer>"`` so downstream
    typed-atom classification can fire on the full Q+A context.

    A trailing declarative tail ("Property has 23 dwellings...") is
    further split on ``". "`` sentence boundaries so each scope-bearing
    fact becomes its own atom.
    """
    if not text:
        return []
    cleaned = _decode_html_entities(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.count("?") < 3:
        return [cleaned]
    if _QA_PAIR_PROBE.search(cleaned):
        # Defer to the strict Q\d./A\d. splitter — don't double-handle.
        return [cleaned]
    segments = [s for s in _FORM_QA_SEGMENT.split(cleaned) if s.strip()]
    if len(segments) < 2:
        return [cleaned]
    chunks: list[str] = []
    # segments[0] is the first question verbatim (it ends in "?").
    current_q: str | None = segments[0].strip()
    for seg in segments[1:]:
        seg = _ANSWER_PREFIX.sub("", seg.strip())
        if seg.endswith("?"):
            answer, next_q = _peel_next_question(seg)
            chunks.append(f"{current_q} {answer}".strip() if current_q else seg)
            current_q = next_q
        else:
            # Final segment: the last answer + a declarative tail. Emit
            # the open question on its own, then split the tail so each
            # scope-bearing fact becomes its own atom.
            if current_q:
                chunks.append(current_q)
                current_q = None
            chunks.extend(
                p.strip()
                for p in _DECLARATIVE_TAIL_BOUNDARY.split(seg)
                if p.strip()
            )
    if current_q:
        chunks.append(current_q)
    return [c for c in chunks if c]


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


_FORM_FIELD_MARKERS = _FORM_FIELD_STRONG_MARKERS + _FORM_FIELD_WEAK_MARKERS


# A run of "key = value" lines (field-mapping / blob-metadata blocks) the PDF
# extractor glued into one paragraph. Match on '=' ONLY (not ':') so prose with
# colons ("Classification: Mock...") is never split. Key is one identifier token;
# the value runs until the next "identifier =" or end of text.
_PDF_KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]{1,40})\s*=\s*(.+?)(?=\s+[A-Za-z_][A-Za-z0-9_]{1,40}\s*=|$)"
)
# Per-line form: one whole line that is exactly "identifier = value". Used when
# the original line structure survives, so a value is bounded by its real line
# instead of running to the next '=' (which swallows trailing prose lines).
_PDF_KV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]{1,40})\s*=\s*(\S.*)$")
# An all-caps section heading (>=3 caps words) glued mid-paragraph, followed by
# sentence-case body. Avoids single acronyms (PDF/DOCX/CRM) — needs a multi-word
# run. The (?<=[a-z.]) keeps it from firing at the very start of the text.
_PDF_EMBEDDED_HEADING_RE = re.compile(
    r"(?<=[a-z.])\s+([A-Z][A-Z]{2,}(?:\s+(?:AND|OR|OF|&|[A-Z][A-Z-]{2,})){1,5})\s+(?=[A-Z][a-z])"
)


def _split_pdf_kv_blob(
    text: str, lines: list[str] | None = None
) -> tuple[list[str], list[str]] | None:
    """Split a glued 'key = value' paragraph into one chunk per pair.

    Returns ``(kv_chunks, prose_lines)`` when the block is a metadata run
    (≥3 pairs), else ``None``.

    When the original per-line structure survives (``lines``), each pair is
    bounded by its real line, so a trailing non-``key=value`` line (e.g.
    ``contentType should preserve …``) is returned in ``prose_lines`` instead
    of being swallowed into the last value. Without ``lines`` (layout-pipeline
    pages) it falls back to the flattened-text regex.
    """
    if lines:
        kv: list[str] = []
        prose: list[str] = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                continue
            m = _PDF_KV_LINE_RE.match(stripped)
            if m:
                kv.append(f"{m.group(1).strip()} = {m.group(2).strip()}")
            else:
                prose.append(stripped)
        if len(kv) >= 3:
            return kv, prose
        return None
    pairs = list(_PDF_KV_RE.finditer(text))
    if len(pairs) < 3:
        return None
    return [f"{m.group(1).strip()} = {m.group(2).strip()}" for m in pairs], []


def _split_pdf_embedded_heading(text: str) -> tuple[str, str, str] | None:
    """(before, HEADING, after) when an all-caps heading is glued mid-text, else None."""
    m = _PDF_EMBEDDED_HEADING_RE.search(text)
    if not m:
        return None
    before, heading, after = text[: m.start()].strip(), m.group(1).strip(), text[m.end():].strip()
    if len(before) < 10 or len(after) < 10:
        return None
    # The all-caps run may be only PART of a mixed-case heading: 'HME NEXO' is the
    # caps prefix of the section header 'HME NEXO Box Install'. When the matched
    # heading is followed by Title-Case words and THEN a question, those words are
    # the rest of the header (a form section title sits above its Q&A) — pull them
    # into the heading so it stays whole and 'Box Install' isn't orphaned into the
    # answer. Gated on a following question, so ordinary prose isn't over-captured.
    aw = after.split()
    take = 0
    while take < 4 and take < len(aw):
        w = aw[take]
        if _FORM_INTERROG_RE.match(w) or w.endswith("?"):
            break
        if re.fullmatch(r"[A-Z][a-z][\w/&-]*", w):
            take += 1
        else:
            break
    if take and take < len(aw) and (_FORM_INTERROG_RE.match(aw[take]) or aw[take].endswith("?")):
        heading = (heading + " " + " ".join(aw[:take])).strip()
        after = " ".join(aw[take:]).strip()
    return before, heading, after













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
    # A "Label: value" line ("Currency: United States Dollars (USD).") is a
    # real fact, not a bare bullet-fragment label ("Cost Proposal") — keep it.
    if re.search(r"\w:\s+\S", stripped):
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
        # Universal OCR fallback: for text-poor pages where the
        # structured pipeline produced no atoms, try Tesseract OCR
        # and emit scope_item atoms from recovered words. Handles
        # phone-photographed contracts, scan-only PDFs, and image-
        # only marketing PDFs. No-op when Tesseract isn't installed
        # or every page already has body text.
        try:
            atoms.extend(
                _ocr_fallback_atoms(
                    path=path,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    parser_version=self.parser_version,
                    already_emitted=atoms,
                )
            )
        except Exception:  # pragma: no cover - never fail the parse
            pass
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
        # Schematic legend-first pre-pass (PR5).  Only fires when a
        # legend is actually parsed in the document or when the
        # domain pack declares detection targets; otherwise leaves
        # the output stream untouched so RFP-only PDFs are unchanged.
        #
        # We no longer swallow exceptions silently. Failures here
        # used to be invisible: legacy tests stayed green while every
        # schematic atom quietly disappeared. Instead, route any
        # exception into a structured schematic_warning so the
        # operator can see what went wrong and fix it.
        schematic_atoms: list[EvidenceAtom] = []
        schematic_derived: list[dict[str, Any]] = []
        try:
            schematic_atoms, schematic_derived = _run_schematic_pre_pass(
                project_id=project_id,
                artifact_id=artifact_id,
                path=path,
                parser_version=self.parser_version,
                domain_pack=domain_pack,
            )
        except Exception as exc:
            import traceback as _tb

            schematic_atoms = [
                _build_schematic_prepass_failure_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    path=path,
                    parser_version=self.parser_version,
                    exception=exc,
                    traceback=_tb.format_exc(),
                )
            ]
            schematic_derived = []
        if schematic_atoms:
            atoms.extend(schematic_atoms)

        # Site-roster fitz fallback: when the structured-doc pipeline
        # didn't expose any site-roster tables (e.g. reportlab-rendered
        # PDFs whose cells the column-heuristic doesn't recognize),
        # call fitz.find_tables() directly. Any table that smells like
        # a site roster gets fed through site_roster_extractor and
        # emitted as physical_site atoms. This is additive — it does
        # not deduplicate against the structured path because we want
        # at-least-one path to fire.
        try:
            existing_site_ids = {
                (a.value or {}).get("site_id")
                for a in atoms
                if isinstance(a.value, dict) and a.value.get("kind") == "physical_site"
            }
            existing_site_ids.discard(None)
            atoms.extend(
                _fitz_site_roster_fallback(
                    pdf_path=path,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    parser_version=self.parser_version,
                    already_emitted=existing_site_ids,
                )
            )
        except Exception:  # pragma: no cover — never fail the parse
            pass

        # Generic table fallback: when the structured pipeline did NOT
        # emit any tables but fitz can find them (reportlab-generated
        # tables, scanned-then-OCR'd grids, etc.), emit one
        # table_row atom per row so part_numbers / quantities /
        # money inside cells are captured. No-op when the structured
        # pipeline already surfaced tables.
        try:
            atoms.extend(
                _fitz_generic_table_fallback(
                    pdf_path=path,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    parser_version=self.parser_version,
                    structured_doc=structured_doc,
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
        derived_files.extend(schematic_derived)

        # Per-image markers: every embedded image XObject becomes a located
        # marker so a figure / diagram / scanned region can't silently vanish.
        # region_ref (``page{n}/image{xref}``) matches the content census so
        # the region reconciles as MARKED rather than UNCOVERED. Additive and
        # never fatal — a real OCR/vision atom for the same region wins.
        try:
            atoms.extend(
                _pdf_image_markers(
                    path=path,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    parser_version=self.parser_version,
                )
            )
        except Exception:  # pragma: no cover — never fail the parse
            pass

        atoms = _repair_clipped_site_ids(atoms)
        atoms = _weak_label_prose_line_items(atoms)
        atoms = _drop_repeated_header_bands(atoms)
        atoms = _strip_placeholder_table_labels(atoms)
        atoms = _drop_table_header_as_data_rows(atoms)
        atoms = _demote_decorative_dates(atoms)
        atoms = _collapse_toc_atoms(atoms)
        atoms = _fold_answers_into_questions(atoms)
        atoms = _fold_photo_requests_into_images(atoms)

        # Universal hybrid summary+transcript rewrite: when filename/title/
        # content signals a meeting-summary front matter + diarized transcript
        # body, re-atomize transcript pages as per-turn atoms (deal vs
        # conversation_meta). No deal-specific hardcodes — structural only.
        try:
            from app.core.hybrid_summary_transcript import rewrite_hybrid_pdf_atoms

            atoms = rewrite_hybrid_pdf_atoms(
                atoms=atoms,
                structured_doc=structured_doc,
                filename=path.name,
                project_id=project_id,
                artifact_id=artifact_id,
                parser_version=self.parser_version,
            )
        except Exception as exc:  # pragma: no cover — never fail the parse
            import logging

            logging.getLogger(__name__).warning(
                "hybrid_summary_transcript rewrite failed for %s: %s",
                path.name,
                exc,
                exc_info=True,
            )

        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files,
        )


# ──────────────────────── public helpers ─────────────────────────────────
















_TOC_LEADER_RE = re.compile(r"\.{4,}|…{2,}|(?:\.\s){4,}")


def _collapse_toc_atoms(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Collapse a Table-of-Contents atom to a compact marker. A TOC page is rows
    of 'Section Title .......... <page>' dotted leaders — navigation furniture, not
    deal facts, and its entries just duplicate the real section headings that
    appear later in the document. Left alone it becomes a glued 2000-char atom
    ('1.0 SCOPE …… 1  2.0 REQUIREMENTS …… 2 …'). Detect >=3 dotted-leader runs and
    replace the body with a one-line marker (kept, not dropped — no silent loss)."""
    out: list[EvidenceAtom] = []
    for a in atoms:
        t = a.raw_text or ""
        if len(t) > 200 and len(_TOC_LEADER_RE.findall(t)) >= 3:
            marker = ("[Table of contents — document navigation (section titles -> "
                      "page numbers); not deal content. The listed sections are "
                      "captured as their own atoms where they occur.]")
            try:
                a = a.model_copy(update={
                    "raw_text": marker, "normalized_text": marker.lower(),
                    "review_flags": list(a.review_flags or []) + ["table_of_contents"],
                })
            except Exception:
                pass
        out.append(a)
    return out


_ANSWER_PREFIX_RE = re.compile(r"^(?:answer|ans|response|reply|a)\s*[:.\)]", re.I)
_QUESTION_HEAD_RE = re.compile(
    r"^(?:\d+[.\)]\s*)?(?:please\b|could\b|can\b|will\b|would\b|should\b|is\b|are\b|"
    r"do\b|does\b|did\b|how\b|what\b|when\b|where\b|why\b|which\b|who\b|confirm\b|"
    r"as\s+mentioned\b|provide\b|clarify\b)", re.I,
)
_ANSWER_BLOCK_RULE = None


def _answer_block_rule():
    """SemanticRule: does this atom read as the ANSWER/response to the question
    above it (in a numbered Q&A / RFI list), so the two should fold into one Q&A
    fact? Answer-ness is a meaning judgment; the explicit 'Answer:' prefix is the
    offline net."""
    global _ANSWER_BLOCK_RULE
    if _ANSWER_BLOCK_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _ANSWER_BLOCK_RULE = SemanticRule(
            name="qa_answer_block",
            positives=[
                "Answer: These are all wall phones. Single CAT6 cable with wall jack.",
                "Answer: The Site Survey showed (1) Quad, (5) wall phone & (13) Duplex outlets.",
                "Industry Standard.", "There is sufficient space.",
                "Both. Can reuse but must provide where needed.",
                "A 66 block at BDF location and 24-port patch panel inside cabinet.",
            ],
            negatives=[
                "Please confirm whether the 2 wall drops are single-cable drops or another configuration.",
                "As mentioned in the Statement of Work, 2.2.10, cable tray is required.",
                "The contractor shall furnish all materials and labor.",
                "Upload 4 photos of the unit installed.", "Tablet Install",
            ],
            threshold=0.55,
            lexical_fallback=lambda s: bool(_ANSWER_PREFIX_RE.match((s or "").strip())),
        )
    return _ANSWER_BLOCK_RULE


def _is_answer_block(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if _ANSWER_PREFIX_RE.match(s):   # explicit 'Answer:' — always an answer
        return True
    try:
        return _answer_block_rule().fires(s)
    except Exception:
        return False


def _fold_answers_into_questions(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Pair an answer atom with the question atom directly above it into ONE Q&A
    atom — a numbered Q&A / RFI list ('1. … Could I get clarification?' + 'Answer:
    …'). Split, the question and its answer lose their linkage. Answer-ness is an
    embedding judgment ('Answer:' prefix is the offline net); the pairing is
    positional (an answer belongs to the question right above it, same page).
    Conservative: only folds into a preceding QUESTION-ish atom on the same page,
    never into another answer or a heading — so nothing is mis-merged or lost."""
    def _page(a: EvidenceAtom):
        try:
            return (a.source_refs[0].locator or {}).get("page") if a.source_refs else None
        except Exception:
            return None

    def _is_questionish(t: str) -> bool:
        return bool(t) and (t.endswith("?") or "?" in t or bool(_QUESTION_HEAD_RE.match(t)))

    def _is_declarative_answer(t: str) -> bool:
        # A response with NO explicit marker: a declarative line that is not itself
        # a question, instruction, photo request, label, or bullet. Lets a Q&A pair
        # when the answer doesn't say 'Answer:' ('… needed?' -> 'Industry Standard.')
        # — works offline, where the answer EMBEDDING is unavailable.
        if not t or len(t.split()) < 2 or t.endswith("?") or "?" in t:
            return False
        if (_QUESTION_HEAD_RE.match(t) or _FORM_INSTRUCTION_RE.match(t)
                or _is_photo_request(t) or _is_value_field_label(t)):
            return False
        return True

    out: list[EvidenceAtom] = []
    for a in atoms:
        txt = (a.raw_text or "").strip()
        v = a.value if isinstance(a.value, dict) else {}
        if (out and txt and v.get("kind") != "image_marker"
                and "binary_region_marker" not in (a.review_flags or [])):
            prev = out[-1]
            ptxt = (prev.raw_text or "").strip()
            pv = prev.value if isinstance(prev.value, dict) else {}
            explicit = bool(_ANSWER_PREFIX_RE.match(txt))
            # Three ways to recognise the answer below a question:
            #   1. explicit 'Answer:'   — folds into any non-answer prev (reliable
            #      everywhere; the prev is the question or its wrapped tail);
            #   2. embedding answer      — needs a clearly question-ish prev (online);
            #   3. positional, no marker — a STRONG question (ends '?') followed by a
            #      declarative line is the answer (offline-safe; no colon needed).
            prev_mergeable = (
                _page(prev) == _page(a)
                and not _ANSWER_PREFIX_RE.match(ptxt)
                and pv.get("kind") != "image_marker"
                and "binary_region_marker" not in (prev.review_flags or [])
                and len(ptxt) + len(txt) < 3900
            )
            do_merge = prev_mergeable and (
                explicit                                            # 1. 'Answer:' -> any non-answer prev
                or (_is_answer_block(txt) and _is_questionish(ptxt))  # 2. embedding answer (online)
                or (ptxt.endswith("?") and _is_declarative_answer(txt))  # 3. positional, no marker
            )
            if do_merge:
                merged = f"{ptxt}  {txt}"
                try:
                    out[-1] = prev.model_copy(update={
                        "raw_text": merged, "normalized_text": normalize_text(merged),
                    })
                    continue
                except Exception:
                    pass
        out.append(a)
    return out


def _fold_photo_requests_into_images(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Fold a photo-request text atom ('Upload 4 Photos of the Nexeo …') into the
    images it asks for: the request's 'answer' IS those photos, so it belongs as
    the images' linkage reference, not a duplicate scope_item.

    The images are extracted in a separate pass and appended at the END of the
    atom list, so they sink below all text. Here we MOVE each matching image to
    its request's position (reading order — under the right section header) and
    give it the request's section_path, then drop the request text. Images with
    no local request (a multi-page request's continuation photos) stay where they
    are. Safe: only folds when a captioned image exists, so nothing is lost."""
    img_by_caption: dict[str, list[EvidenceAtom]] = {}
    for a in atoms:
        v = a.value if isinstance(a.value, dict) else {}
        if v.get("kind") == "image_marker":
            cap = (v.get("expected_content") or "").strip().lower()
            if cap:
                img_by_caption.setdefault(cap, []).append(a)
    if not img_by_caption:
        return atoms

    def _with_section(img: EvidenceAtom, req: EvidenceAtom) -> EvidenceAtom:
        # Give the image the request's section_path so it files under the same
        # header (e.g. 'Tablet Install') instead of floating section-less.
        try:
            rsec = (req.source_refs[0].locator or {}).get("section_path") if req.source_refs else None
            if rsec and img.source_refs:
                loc = dict(img.source_refs[0].locator or {})
                loc["section_path"] = rsec
                newref = img.source_refs[0].model_copy(update={"locator": loc})
                return img.model_copy(update={"source_refs": [newref, *img.source_refs[1:]]})
        except Exception:
            pass
        return img

    placed: set[int] = set()
    out: list[EvidenceAtom] = []
    for a in atoms:
        v = a.value if isinstance(a.value, dict) else {}
        txt = (a.raw_text or "").strip()
        if v.get("kind") != "image_marker" and _is_photo_request(txt):
            low = txt.lower()
            cap = next((c for c in img_by_caption if low.startswith(c) or c in low), None)
            if cap:
                # place the matching image(s) HERE, in the request's slot
                for img in img_by_caption[cap]:
                    if id(img) not in placed:
                        placed.add(id(img))
                        out.append(_with_section(img, a))
                continue  # the request itself folds into those images
        if v.get("kind") == "image_marker" and id(a) in placed:
            continue  # already moved up to its request slot — drop the trailing dup
        out.append(a)
    return out
















def _last_block_if_paragraph(page: dict[str, Any]) -> dict[str, Any] | None:
    """The page's final content block, but only if it's a paragraph."""
    for sec in reversed(page.get("sections") or []):
        blocks = sec.get("blocks") or []
        if not blocks:
            continue
        last = blocks[-1]
        return last if last.get("kind") == "paragraph" else None
    return None


def _stitch_cross_page_continuations(pages: list[dict[str, Any]]) -> None:
    """Re-join a paragraph that the PDF wrapped across a page boundary.

    When a page's last paragraph ends mid-sentence (no terminal punctuation)
    and the next page opens with a lowercase continuation paragraph *before*
    any heading, the sentence was split by the page break. Splice the
    continuation back onto the previous block so it doesn't orphan into a
    fragment atom (e.g. "and payment schedule."). Mutates ``pages`` in place.
    """
    for i in range(len(pages) - 1):
        prev_block = _last_block_if_paragraph(pages[i])
        if prev_block is None:
            continue
        ptext = (prev_block.get("text") or "").rstrip()
        if not ptext or ptext[-1] in ".!?:":
            continue  # previous page ended a sentence cleanly — no wrap
        nxt_sections = pages[i + 1].get("sections") or []
        if not nxt_sections:
            continue
        first_sec = nxt_sections[0]
        if (first_sec.get("heading") or "").strip():
            continue  # a heading precedes the text — a new section, not a wrap
        nblocks = first_sec.get("blocks") or []
        if not nblocks or nblocks[0].get("kind") != "paragraph":
            continue
        cont = nblocks[0]
        ctext = (cont.get("text") or "").strip()
        if not ctext or not ctext[0].islower():
            continue  # continuation must start lowercase (mid-sentence)
        prev_block["text"] = f"{ptext} {ctext}".strip()
        prev_lines = prev_block.get("lines")
        if isinstance(prev_lines, list):
            prev_lines.extend(cont.get("lines") or [ctext])
        del nblocks[0]
        if not nblocks:
            del nxt_sections[0]


def _carry_cross_page_section_headings(pages: list[dict[str, Any]]) -> None:
    """Root a page's heading-less opening content under the clause it continues.

    A numbered RFP clause often spills across a page break: item 8 "Contract
    Award and Interpretations" continues onto the next page, item 23 "Proposal
    Format" sub-items land on the following page, etc. The next page then opens
    with a *heading-less* section (content before any heading) that would
    otherwise float at the document root (wrong section_path). Carry the last
    real section heading across the page boundary so that opening content
    inherits its true clause. Blocks stay on their own page (provenance intact);
    only the section heading is inherited. Mutates ``pages`` in place. Runs
    AFTER ``_stitch_cross_page_continuations`` (which first merges mid-sentence
    wraps), so this only attributes genuine full-paragraph continuations.
    """
    last_heading = ""
    for page in pages:
        secs = page.get("sections") or []
        if not secs:
            continue
        first = secs[0]
        if last_heading and not (first.get("heading") or "").strip() \
                and (first.get("blocks") or []):
            first["heading"] = last_heading
        for s in secs:
            h = (s.get("heading") or "").strip()
            if h:
                last_heading = h


_COL_PLACEHOLDER_LABEL = re.compile(r"\bcol_\d+:\s*")

# A line whose ENTIRE content is a date (cover/letterhead "May 14,2026",
# "June 5, 2026", "5/14/2026", "2026-05-14").
_BARE_DATE_LINE = re.compile(
    r"^\s*(?:[A-Z][a-z]+\.?\s+\d{1,2},?\s*\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2})\s*$"
)


# operative_date / section_title rules now live in the shared registry
# (app/core/semantic_rules.py) so docx/xlsx can pull the same rule + examples.
def _operative_date_rule():
    from app.core.semantic_rules import operative_date_rule
    return operative_date_rule()


_OPERATIVE_DATE = None
_SECTION_TITLE = None


def _section_title_rule():
    from app.core.semantic_rules import section_title_rule
    return section_title_rule()


def _is_section_title(text: str) -> bool:
    global _SECTION_TITLE
    if not text:
        return False
    if _SECTION_TITLE is None:
        _SECTION_TITLE = _section_title_rule()
    return _SECTION_TITLE.fires(text.strip())


def _demote_decorative_dates(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Re-type a bare cover/letterhead date from a scope atom to document
    metadata — it's *when the doc was made*, not deal scope. Operative dates
    (deadlines, timeline/award dates) are KEPT untouched. Conservative: a bare
    date is only demoted when the operative-date rule does NOT fire on its
    context, so a real deadline is never silently relegated. No drop — the date
    survives as ``deal_metadata`` (kind=document_date)."""
    global _OPERATIVE_DATE
    out: list[EvidenceAtom] = []
    for a in atoms:
        rt = (getattr(a, "raw_text", "") or "").strip()
        if a.atom_type in (AtomType.assumption, AtomType.scope_item) and _BARE_DATE_LINE.match(rt):
            refs = getattr(a, "source_refs", None) or []
            loc = getattr(refs[0], "locator", None) if refs else None
            sec = " ".join(loc.get("section_path") or []) if isinstance(loc, dict) else ""
            if _OPERATIVE_DATE is None:
                _OPERATIVE_DATE = _operative_date_rule()
            # Judge the SECTION the date lives under, not the bare digits — a
            # date in a "Projected Timeline / Key Dates / Submission Deadlines"
            # section is operative; one under a generic cover heading is not.
            # (The digits carry no meaning and only dilute the embedding.) Fall
            # back to the date's own text when it has no section.
            if not _OPERATIVE_DATE.fires((sec or rt).strip()):
                try:
                    a = a.model_copy(update={
                        "atom_type": AtomType.deal_metadata,
                        "value": {**(a.value or {}), "kind": "document_date", "date": rt},
                        "review_flags": sorted(set((a.review_flags or []) + ["demoted_decorative_date"])),
                    })
                except Exception:
                    pass
        out.append(a)
    return out


def _strip_placeholder_table_labels(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Drop ``col_N:`` placeholder labels from a table row that the extractor
    couldn't header.

    When pdfplumber finds no header row, the table emitter falls back to
    ``col_0``, ``col_3`` … placeholders, so a row renders as the meaningless
    ``col_0: Price | col_3: 50%``. The labels are parser-generated noise, not
    document text, so strip them and keep the faithful values (``Price | 50%``).
    Only fires when EVERY labeled segment is a placeholder — a table that found
    real headers (``Site: ATL | Qty: 5``) is left untouched. (Regex is the right
    tool here: ``col_N:`` is a fixed sentinel WE emit, not a fuzzy judgment.)"""
    out: list[EvidenceAtom] = []
    for a in atoms:
        rt = getattr(a, "raw_text", "") or ""
        if "col_" in rt and _COL_PLACEHOLDER_LABEL.search(rt):
            segs = [s.strip() for s in rt.split(" | ") if s.strip()]
            labeled = [s for s in segs if ": " in s]
            if labeled and all(_COL_PLACEHOLDER_LABEL.match(s) for s in labeled):
                new = " | ".join(_COL_PLACEHOLDER_LABEL.sub("", s) for s in segs).strip()
                if new and new != rt:
                    try:
                        a = a.model_copy(update={
                            "raw_text": new, "normalized_text": normalize_text(new)})
                    except Exception:
                        pass
        out.append(a)
    return out


def _drop_table_header_as_data_rows(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Drop a table's HEADER row that leaked back in as a data atom.

    When the layout pipeline can't separate a small table's header from its body
    it emits the header line as a row, so the column labels render as their own
    "key: value" atom with key == value: ``Type: Type | Qty.: Qty.`` (anyWAIR
    page 5). Every cell being ``X: X`` is an unmistakable header-as-data signal —
    a real data row pairs a label with a DIFFERENT value — so the row is pure
    duplication of the column headers and carries no fact. Conservative: fires
    only when ALL labeled cells are key==value."""
    out: list[EvidenceAtom] = []
    for a in atoms:
        rt = (getattr(a, "raw_text", "") or "").strip()
        cells = [s.strip() for s in rt.split(" | ") if s.strip()]
        labeled = [c for c in cells if ": " in c]
        if cells and labeled and len(labeled) == len(cells):
            def _kv_equal(cell: str) -> bool:
                k, _, v = cell.partition(": ")
                return k.strip().casefold() == v.strip().casefold() and bool(k.strip())
            if all(_kv_equal(c) for c in cells):
                continue  # header row leaked as data — drop

            # A "key: value" cell whose value OPENS with a lowercase coordinating
            # joiner ("Qty.: and Install") is a column-split heading, not data — a
            # real tabular value never starts with "and"/"or". This is the layout
            # pipeline slicing a Title-Case sub-heading ("Access Control Rough and
            # Install") across the table's column boundary. Drop the fragment row.
            def _split_header_fragment(cell: str) -> bool:
                _, _, v = cell.partition(": ")
                vw = v.strip().split()
                return bool(vw) and vw[0].lower() in {"and", "or", "of", "the", "to"} \
                    and len(vw) <= 3
            if any(_split_header_fragment(c) for c in labeled):
                continue
        out.append(a)
    return out


def _drop_repeated_header_bands(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Drop a running header/footer band that repeats verbatim across pages.

    A short line ("000087 - OPTBOT … | HubSpot 60355665326") printed at the top
    of every page is furniture, not per-page content. Keep the first occurrence
    (it carries the deal id once) and drop the repeats.
    """
    from collections import defaultdict

    def _page(a: EvidenceAtom) -> Any:
        refs = getattr(a, "source_refs", None) or []
        if refs:
            loc = getattr(refs[0], "locator", None)
            if isinstance(loc, dict):
                return loc.get("page")
        return None

    pages_by_text: dict[str, set] = defaultdict(set)
    for a in atoms:
        txt = (getattr(a, "raw_text", "") or "").strip()
        if 0 < len(txt) <= 90:
            pages_by_text[txt].add(_page(a))
    repeated = {
        t for t, pgs in pages_by_text.items()
        if len([p for p in pgs if p is not None]) >= 2
    }
    if not repeated:
        return atoms
    seen: set[str] = set()
    out: list[EvidenceAtom] = []
    for a in atoms:
        txt = (getattr(a, "raw_text", "") or "").strip()
        if txt in repeated:
            if txt in seen:
                continue
            seen.add(txt)
        out.append(a)
    return out


def _weak_label_prose_line_items(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Demote a vendor_line_item that the classifier pinned on a prose block.

    ``vendor_line_item`` is a structured table-row type (SKU / qty / price). When
    section/keyword heuristics stamp it on a prose paragraph or bullet (a pricing
    section makes "Taxes: Excluded" look commercial), it's a brittle guess, not a
    real line item — flag weak_label and lower confidence so the type head
    re-decides instead of trusting it.
    """
    for a in atoms:
        if getattr(getattr(a, "atom_type", None), "value", None) != "vendor_line_item":
            continue
        v = a.value if isinstance(a.value, dict) else {}
        if v.get("kind") not in ("paragraph", "bullet", "bullet_item"):
            continue  # a genuine table row keeps full confidence
        flags = list(getattr(a, "review_flags", None) or [])
        if "weak_label" not in flags:
            flags.append("weak_label")
            try:
                a.review_flags = flags
            except Exception:  # pragma: no cover — frozen atom
                pass
        try:
            if a.confidence and a.confidence > 0.5:
                a.confidence = 0.45
        except Exception:  # pragma: no cover
            pass
    return atoms


def _repair_clipped_site_ids(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """Reconcile a site_id that a narrow table column clipped against its full
    form found elsewhere in the document.

    A ruled-table roster row can lose the tail of its site_id when the PDF
    visually clips the cell ("ATL-WEST-02" -> "ATL-WEST-0"), while the full id
    still appears in prose. The prose site scanner then emits a bare id-only
    physical_site for the full id — a phantom duplicate of the real row.

    When a bare id-mention is a short (≤2 char) extension of a full roster
    row's clipped id, repair the roster row to the full id and drop the
    phantom. Cross-doc joins key on site_id, so the clipped id would otherwise
    fail to match the same site in the other documents.
    """
    sites = [
        a for a in atoms
        if isinstance(a.value, dict) and a.value.get("kind") == "physical_site"
    ]
    if len(sites) < 2:
        return atoms

    def _is_bare(a: EvidenceAtom) -> bool:
        v = a.value
        sid = v.get("site_id") or ""
        return bool(sid) and (v.get("name") or "") == sid and (v.get("facility_name") or "") == sid

    drop_ids: set[int] = set()
    for bare in sites:
        if id(bare) in drop_ids or not _is_bare(bare):
            continue
        full = bare.value.get("site_id") or ""
        if len(full) < 8:
            continue
        for row in sites:
            if row is bare or _is_bare(row):
                continue
            clipped = row.value.get("site_id") or ""
            if (
                len(clipped) >= 8
                and full.startswith(clipped)
                and 0 < len(full) - len(clipped) <= 2
            ):
                row.value["site_id"] = full
                row.value["id"] = full
                # Repair the rendered identity too (the address / access cells
                # stay clipped — the source PDF has no full form of those).
                rt = getattr(row, "raw_text", None)
                if isinstance(rt, str) and f"site_id: {clipped}" in rt:
                    try:
                        row.raw_text = rt.replace(f"site_id: {clipped}", f"site_id: {full}", 1)
                    except Exception:  # pragma: no cover — frozen atom
                        pass
                drop_ids.add(id(bare))
                break
    if not drop_ids:
        return atoms
    return [a for a in atoms if id(a) not in drop_ids]


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

    # A2 large-PDF safety net: when a PDF is bigger than the soft
    # cap, process only the first ``MAX_PAGES_LARGE_PDF`` pages and
    # add a warning to the metadata so the PM sees the partial-
    # parse explicitly. Prevents OOM on 500MB+ scanned dumps while
    # still surfacing actionable evidence from the first chunk.
    # Tunable via env vars so on H100/large-RAM hosts the caller
    # can lift the limits.
    import os as _os
    LARGE_PDF_SOFT_CAP_MB = float(_os.environ.get("PARSER_OS_PDF_SOFT_CAP_MB", "50"))
    MAX_PAGES_LARGE_PDF = int(_os.environ.get("PARSER_OS_PDF_MAX_PAGES", "200"))
    try:
        pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    except OSError:
        pdf_size_mb = 0.0
    is_large_pdf = pdf_size_mb > LARGE_PDF_SOFT_CAP_MB
    if is_large_pdf:
        warning = (
            f"[A2 large-PDF guard] {pdf_path.name} is "
            f"{pdf_size_mb:.0f} MB > {LARGE_PDF_SOFT_CAP_MB:.0f} MB; "
            f"processing only the first {MAX_PAGES_LARGE_PDF} pages. "
            f"Set PARSER_OS_PDF_MAX_PAGES or PARSER_OS_PDF_SOFT_CAP_MB "
            f"to lift this limit."
        )
        if warning not in seen_metadata:
            seen_metadata.add(warning)
            document_metadata.append(warning)

    # P2.1: pre-scan the PDF for per-page text length so we can fast-path
    # low-text pages (scanned drawings, image-only floor plans) without
    # running the heavyweight layout-detection pipeline on them.
    # Insanity-perf: ALSO collect the actual page text so a text-rich
    # page can be parsed via a lightweight prose splitter without ever
    # touching the layout pipeline (which costs 5–10 s/page).
    page_text_lengths: list[int] = []
    page_texts: list[str] = []
    page_image_counts: list[int] = []
    with fitz.open(str(pdf_path)) as doc:
        # Encrypted PDF detection — explicit signal for PM_HANDOFF so
        # the file gets routed to manual unlock rather than silently
        # producing 0 atoms. ``doc.needs_pass`` is True when the PDF
        # is password-protected and the open call didn't supply one.
        if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
            encrypt_msg = (
                f"[Encrypted PDF — {pdf_path.name} is password-protected. "
                f"Manual unlock required: open in Acrobat / Preview, supply "
                f"the password, save as an unencrypted copy, then re-attach "
                f"to the intake. parser-os marks this file as needs_review "
                f"and emits 0 evidence atoms until unlocked.]"
            )
            document_metadata.append(encrypt_msg)
            # Skip the rest of the parse — return an empty page list
            # so the rest of the pipeline degrades gracefully via A6.
            return {
                "schema_version": STRUCTURED_SCHEMA_VERSION,
                "source": {
                    "filename": pdf_path.name,
                    "page_count": 0,
                    "encrypted": True,
                },
                "document": {"title": None, "metadata": document_metadata},
                "pages": [],
            }
        full_page_count = len(doc)
        # A2: cap the working page_count for large PDFs but
        # remember the original so the metadata can report it.
        page_count = (
            min(full_page_count, MAX_PAGES_LARGE_PDF)
            if is_large_pdf
            else full_page_count
        )
        if is_large_pdf and full_page_count > MAX_PAGES_LARGE_PDF:
            skipped_msg = (
                f"[A2 large-PDF guard] truncated {full_page_count} pages "
                f"→ first {MAX_PAGES_LARGE_PDF}; "
                f"{full_page_count - MAX_PAGES_LARGE_PDF} pages skipped."
            )
            if skipped_msg not in seen_metadata:
                seen_metadata.add(skipped_msg)
                document_metadata.append(skipped_msg)
        for page_idx in range(page_count):
            try:
                page_text = doc[page_idx].get_text("text") or ""
            except Exception:  # pragma: no cover — bad page shouldn't kill compile
                page_text = ""
            page_texts.append(page_text)
            page_text_lengths.append(len(page_text.strip()))
            try:
                page_image_counts.append(len(doc[page_idx].get_images()))
            except Exception:  # pragma: no cover — bad page shouldn't kill compile
                page_image_counts.append(0)

    # Page bucketing thresholds:
    #   < LOW_TEXT_PAGE_THRESHOLD       → marker page only (scanned)
    #   >= TEXT_RICH_PAGE_THRESHOLD     → text-only fast path
    #   else                             → heavyweight layout pipeline
    LOW_TEXT_PAGE_THRESHOLD = 80
    TEXT_RICH_PAGE_THRESHOLD = 1200

    def _build_low_text_page(page_index: int) -> dict[str, Any]:
        # A genuinely sparse text page (a trailing line, a short final page) is
        # NOT a scanned image. Only treat low-text as scanned when the page
        # actually carries visual objects (images / vector drawings) worth an
        # OCR/vision pass. Otherwise emit the little text as a normal page — no
        # bogus "scanned image / needs_extractor" marker on near-empty pages.
        try:
            with fitz.open(str(pdf_path)) as _vd:
                _pg = _vd[page_index]
                _has_raster = bool(_pg.get_images(full=True))
                _has_visual = _has_raster or bool(_pg.get_drawings())
        except Exception:
            # fail-safe: if we can't inspect the page, keep scanned-page behavior
            # AND keep the review warning (don't suppress when we're unsure).
            _has_raster = False
            _has_visual = True
        if not _has_visual:
            return _build_text_rich_page(page_index)
        # Low-text WITH visuals = likely scanned. Try the OCR chain
        # (PyMuPDF Tesseract → pytesseract → easyocr → Ollama vision).
        # If any backend recovers text, treat the page as text-rich.
        # If nothing fires, keep the marker so PM_HANDOFF surfaces it
        # under "Files requiring manual review".
        try:
            from app.parsers._ocr_chain import ocr_pdf_page
            # Re-open the doc inside the OCR scope to keep fitz state
            # isolated from the outer page-loop. PyMuPDF docs / pages
            # are not thread-safe.
            with fitz.open(str(pdf_path)) as _doc:
                ocr_result = ocr_pdf_page(_doc[page_index])
        except Exception as exc:
            ocr_result = {
                "text": "",
                "backend": "",
                "notes": [f"ocr_chain crashed: {type(exc).__name__}"],
            }
        if (ocr_result.get("text") or "").strip():
            # Promote the page through the text-rich path using the
            # OCR'd text. Stash the page text in our cache so any
            # downstream consumer that re-reads ``page_texts`` sees
            # the OCR result.
            page_texts[page_index] = ocr_result["text"]
            page_text_lengths[page_index] = len(ocr_result["text"].strip())
            page_dict = _build_text_rich_page(page_index)
            page_dict.setdefault("metadata", []).insert(
                0,
                f"[OCR-recovered via {ocr_result.get('backend','')} — "
                f"text layer was missing; treat as scanned-source evidence]",
            )
            return page_dict
        # OCR recovered nothing. If the page carries RASTER images, those are
        # captured by the separate image-marker pass (each saved, captioned with
        # its 'Upload N photos…' request, and flagged 'awaiting OCR / vision') —
        # so a page-level "no text, needs manual review" atom would be redundant
        # AND misleading (the page's content is the photos, not lost). Suppress it;
        # the image markers carry the page + the vision signal. Only a page with NO
        # raster images (vector-only / truly unreadable, nothing else captured it)
        # still gets the manual-review marker.
        if _has_raster:
            # Emit NOTHING for the page itself — its content is the embedded
            # image(s), which the image-marker pass already captures (saved,
            # captioned with their photo-request, flagged 'awaiting OCR / vision').
            # A page-level marker here would be a redundant, misleading duplicate.
            return {"page": page_index, "title": None, "metadata": [],
                    "outline": [], "sections": []}
        return {
            "page": page_index,
            "title": None,
            "metadata": [
                f"[low-text page (≤{LOW_TEXT_PAGE_THRESHOLD} chars) "
                "— likely scanned image; OCR chain "
                f"({', '.join(ocr_result.get('notes', []) or ['no backend reachable'])}) "
                "produced no text. PM_HANDOFF will surface this page for manual review.]"
            ],
            "outline": [],
            "sections": [],
        }

    def _build_text_rich_page(page_index: int) -> dict[str, Any]:
        # A text-rich page can still contain a ruled table (e.g. an
        # authoritative site roster drawn with vector lines). The prose
        # splitter has zero table awareness, so without this step the
        # roster's columns bleed into the running text and shatter into
        # ghost atoms ("site id", "mon-fri 07:00", "optbot facil"...).
        # Recover any line-ruled tables first, strip their bbox regions
        # from the text the splitter sees, then re-attach them as proper
        # table blocks so the roster fast-path in _atoms_for_block emits
        # clean physical_site atoms instead.
        ruled_blocks, ruled_bboxes = _extract_ruled_tables(pdf_path, page_index)
        # Also recover UNRULED column tables (date|event timelines, Factor|Weight
        # grids, key|value blocks) the prose splitter would otherwise scramble.
        # BUT skip whitespace-column extraction on a questionnaire page: a form's
        # question and its answer sit in different x-columns, so the column
        # detector would grab them as a 2-col table and fragment the Q&A — the
        # Q&A regrouper below owns those pages instead.
        # Also skip on a bulleted-list slide: its bullets sit in a column beside a
        # diagram, so the column detector grabs them as a 2-col table and scrambles
        # the BOM list ('• 24 x FS-1024E' -> 'x x x FS-1024E ...'). The bullet
        # splitter below keeps them in clean reading order.
        if (_is_questionnaire_page(page_texts[page_index])
                or _page_is_form(page_texts[page_index].splitlines())
                or _is_bulleted_list_page(page_texts[page_index])):
            col_blocks, col_bboxes = [], []
        else:
            col_blocks, col_bboxes = _extract_column_tables(pdf_path, page_index)
        table_blocks, table_bboxes = _merge_table_extractions(
            ruled_blocks, ruled_bboxes, col_blocks, col_bboxes
        )
        prose_text = page_texts[page_index]
        if table_blocks:
            stripped = _page_prose_excluding_tables(
                pdf_path, page_index, table_bboxes
            )
            if stripped is not None:
                prose_text = stripped

        # Diarized transcript pages (meeting summary + full transcript exports)
        # must not go through form Q&A regroup / form_field tagging — speaker
        # stamps look like Title-Case headers and rhetorical "?" trips the form
        # gate. Hybrid rewrite owns turn atomization for these pages.
        try:
            from app.core.hybrid_summary_transcript import count_speaker_timestamp_hits

            _diarized = count_speaker_timestamp_hits(prose_text or "") >= 2
        except Exception:
            _diarized = False

        # On a questionnaire / field-report page, join each question with its
        # answer ("Have you installed the NEXEO Box?\nYes" -> one Q&A unit) before
        # the prose splitter sees it, so the answer isn't buried in a blob with the
        # next instruction. No-op on non-form pages (gated on >=2 question lines).
        if not _diarized:
            prose_text = _regroup_form_qa(prose_text)

        sections = _text_rich_sections(prose_text)
        if table_blocks:
            _place_tables_in_sections(
                pdf_path, page_index, sections, table_blocks, table_bboxes
            )
        # A field-report / form page has no prose title — its content is Q&A and
        # labelled fields. Tag its blocks so the short-fragment drops in
        # _atoms_for_block (len<10, title-case-label = 'fragment') don't eat real
        # form values ('Signature', 'Managers Name', 'Diedra Kennedy'), and skip
        # page-title detection (it was grabbing the first question / a field label
        # as the 'title' and stripping it).
        if (not _diarized) and _page_is_form(prose_text.splitlines()):
            for _sec in sections:
                for _blk in _sec.get("blocks") or []:
                    _blk["form_field"] = True
        # The page title becomes the document's main section. A heading is
        # structure, not a fact — drop its content block so the title isn't
        # both the section root AND its own atom. (_detect_text_title rejects
        # questions / field values, so a form page's first Q isn't taken as
        # the title — while a real title like 'Burger King HME Install' is kept.)
        # On diarized transcript pages, prefer a stable chrome title over the
        # first spoken fragment ("something intune related…").
        if _diarized:
            page_title = None
            for _probe in (prose_text or "").splitlines()[:8]:
                _p = _probe.strip()
                if re.match(
                    r"^(?:meeting\s+summary(?:\s+and\s+full\s+transcripts?)?|"
                    r"full\s+transcripts?|executive\s+summary)\b",
                    _p,
                    re.I,
                ):
                    page_title = _p.split(":")[0].strip()
                    break
        else:
            page_title = _detect_text_title(prose_text)
        if page_title:
            _strip_title_block(sections, page_title)
        _stamp_section_and_block_ids(sections, page_index)
        metadata = [
            "[text-rich page — heavyweight layout pipeline skipped; "
            "prose extracted via lightweight text splitter]"
        ]
        if table_blocks:
            metadata.append(
                f"[recovered {len(table_blocks)} line-ruled table(s) via "
                "PyMuPDF; their cell regions were removed from the prose "
                "stream to prevent column-bleed ghost atoms]"
            )
        return {
            "page": page_index,
            "title": page_title,
            "metadata": metadata,
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
        # A DocuSign / e-sign "Certificate of Completion" audit page is pure
        # signature-trail boilerplate — collapse it to ONE marker instead of
        # minting dozens of junk atoms (anyWAIR: ~46). Checked first so it
        # applies regardless of the page's text length.
        if _is_signature_certificate_page(page_texts[page_index]):
            return {
                "page": page_index,
                "title": None,
                "metadata": ["[signature certificate page — e-signature audit "
                             "trail; collapsed to one boilerplate marker]"],
                "outline": [],
                "sections": [{
                    "heading": "", "level": 2, "subsections": [],
                    "blocks": [{
                        "kind": "note",
                        "text": ("[Signature certificate of completion — e-signature "
                                 "audit trail (signer events, envelope id, delivery "
                                 "timestamps). Boilerplate; no deal content.]"),
                    }],
                }],
            }
        if page_text_lengths[page_index] < LOW_TEXT_PAGE_THRESHOLD:
            return _build_low_text_page(page_index)
        if page_text_lengths[page_index] >= TEXT_RICH_PAGE_THRESHOLD:
            return _build_text_rich_page(page_index)
        # Mid-band (80–1199 chars): the heavyweight layout pipeline is only
        # needed when a page carries visual structure (figures, scanned
        # drawings). A pure-text page with no raster images — e.g. a short
        # continuation page with a signature roster — is parsed correctly and
        # far more cheaply by the prose splitter, which also recovers any
        # line-ruled tables. The 1200 cutoff is a perf guard, not correctness.
        #
        # ALSO take the prose path when the page is clearly multi-paragraph prose
        # even if it carries an image (a cover letter with a letterhead logo):
        # the heavyweight pipeline MERGES the blank-line-separated paragraphs into
        # one glued mega-atom (solicitation + award + rejection rights + contact
        # all in one), whereas the prose splitter respects the paragraph breaks.
        # The letterhead image is still captured by the separate image-marker pass.
        if page_image_counts[page_index] == 0 \
                or _is_multi_paragraph_prose(page_texts[page_index]) \
                or _is_bulleted_list_page(page_texts[page_index]) \
                or _is_form_page(page_texts[page_index]):
            # A questionnaire / field-report page (questions OR a 'Upload N photos'
            # request) is form text, NOT a visual layout — the heavyweight pipeline
            # scrambles it: it splits a section header ('HME NEXO Box Install' ->
            # heading 'HME NEXO' + body 'Box Install'), glues Q&A, and reorders by
            # geometry. Send it to the text/form path even though it carries photos
            # (captured separately by the image-marker pass).
            return _build_text_rich_page(page_index)
        hv = _build_heavyweight_page(page_index)
        # Coverage backstop — a clean text layer must NEVER be silently dropped.
        # The heavyweight layout pipeline can drop most/all of a page's text on a
        # form or field-report page that carries embedded photos (the Burger King
        # HME form: Name / Store # / Site # / arrival-departure / the NEXEO Q&A all
        # vanished while 30 photos were marked). This is OUTCOME-based, not a
        # predictive router: compare what heavyweight KEPT against the page's real
        # text layer and, if it kept too little (< 45%), re-run through the prose/
        # form splitter. Photos are still captured by the separate image-marker
        # pass. Outcome-checking beats any page-type guesser — a page heavyweight
        # genuinely handles (anyWAIR geometry tables) keeps ~all its text, so it is
        # never rerouted; a page it mangles is always caught, even ones unseen.
        src_len = page_text_lengths[page_index]
        if src_len >= LOW_TEXT_PAGE_THRESHOLD \
                and _page_captured_text_len(hv) < 0.45 * src_len:
            return _build_text_rich_page(page_index)
        return hv

    # Embed every candidate line for the WHOLE document in one round trip
    # before the page loop. The heading/lead-in/label semantic rules embed one
    # line per call; on this corpus that is ~900 distinct lines per PDF, i.e.
    # ~900 serial round trips (~38 min against the remote embedder) for a parse
    # that should take seconds. Prewarming fills the shared embedding cache in a
    # single request, so each rule call below is a local hit. Pure optimisation:
    # no decision changes, and it silently no-ops when the embedder is offline.
    try:
        from app.core.semantic_rules import prewarm as _prewarm_semantic
        _prewarm_semantic(
            line for text in page_texts for line in (text or "").splitlines()
        )
    except Exception:  # pragma: no cover — a warm-up must never fail a parse
        pass

    # NOTE: PyMuPDF is NOT thread-safe — running the page loop on a
    # ThreadPoolExecutor crashes with SIGSEGV inside libmupdf. The
    # text-rich fast path (above) is the dominant speedup; pages
    # that still hit the heavyweight pipeline run serially. A future
    # optimization could spawn a process per page (multiprocessing
    # with each worker opening its own fitz doc), at the cost of
    # ~2 s per-fork startup on macOS.
    pages: list[dict[str, Any]] = [_build_one_page(i) for i in range(page_count)]
    _stitch_cross_page_continuations(pages)
    _carry_cross_page_section_headings(pages)

    # Aggregate document title + metadata across pages (in order).
    for p in pages:
        # Pick the document title from the FIRST page that yields one, then stop
        # embedding — the semantic section-vs-title check only needs to run until
        # a title is found, not on every page (keeps the embedder calls bounded).
        if not document_title:
            page_title = p.get("title")
            # A SECTION heading ("INTRODUCTION", "General Conditions") is NOT the
            # document title — it's a sibling of every other section. Crowning it
            # as the title force-nests all other sections beneath it
            # ("INTRODUCTION > General Conditions"). Reject it, and fall back to
            # the page's first real (non-section) heading — the cover org name.
            if page_title and _is_section_title(page_title):
                page_title = None
            if not page_title:
                for s in (p.get("sections") or []):
                    h = (s.get("heading") or "").strip()
                    if h and len(h.split()) >= 2 and not _is_section_title(h):
                        page_title = h
                        break
            if page_title:
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
    from app.core.longpath import long_write_text
    derived_dir = derived_dir_for(pdf_path)
    out = derived_dir / STRUCTURED_FILENAME
    # long_write_text: deep _rerun paths exceed Windows MAX_PATH (260); a plain
    # write_text here throws WinError 206 after the parse, silently dropping the
    # whole PDF's output. No-op cost on POSIX.
    long_write_text(out, json.dumps(structured_doc, indent=2, ensure_ascii=False))
    return out


def write_structured_markdown(pdf_path: Path, structured_doc: dict[str, Any]) -> Path:
    """Persist the LLM-friendly markdown projection next to the JSON.

    The markdown mirrors the JSON structure 1:1 with stable HTML anchors
    (``<a id="blk_..."></a>`` / ``<a id="sec_..."></a>``) so an LLM can
    cite a region by anchor and a UI can scroll to the same place.
    """
    from app.core.longpath import long_write_text
    derived_dir = derived_dir_for(pdf_path)
    out = derived_dir / STRUCTURED_MARKDOWN_FILENAME
    long_write_text(out, structured_doc_to_markdown(structured_doc))
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
    # Root every atom's section_path at the document's main section (its
    # title), so a sub-heading renders as a path ("<main section> > <heading>")
    # rather than a flat sibling label. Add it for DISPLAY only (after the
    # atom is classified) — a broad title like "Commercial, Pricing &
    # Acceptance" must not feed the section-rule classifier, or it would stamp
    # vendor_line_item on every paragraph in the doc (electrical test specs
    # included). Classification runs on the sub-headings; the root is prepended
    # to the locator afterward.
    #
    # Also stamp ``block_index`` (monotonic emit order) and carry the title on
    # ``lead_in``. The compiler/envelope sort atoms by id, so audit UIs restore
    # reading order via block_index — same role as DOCX ``para_order`` / email
    # ``line_start``. Title-on-lead_in matches DOCX Title→section root and email
    # framing connective tissue (meeting-summary / hybrid summary pages used to
    # skip this and only got section headers).
    doc_title = (structured_doc.get("document") or {}).get("title")
    emit_seq = [0]

    def _root_atom(atom: EvidenceAtom) -> EvidenceAtom:
        if not getattr(atom, "source_refs", None):
            return atom
        loc = getattr(atom.source_refs[0], "locator", None)
        if not isinstance(loc, dict):
            return atom
        # Stable reading-order index for post-compile id-sort recovery.
        # Stamp both block_index (PDF/DOCX audit key) and line_start (email
        # audit key) so every consumer restores reading order the same way.
        if "block_index" not in loc:
            loc["block_index"] = emit_seq[0]
        if "line_start" not in loc:
            loc["line_start"] = emit_seq[0]
            loc["line_end"] = emit_seq[0]
        emit_seq[0] += 1
        if not doc_title:
            return atom
        sp = loc.get("section_path") or []
        if not sp or sp[0] != doc_title:
            loc["section_path"] = [doc_title, *sp]
        lead = [str(x).strip() for x in (loc.get("lead_in") or []) if str(x or "").strip()]
        if doc_title not in lead:
            lead = [doc_title, *lead]
            loc["lead_in"] = lead
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            vlead = [str(x).strip() for x in (val.get("lead_in") or []) if str(x or "").strip()]
            if doc_title not in vlead:
                vlead = [doc_title, *vlead]
                val["lead_in"] = vlead
            # Keep intro as the deepest connective (section header), not the
            # document title alone — title still rides on lead_in / section_path.
            if vlead:
                val["intro"] = vlead[-1] if len(vlead) > 1 else vlead[0]
        return atom

    for page in structured_doc.get("pages", []):
        page_index = int(page.get("page", 0))
        sections = page.get("sections", []) or []
        for _atom in _atoms_for_sections(
            sections=sections,
            section_path=[],
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        ):
            yield _root_atom(_atom)
        # Metadata-fallback path: when the structured extractor was
        # unable to assemble any sections from the page (heading
        # classifier misfired on short-paragraph PDFs, weak heading
        # styling, scanned/rasterized documents), the page's body
        # content ends up classified as ``metadata`` and is otherwise
        # silently dropped. Emit one scope_item atom per metadata
        # line as a fallback so content like a date roster or a
        # one-paragraph SLA isn't completely invisible to the
        # downstream compiler.
        if not sections:
            page_metadata = page.get("metadata") or []
            for meta_index, meta_text in enumerate(page_metadata):
                text = (str(meta_text or "")).strip()
                if not text or len(text) < 6:
                    continue
                # Parser pipeline-diagnostic breadcrumbs (how the page was
                # parsed) are not deal facts and not coverage gaps — never emit
                # them as atoms. Genuine coverage markers ("awaiting OCR",
                # low-text/visual) DO stay; they're needs_extractor signals.
                _low = text.lower()
                if ("heavyweight layout pipeline" in _low
                        or ("recovered" in _low and "line-ruled table" in _low)):
                    continue
                if _looks_like_form_field(text) or _looks_like_page_footer(text):
                    continue
                # Apply the same text-pattern classifier the normal
                # paragraph path uses — without it, SLA / decision /
                # constraint / risk shapes that arrive via the
                # fallback all get the default scope_item label.
                atom_type, authority = _classify_text_block(
                    text=text, section_path=[], kind="paragraph"
                )
                yield _root_atom(
                    _make_atom(
                        text=text,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=filename,
                        parser_version=parser_version,
                        atom_type=atom_type,
                        authority_class=authority,
                        confidence=DEFAULT_BLOCK_CONFIDENCE,
                        locator={
                            "page": page_index,
                            "block_kind": "metadata_fallback",
                            "meta_index": meta_index,
                        },
                        value={"kind": "paragraph", "fallback": "page_metadata"},
                    )
                )


_ITEM_ENUM_RE = re.compile(r"^\s*\(?(?:[a-zA-Z]|\d{1,2}|[ivxIVX]{1,4})\)?[.)]\s+(?=\S)")


def _split_enumerated_item_title(lines: Any) -> tuple[str, str] | None:
    """A lettered/numbered list item often packs a short TITLE on its first line
    ("a. Table of Contents") above its body ("Responses shall include …"). Return
    (title, body) so the title can ride as [intro:] connective tissue instead of
    being glued into the requirement text. STRUCTURAL (enumeration + line
    geometry) — the same kind of signal as bullet detection, not a fuzzy meaning
    judgment — so no embedding needed. Fires only on a real title (short, label-
    like first line) over a real body (multi-word prose)."""
    if not lines or len(lines) < 2:
        return None

    def _txt(l: Any) -> str:
        return (l if isinstance(l, str) else (l.get("text") if isinstance(l, dict) else "")) or ""

    l0 = _txt(lines[0]).strip()
    m = _ITEM_ENUM_RE.match(l0)
    if not m:
        return None
    title = l0[m.end():].strip().rstrip(":")
    tw = title.split()
    # A title is a short label, not a full sentence (no sentence-ending period).
    if not (1 <= len(tw) <= 8) or title.endswith("."):
        return None
    # If the NEXT line is itself enumerated ("b. …"), this is a FLAT list of
    # sibling items (a. main office / b. guidance office / …), not a title over a
    # body — don't mis-title the first item.
    if _ITEM_ENUM_RE.match(_txt(lines[1]).strip()):
        return None
    body = " ".join(t for t in (_txt(x).strip() for x in lines[1:]) if t).strip()
    if len(body.split()) < 4:   # need a real body beneath the title
        return None
    return title, body


def _pdf_is_framing_lead_in(text: str) -> bool:
    """Is a paragraph a FRAMING lead-in ("The following are the General
    Conditions…", "Services include:") that introduces the block(s) after it?
    Connective tissue lifted onto what it governs as [intro:] context. Delegates
    to the shared registry rule (same one docx uses), so PDF + docx stay in
    lockstep and there's no cross-parser import."""
    try:
        from app.core.semantic_rules import is_framing_lead_in
        return bool(is_framing_lead_in(text))
    except Exception:
        return text.rstrip().endswith(":")


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
    from app.core.semantic_rules import lead_in_rule as _lead_in_rule_fn
    for section in sections:
        heading = section.get("heading")
        path = section_path + ([heading] if heading else [])
        blocks = section.get("blocks", []) or []

        def _emit(b, lead=None):
            yield from _atoms_for_block(
                block=b, section_path=path, page_index=page_index,
                project_id=project_id, artifact_id=artifact_id,
                filename=filename, parser_version=parser_version, lead_in=lead,
            )

        def _enum(b) -> bool:
            return b.get("kind") == "paragraph" and bool(
                _ITEM_ENUM_RE.match((b.get("text") or "").strip()))

        pending: tuple[dict[str, Any], str] | None = None  # single un-consumed lead-in
        sticky: str | None = None                          # intro governing an enumerated list
        pending_meeting_section: str | None = None         # trailing header before a bullet list
        for idx, block in enumerate(blocks):
            nxt = blocks[idx + 1] if idx + 1 < len(blocks) else None
            # Stamp a trailing meeting header from the previous glued paragraph
            # onto this bullet list so Key Decisions bullets nest correctly.
            if pending_meeting_section and block.get("kind") == "bullet_list":
                if not block.get("meeting_section"):
                    block = {**block, "meeting_section": pending_meeting_section}
                pending_meeting_section = None
            elif pending_meeting_section and block.get("kind") != "bullet_list":
                pending_meeting_section = None
            if pending is not None:
                # Previous block was a framing lead-in — lift onto THIS block.
                yield from _emit(block, [pending[1]])
                pending = None
                continue
            # A sticky list-intro rides onto every enumerated item it governs.
            if sticky and _enum(block):
                yield from _emit(block, [sticky])
                continue
            if sticky and not _enum(block):
                sticky = None  # the enumerated list ended
            btext = (block.get("text") or "").strip() if block.get("kind") == "paragraph" else ""
            # Glued meeting-summary paragraph sitting above a bullet list:
            # split Action Items bullets here and carry Key Decisions onto nxt.
            if btext and nxt is not None and nxt.get("kind") == "bullet_list":
                glued = _split_glued_meeting_summary_paragraph(btext)
                if glued is not None:
                    glued_blocks, trailing = glued
                    for sub in glued_blocks:
                        sub_path = path + (
                            [sub["meeting_section"]] if sub.get("meeting_section") else []
                        )
                        yield from _atoms_for_block(
                            block=sub,
                            section_path=sub_path,
                            page_index=page_index,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=filename,
                            parser_version=parser_version,
                            lead_in=(
                                [sub["meeting_section"]]
                                if sub.get("meeting_section")
                                else None
                            ),
                        )
                    if trailing:
                        pending_meeting_section = trailing
                    continue
            # A (possibly long) framing intro directly above an enumerated list
            # governs the WHOLE list ("The intent … all responses follow the same
            # format" over a–g). SEMANTIC, not regex: the 'next block is
            # enumerated' STRUCTURE finds the candidate and bounds the embed; the
            # lead-in rule confirms it reads as an intro. Keep it standalone (it
            # carries content) AND lift it as [intro:] onto each item below.
            if (btext and nxt is not None and _enum(nxt) and len(btext.split()) >= 6
                    and _lead_in_rule_fn().fires(btext)):
                sticky = btext
                yield from _emit(block)
                continue
            if btext and _pdf_is_framing_lead_in(btext):
                pending = (block, btext)   # short lead-in: attach to the next block
                continue
            yield from _emit(block)
        if pending is not None:
            # Nothing followed it — a lead-in with no governed block is just a
            # statement; emit it normally (never drop it).
            yield from _emit(pending[0])
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
    lead_in: list[str] | None = None,
) -> Iterator[EvidenceAtom]:
    kind = block.get("kind")
    block_id = block.get("id") or stable_id("blk", page_index, kind or "?", id(block))
    base_locator: dict[str, Any] = {
        "page": page_index,
        "block_id": block_id,
        "block_kind": kind,
        "section_path": section_path,
        # Governing lead-in lifted from a preceding framing sentence ("The
        # following are…") — connective-tissue context the heads see as
        # [intro:], same as the docx path.
        "lead_in": lead_in or [],
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
        # A form/field-report page's values are legitimately short ('Signature',
        # 'Yes', 'Diedra Kennedy') and read as Title-Case labels — they are real
        # captured content, so the proposal-checklist fragment/length drops below
        # must NOT apply to them. But even on a form page, a bare bullet glyph
        # ('•', 'o', '▪') or a page-number label ('Page 1') is furniture, never an
        # atom — keep dropping those.
        _is_form_field = bool(block.get("form_field"))
        if not text or len(re.sub(r"[^0-9A-Za-z]", "", text)) < 2:
            return
        if re.fullmatch(r"(?i)page\s*\d+", text):
            return
        # A paragraph that is only space-separated bare numbers ('32 112 17 14
        # 296') is diagram / table-label noise (counts lifted off a figure), not a
        # fact. A single number can be a real answer, so require >=2.
        if re.fullmatch(r"[\d.,\s]+", text) and len(re.findall(r"\d+", text)) >= 2:
            return
        if len(text) < 10 and not _is_form_field:
            return
        # Bare meeting-summary section headers are connective tissue, not facts.
        # Check BEFORE the Title-Case fragment drop — "Action Items" / "Key
        # Decisions" match the fragment shape and would otherwise vanish
        # without stamping section_path on the bullets beneath.
        if _is_meeting_section_heading_line(text):
            return
        # P1.4: skip pure-title-case bullet-fragment labels like "Cost
        # Proposal", "Project Description", "Addendums".  These come
        # from proposal-format checklists and carry no scope data.
        if not _is_form_field and _looks_like_fragment(text):
            return
        # Repair glued meeting-summary header + checkbox-``I`` bullet blobs
        # (Action Items / Key Decisions) into per-bullet atoms with section_path.
        glued = _split_glued_meeting_summary_paragraph(text)
        if glued is not None:
            glued_blocks, trailing_header = glued
            for sub in glued_blocks:
                sub_path = list(section_path)
                meeting_sec = sub.get("meeting_section")
                if meeting_sec:
                    sub_path = sub_path + [meeting_sec]
                yield from _atoms_for_block(
                    block=sub,
                    section_path=sub_path,
                    page_index=page_index,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    parser_version=parser_version,
                    lead_in=([meeting_sec] if meeting_sec else None) or lead_in,
                )
            # Trailing header with no bullets in this blob — emit a marker
            # paragraph the sections walker cannot see. Instead, if we only
            # had a trailing header, fall through so the original text is not
            # silently dropped when there is no next-block stamp available.
            if glued_blocks:
                return
            if trailing_header:
                # Header-only carry without sibling blocks: drop the chrome.
                return
        # P1.1: when a paragraph contains ≥2 Q\d. / A\d. markers, split
        # it into one atom per Q&A pair so packet anchors don't end up
        # as 2,400-char transcripts.  Single-Q paragraphs and
        # paragraphs without Q&A markers fall through to the original
        # single-atom path below.
        qa_chunks = _split_qa_blob(text)
        # v57 P1.1b: free-form Q&A (no Q1./A1. markers, but many "?")
        # also need splitting — discovery notes / interview transcripts.
        if len(qa_chunks) < 2:
            form_qa_chunks = _split_form_qa_blob(text)
            if len(form_qa_chunks) >= 2:
                qa_chunks = form_qa_chunks
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
        # A field-mapping / metadata block is a run of "key = value" lines the
        # extractor glued into one paragraph — split into one deal_metadata atom
        # per key so the head gets clean facts, not a 9-field blob.
        kv_split = _split_pdf_kv_blob(text, lines=block.get("lines"))
        if kv_split:
            kv_chunks, kv_prose = kv_split
            _, kv_auth = _classify_text_block(text=text, section_path=section_path, kind="paragraph")
            for kv_idx, chunk in enumerate(kv_chunks):
                yield _make_atom(
                    text=chunk,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    parser_version=parser_version,
                    atom_type=AtomType.deal_metadata,
                    authority_class=kv_auth,
                    confidence=DEFAULT_BLOCK_CONFIDENCE,
                    locator={**base_locator, "kv_index": kv_idx, "kv_count": len(kv_chunks)},
                    value={"kind": "key_value"},
                )
            # A trailing prose line that isn't a key=value pair (a note glued
            # under the metadata run) becomes its own classified atom instead of
            # being absorbed into the last value.
            for prose in kv_prose:
                if len(prose) < 10:
                    continue
                p_type, p_auth = _classify_text_block(
                    text=prose, section_path=section_path, kind="paragraph"
                )
                yield _make_atom(
                    text=prose,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    parser_version=parser_version,
                    atom_type=p_type,
                    authority_class=p_auth,
                    confidence=DEFAULT_BLOCK_CONFIDENCE,
                    locator=base_locator,
                    value={"kind": "paragraph"},
                )
            return
        # An all-caps section heading the extractor missed, glued mid-paragraph:
        # split so its body becomes its own sectioned atom instead of trailing
        # the previous paragraph.
        heading_split = _split_pdf_embedded_heading(text)
        if heading_split:
            before, heading, after = heading_split
            for chunk, sp in ((before, section_path), (after, section_path + [heading])):
                at, auth = _classify_text_block(text=chunk, section_path=sp, kind="paragraph")
                yield _make_atom(
                    text=chunk,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    parser_version=parser_version,
                    atom_type=at,
                    authority_class=auth,
                    confidence=DEFAULT_BLOCK_CONFIDENCE,
                    locator={**base_locator, "section_path": sp},
                    value={"kind": "paragraph"},
                )
            return
        # A lettered/numbered list item with a short TITLE on its first line
        # ("a. Table of Contents") over a body — lift the title as [intro:]
        # connective tissue so the atom is the requirement, not "a. Table of
        # Contents Responses shall include …" with the label buried inside.
        item_split = _split_enumerated_item_title(block.get("lines"))
        if item_split:
            it_title, it_body = item_split
            it_type, it_auth = _classify_text_block(
                text=it_body, section_path=section_path, kind="paragraph"
            )
            yield _make_atom(
                text=it_body,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=it_type,
                authority_class=it_auth,
                confidence=DEFAULT_BLOCK_CONFIDENCE,
                locator={**base_locator, "lead_in": (base_locator.get("lead_in") or []) + [it_title]},
                value={"kind": "paragraph", "item_title": it_title},
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
        meeting_sec = (block.get("meeting_section") or "").strip() or None
        # Prefer an explicit meeting_section stamp; else derive from path.
        list_sec, sec_header = _meeting_section_connective(
            section_path + ([meeting_sec] if meeting_sec else [])
        )
        if meeting_sec and not sec_header:
            sec_header = meeting_sec
            list_sec = list_sec or _meeting_section_connective([meeting_sec])[0]
        effective_lead = list(lead_in or [])
        if sec_header and sec_header not in effective_lead:
            effective_lead = effective_lead + [sec_header]
        bullet_section_path = section_path + (
            [meeting_sec] if meeting_sec and meeting_sec not in section_path else []
        )
        if intro:
            intro_type, intro_auth = _classify_text_block(
                text=intro, section_path=bullet_section_path, kind="bullet_intro"
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
                locator={**base_locator, "bullet_role": "intro", "section_path": bullet_section_path},
                value={"kind": "bullet_intro"},
            )
        # Bullets inherit the intro's classification context; for many docs
        # the intro line ("Partner(s) must:") is what colors every child.
        if intro:
            bullet_section_path = bullet_section_path + [intro]
        for index, item in enumerate(block.get("items", []) or []):
            yield from _atoms_for_bullet(
                item=item,
                depth=1,
                path_indices=[index],
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                base_locator={
                    **base_locator,
                    "section_path": bullet_section_path,
                    "lead_in": effective_lead,
                },
                section_path=bullet_section_path,
                list_section=list_sec,
                section_header=sec_header,
                lead_in=effective_lead,
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

        # Site-roster fast path: when the table looks like a list of
        # physical sites (column headers like Site ID / Facility Name
        # / Street Address, OR surrounding prose declares
        # kind=physical_site), emit one structured ``site`` atom per
        # row carrying all the canonical fields. This bypasses the
        # row-as-prose path that was shattering rosters into junk
        # entity fragments ("site id", "n terminal", "building c").
        try:
            from app.parsers.site_roster_extractor import (
                extract_site_roster,
                looks_like_site_roster,
            )
        except Exception:  # pragma: no cover
            extract_site_roster = None  # type: ignore[assignment]
            looks_like_site_roster = None  # type: ignore[assignment]
        if extract_site_roster is not None and looks_like_site_roster is not None:
            surrounding = " ".join(str(s) for s in (section_path or []))
            try:
                is_roster = looks_like_site_roster(
                    columns=columns, rows=rows, surrounding_text=surrounding
                )
            except Exception:  # pragma: no cover
                is_roster = False
            if is_roster:
                try:
                    roster_rows = extract_site_roster(
                        columns=columns, rows=rows, surrounding_text=surrounding
                    )
                except Exception:  # pragma: no cover
                    roster_rows = []
                for site_row in roster_rows:
                    # The site_id is the canonical key. When absent,
                    # fall back to a slug of the facility_name.
                    canon_id = site_row.site_id or site_row.facility_name or ""
                    if not canon_id:
                        continue
                    site_text = " | ".join(
                        f"{k}: {v}"
                        for k, v in [
                            ("site_id", site_row.site_id),
                            ("facility", site_row.facility_name),
                            ("address", site_row.street_address),
                            ("mdf_idf", site_row.mdf_idf),
                            ("access", site_row.access_window),
                            ("escort", site_row.escort_owner),
                            ("contact", site_row.contact),
                            ("phone", site_row.phone),
                            ("email", site_row.email),
                            ("notes", site_row.notes),
                        ]
                        if v
                    )
                    yield _make_atom(
                        text=site_text or canon_id,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=filename,
                        parser_version=parser_version,
                        # v53.2 ROOT-CAUSE FIX: must be physical_site
                        # (was AtomType.entity — meant ALL site roster
                        # rows from PDF v1 extraction path were invisible
                        # to downstream physical_site filters).
                        atom_type=AtomType.physical_site,
                        authority_class=AuthorityClass.contractual_scope,
                        confidence=site_row.confidence,
                        locator={
                            **base_locator,
                            "row_index": site_row.row_index,
                            "extraction": "site_roster_v1",
                        },
                        value={
                            "kind": "physical_site",
                            "id": site_row.site_id,  # canonical id
                            "site_id": site_row.site_id,
                            "name": site_row.facility_name,
                            "facility_name": site_row.facility_name,
                            "address": site_row.street_address,
                            "street_address": site_row.street_address,
                            "mdf_idf": site_row.mdf_idf,
                            "access_window": site_row.access_window,
                            "escort_owner": site_row.escort_owner,
                            "contact": site_row.contact,
                            "phone": site_row.phone,
                            "email": site_row.email,
                            "city_state": site_row.city_state,
                            # Resolved on the row, never carried on the atom until
                            # live 010300 (HQ with no city/state/ZIP).
                            "city": site_row.city,
                            "state": site_row.state,
                            "zip": site_row.zip,
                            "sqft": site_row.sqft,
                            "occupancy": site_row.occupancy,
                            "notes": site_row.notes,
                            "extras": dict(site_row.extra_fields),
                        },
                    )
                # Site-roster rows are emitted as structured ``entity``
                # atoms above; the legacy table-as-prose path is
                # skipped for this block. We return here to prevent
                # duplicate scope_item atoms covering the same cells.
                return

        truncated_cells = block.get("truncated_cells") or []
        for row_index, row in enumerate(rows):
            row_text = _row_to_text(row)
            if not row_text:
                continue
            row_trunc = (
                truncated_cells[row_index] if row_index < len(truncated_cells) else []
            )
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
            value: dict[str, Any] = {
                "kind": "table_row",
                "columns": columns,
                "cells": dict(row),
            }
            if row_trunc:
                value["truncated_cols"] = list(row_trunc)
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
                value=value,
                review_flags=(["truncated_cell"] if row_trunc else []),
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
    list_section: str | None = None,
    section_header: str | None = None,
    lead_in: list[str] | None = None,
) -> Iterator[EvidenceAtom]:
    text = (item.get("text") or "").strip()
    if text:
        # Strip page-band prefix that some extractors fold into bullet text.
        text = _strip_page_band_prefix(text)
    # NOTE: _looks_like_fragment is deliberately NOT applied here. It exists to
    # drop STANDALONE Title-Case checklist labels emitted as paragraphs ("Cost
    # Proposal", "Project Description"). A bullet is, by construction, an item of
    # a real list ("a. main office", "1. Total Cost of Hardware") — a genuine
    # fact, not a stray fragment — so applying that filter here silently drops
    # legitimate short list items (intern reports: coverage areas + pricing lines).
    if text and len(text) >= 10 and not _looks_like_form_field(text) and not _looks_like_page_footer(text):
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="bullet")
        # Derive meeting-summary connective tissue when caller didn't stamp it.
        ls = list_section
        sh = section_header
        if not ls or not sh:
            derived_ls, derived_sh = _meeting_section_connective(section_path)
            ls = ls or derived_ls
            sh = sh or derived_sh
        effective_lead = list(lead_in or base_locator.get("lead_in") or [])
        if sh and sh not in effective_lead:
            effective_lead = effective_lead + [sh]
        value: dict[str, Any] = {"kind": "bullet", "depth": depth}
        if ls:
            value["list_section"] = ls
        if sh:
            value["section_header"] = sh
        if effective_lead:
            value["lead_in"] = list(effective_lead)
            value["intro"] = effective_lead[0]
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
                "lead_in": effective_lead,
                "section_path": section_path,
            },
            value=value,
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
            list_section=list_section,
            section_header=section_header,
            lead_in=lead_in,
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




# ──────────────────────── classification ─────────────────────────────────


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
    # SLA / managed-services constraint shapes — response/resolution
    # times, uptime percentages, service credits. These appear in
    # every managed-service contract and were previously falling
    # through to scope_item, hiding the operational commitments.
    (re.compile(r"\b(?:response|resolution|repair|restoration|acknowledg(?:e|ement))\s+(?:time\s+)?(?:within|of|<|≤|in)\s+\d+\s*(?:business\s+)?(?:hours?|days?|minutes?)\b", re.I), AtomType.constraint),
    (re.compile(r"\bpriority\s+\d\b.*\b(?:response|resolution)\b", re.I), AtomType.constraint),
    (re.compile(r"\bp[1-4]\b.*\b(?:response|resolution|hours?|days?)\b", re.I), AtomType.constraint),
    (re.compile(r"\b(?:uptime|availability)\b.*?\d+(?:\.\d+)?\s*%", re.I), AtomType.constraint),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+(?:uptime|availability|sla)\b", re.I), AtomType.constraint),
    (re.compile(r"\bservice\s+credits?\s+(?:apply|granted|owed|due)\b", re.I), AtomType.constraint),
    (re.compile(r"\bservice\s+level\s+(?:agreement|objective|commitment)\b", re.I), AtomType.constraint),
    (re.compile(r"\bmean\s+time\s+(?:to|between)\s+(?:repair|restore|failure|recovery)\b", re.I), AtomType.constraint),
    (re.compile(r"\b(?:mttr|mtbf|rpo|rto)\s*[:=]?\s*\d+\s*(?:hours?|days?|minutes?)\b", re.I), AtomType.constraint),
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


_BULLET_LINE_RE = re.compile(
    r"^\s*([-*•·\u2022]|\d+[.)]|[a-z][.)]|[ivx]{1,4}[.)])\s+(.+?)\s*$"
)  # bullets, numbered, AND lettered/roman sub-items ("a. main office", "iv) ...").
# A bullet GLYPH alone on its own line — common in slide / docx->PDF exports,
# where the marker and its text render on separate lines ('•\n74 x FG-30Gs').
# The next content line is the bullet's text. True bullet glyphs only (not 'o'/'-',
# ambiguous with the letter o / a dash), so prose is never mis-read.
_BARE_BULLET_RE = re.compile(r"^\s*[•▪●◦‣]\s*$")
# A list MARKER alone on its line ("a.", "c.", "3.", "iv)") -- the text layer
# of a lettered list often splits marker and item onto separate lines while
# its neighbours stay joined. Live 010300 (signed PSOW): "a. / Switch / b.
# Access Point (Indoor Only) / c. / Camera / d. Firewall" lost Switch and
# Camera and glued "c." onto the Access Point. A sentence never ends on a line
# that is only "c.", so this reading has no rival.
_BARE_ENUM_RE = re.compile(r"^\s*(?:[a-z]|\d{1,2}|[ivx]{1,4})[.)]\s*$")
_HEADING_LINE_RE = re.compile(
    r"^\s*((?:[A-Z0-9][A-Z0-9 &/\-,()]{2,80})|(?:#{1,6}\s+.{2,80}))\s*$"
)


def _detect_truncated_cells(
    columns: list[str], rows: list[dict[str, str]]
) -> list[list[str]]:
    """Per-row list of columns whose cell value is truncated in the source.

    Two signals (the source data itself is cut, not a render clip):
      * an unbalanced opening bracket — "Core data switches (Cisco C930";
      * a fixed-width truncation — the cell is at the column's max length, that
        max is shared by >=2 cells (a fixed character cap), and it ends
        mid-word (lowercase letter). "Fiber patch panels / MPO casse" (30) and
        "Custom millwork / furniture in" (30) alongside shorter complete cells.
    """
    lengths: dict[str, list[int]] = {c: [] for c in columns}
    for r in rows:
        for c in columns:
            v = r.get(c)
            if v:
                lengths[c].append(len(v))
    col_max = {c: (max(ls) if ls else 0) for c, ls in lengths.items()}
    col_max_n = {c: sum(1 for L in lengths[c] if L == col_max[c]) for c in columns}
    out: list[list[str]] = []
    for r in rows:
        trunc: list[str] = []
        for c in columns:
            v = r.get(c)
            if not v:
                continue
            unbalanced = v.count("(") > v.count(")") or v.count("[") > v.count("]")
            fixed_width = (
                len(v) == col_max[c]
                and col_max[c] >= 15
                and col_max_n[c] >= 2
                and v[-1].isalpha()
                and v[-1].islower()
            )
            if unbalanced or fixed_width:
                trunc.append(c)
        out.append(trunc)
    return out


def _extract_ruled_tables(pdf_path: Path, page_index: int) -> tuple[list[dict[str, Any]], list[Any]]:
    """Recover vector-ruled tables on a text-rich page.

    The text-rich fast path (``_build_text_rich_page``) runs a PROSE splitter
    that has no notion of columns. On a page that is actually a TABLE (an
    authoritative site roster, a BOM, a pricing grid), feeding the raw text to
    the prose splitter mashes the columns together line-by-line — every
    attribute-column value (access window, escort owner, MDF/IDF) bleeds into
    prose and later becomes a ghost site/entity.

    PyMuPDF's ``find_tables(strategy="lines")`` reconstructs the grid exactly
    when the PDF has ruling lines (reportlab tables, most exported grids). This
    helper returns:
      * a list of ``kind="table"`` blocks (columns + rows) that the existing
        atom emitter already knows how to turn into clean physical_site /
        table_row atoms via ``looks_like_site_roster`` / ``extract_site_roster``;
      * the list of table bounding boxes so the caller can EXCLUDE the table
        region from the prose splitter (killing the column-bleed at the source).

    Universal: no deal-specific logic. Any ruled table on any text-rich page
    benefits. Returns ``([], [])`` on any failure or when no table is found, so
    the page falls back to byte-identical prose-only behavior.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return [], []
    table_blocks: list[dict[str, Any]] = []
    bboxes: list[Any] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_index]
            try:
                finder = page.find_tables(strategy="lines")
            except Exception:
                return [], []
            tables = list(getattr(finder, "tables", []) or [])
            for table in tables:
                try:
                    extracted = _table_rows_repaired(page, table)
                except Exception:
                    continue
                if not extracted or len(extracted) < 2:
                    continue
                header = [(c or "").strip() for c in extracted[0]]
                ncols = len(header) if header else len(extracted[0])
                columns = [
                    header[i] if i < len(header) and header[i] else f"col_{i}"
                    for i in range(ncols)
                ]
                rows: list[dict[str, str]] = []
                for raw_row in extracted[1:]:
                    if not raw_row:
                        continue
                    cells: dict[str, str] = {}
                    for i, c in enumerate(raw_row):
                        col = columns[i] if i < len(columns) else f"col_{i}"
                        val = " ".join(str(c or "").split()).strip()
                        if val:
                            cells[col] = val
                    if cells:
                        rows.append(cells)
                if not rows:
                    continue
                block: dict[str, Any] = {"kind": "table", "columns": columns, "rows": rows}
                trunc_by_row = _detect_truncated_cells(columns, rows)
                if any(trunc_by_row):
                    block["truncated_cells"] = trunc_by_row
                table_blocks.append(block)
                try:
                    bboxes.append(fitz.Rect(table.bbox))
                except Exception:
                    pass
    except Exception:
        return [], []
    return table_blocks, bboxes




def _merge_table_extractions(
    ruled_blocks: list[dict[str, Any]], ruled_bboxes: list[Any],
    col_blocks: list[dict[str, Any]], col_bboxes: list[Any],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Combine ruled + whitespace-column tables, resolving overlap.

    A whitespace-column table SUPERSEDES an overlapping ruled table only when the
    ruled one is *degenerate* — ≥2 ``col_N`` phantom columns, the signature of
    fitz mis-splitting a header into bogus columns (the ACE ``Factor | Weight``
    grid came out as 6 columns). A CLEAN ruled table (real header names — a site
    roster, a BOM) always wins, so this never clobbers good ruled extraction.
    """
    def _overlaps(a, b) -> bool:
        try:
            inter = a & b
            return inter.is_valid and inter.get_area() > 0.4 * min(
                max(a.get_area(), 1.0), max(b.get_area(), 1.0))
        except Exception:
            return False

    def _degenerate(block: dict[str, Any]) -> bool:
        cols = block.get("columns") or []
        return sum(1 for c in cols if str(c).startswith("col_")) >= 2

    keep_ruled: list[tuple[dict, Any]] = []
    for rb, rbox in zip(ruled_blocks, ruled_bboxes):
        clobber = False
        if _degenerate(rb) and rbox is not None:
            for cbox in col_bboxes:
                if cbox is not None and _overlaps(rbox, cbox):
                    clobber = True
                    break
        if not clobber:
            keep_ruled.append((rb, rbox))

    # drop any column table that overlaps a KEPT (clean) ruled table — the ruled
    # one owns that region.
    keep_cols: list[tuple[dict, Any]] = []
    for cb, cbox in zip(col_blocks, col_bboxes):
        if cbox is not None and any(
            rbox is not None and _overlaps(cbox, rbox) for _, rbox in keep_ruled
        ):
            continue
        keep_cols.append((cb, cbox))

    blocks = [b for b, _ in keep_ruled] + [b for b, _ in keep_cols]
    bboxes = [x for _, x in keep_ruled] + [x for _, x in keep_cols]
    return blocks, bboxes


def _page_prose_excluding_tables(pdf_path: Path, page_index: int, bboxes: list[Any]) -> str | None:
    """Return the page's text with any text falling inside a table bbox removed.

    Used so the prose splitter never sees the table region (which would
    otherwise mash columns into ghost prose). Returns ``None`` on failure so
    the caller falls back to the full page text.
    """
    if not bboxes:
        return None
    try:
        import fitz  # type: ignore[import-not-found]
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_index]
            # LINE-level exclusion. Block-level was too coarse: PyMuPDF lumps a
            # heading + an unruled table + the following prose into ONE block, so
            # an area-overlap test keeps the whole thing and the table text leaks
            # back as duplicate ghost paragraphs. Dropping individual lines whose
            # CENTER sits inside a table bbox removes exactly the table region and
            # nothing else.
            data = page.get_text("dict") or {}
            kept: list[tuple[float, float, str]] = []
            for blk in data.get("blocks", []) or []:
                for ln in blk.get("lines", []) or []:
                    spans = ln.get("spans", []) or []
                    text = "".join(s.get("text", "") for s in spans)
                    if not text.strip():
                        continue
                    lb = ln.get("bbox") or blk.get("bbox") or [0, 0, 0, 0]
                    cx = (lb[0] + lb[2]) / 2.0
                    cy = (lb[1] + lb[3]) / 2.0
                    in_table = False
                    for tb in bboxes:
                        try:
                            if tb.x0 - 1 <= cx <= tb.x1 + 1 and tb.y0 - 1 <= cy <= tb.y1 + 1:
                                in_table = True
                                break
                        except Exception:
                            continue
                    if not in_table:
                        kept.append((round(lb[1], 1), lb[0], text))
            kept.sort(key=lambda t: (t[0], t[1]))
            return "\n".join(t for _, _, t in kept)
    except Exception:
        return None


# A record label: a short "Name:" / "Role:" prefix that opens a list record
# ("Jordan Ames: Approved …", "OPTBOT Business Sponsor: Jordan Ames | …",
# "Grand Total (not-to-exceed for defined scope): USD …"). Allows parens and a
# longer label so a parenthetical qualifier doesn't glue the record onto the
# previous one.
_RECORD_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9 .,'&/()-]{1,60}:\s+\S")


def _split_structured_records(lines: list[str]) -> list[str] | None:
    """Split a block that is an unambiguous record-list into one string per
    record, else ``None``.

    Two structures only (deliberately conservative — prose and imperative
    checklists are left whole):

      * pipe-delimited rows — every line carries ≥2 ``|`` field separators
        (signature / sign-off rosters);
      * label-prefixed records — every record opens with a ``Name:`` / ``Role:``
        label; lines without a label are continuations of the preceding record
        (so a sentence wrapped across lines, or stitched across a page break,
        stays in one record).
    """
    rows = [ln.strip() for ln in lines if ln.strip()]
    if len(rows) < 2:
        return None
    # Pipe-delimited roster.
    if all(ln.count("|") >= 2 for ln in rows):
        return rows
    # A leading "<label>:" intro (ends with a colon, no value of its own) can
    # sit directly above the records — "Approved for SOW incorporation:" over
    # the signature lines. Keep it as its own chunk and split the records below.
    prefix: list[str] = []
    body = rows
    if rows[0].endswith(":") and len(rows) >= 3 and _RECORD_LABEL_RE.match(rows[1]):
        prefix = [rows[0]]
        body = rows[1:]
    # Label-prefixed records.
    if not _RECORD_LABEL_RE.match(body[0]):
        return None
    records: list[str] = []
    for ln in body:
        if _RECORD_LABEL_RE.match(ln):
            records.append(ln)
        elif records:
            records[-1] += " " + ln
        else:
            return None
    label_starts = sum(1 for ln in body if _RECORD_LABEL_RE.match(ln))
    # Need ≥2 real records and every record genuinely label-opened (guards
    # against one stray "Word:" opener in an otherwise prose paragraph).
    if len(records) < 2 or label_starts < len(records):
        return None
    return prefix + records


# A numbered section heading: "1. Authoritative physical site roster …".
_NUMBERED_HEADING_RE = re.compile(r"^\d+\.\s+([A-Z].{2,68})$")


def _numbered_heading(line: str) -> str | None:
    """Heading text of a short numbered section heading, else None.

    Distinguishes "1. Authoritative physical site roster (site_roster v5)" (a
    heading) from a numbered list item, which is a full sentence ending in
    terminal punctuation. Returns the heading without its number.
    """
    m = _NUMBERED_HEADING_RE.match(line.strip())
    if not m:
        return None
    text = m.group(1).strip()
    if text and text[-1] in ".!?,;:":
        return None
    return text


# A list-item opener: a numbered ("3.") or lettered ("a.", "iv)") marker. Used to
# tell a numbered SECTION HEADING (followed by a prose body) apart from a numbered
# LIST ITEM (immediately followed by the next list item — a Pricing Summary line,
# a requirements list). The strict heading regex misses long/lettered siblings.
_LIST_ITEM_OPENER_RE = re.compile(r"^(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})[.)]\s+\S")


_DOTTED_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(\S.*)$")


def _split_dotted_section(line: str) -> tuple[str, str] | None:
    """Split a dotted-decimal SOW/RFP section heading into (heading, body):
    '1.0 SCOPE' -> ('1.0 SCOPE', ''); '2.1 GENERAL REQUIREMENTS. The Contractor
    shall…' -> ('2.1 GENERAL REQUIREMENTS', 'The Contractor shall…'); '2.1.1.1
    Confined Space. N / A' -> ('2.1.1.1 Confined Space', 'N / A').

    Multi-level section numbers (1.0, 2.1, 2.1.1) are missed by the single-level
    numbered/run-on rules, and the title ENDS in a period ('REQUIREMENTS.'), which
    _numbered_heading rejects — so without this every SOW section fell through to
    body content and stayed under a stale carried heading. Boundary: the title is
    the text up to its first sentence period; the rest is the body. Guarded so a
    measurement ('2.5 inch conduit') or a sentence isn't taken as a heading."""
    m = _DOTTED_SECTION_RE.match((line or "").strip())
    if not m:
        return None
    num, rest = m.group(1), m.group(2).strip()
    cut = rest.find(". ")
    if cut == -1:
        head, body = rest.rstrip("."), ""
    else:
        head, body = rest[:cut].strip(), rest[cut + 2:].strip()
    # a heading is a short title that starts capitalised (not 'inch conduit')
    if not head or len(head.split()) > 12 or not head[:1].isupper():
        return None
    return (f"{num} {head}", body)


def _next_content_is_body(lines: list[str], idx: int) -> bool:
    """True when the next non-blank line after ``idx`` is a PROSE body — so a
    numbered line is a section heading over a body, not one entry in a list.

    A numbered line whose next sibling is ANOTHER list item (numbered or lettered)
    is a list member, not a heading: e.g. "1. Total Cost of Hardware / 2. Cost for
    configuration…" under a Pricing Summary, or "7. … coverage in these areas / a.
    main office". A General-Conditions clause ("1. Scope of Work") is instead
    followed by a prose paragraph, so it still promotes.
    """
    for j in range(idx + 1, len(lines)):
        nxt = lines[j].strip()
        if not nxt:
            continue
        return _LIST_ITEM_OPENER_RE.match(nxt) is None
    return False


# Function/stop words that may sit inside a Title-Case clause heading.
_CLAUSE_TITLE_FUNC = {"and", "of", "the", "for", "to", "in", "a", "an", "or",
                      "on", "with", "by", "&", "/", "-", "from", "at"}
_RUNON_CLAUSE_RE = re.compile(r"^\d{1,2}\.\s+(.+)$")


# ── Title-Case scope sub-headings ────────────────────────────────────────
# A SOW lists each work activity as a short Title-Case label ("Unit Wiring",
# "Media Panel Installation", "Camera Rough and Install", "MDF and IDF Closet
# Buildout"), immediately followed by either a "<Provider> will …" sentence or a
# "Type / Qty." mini-table. These are NOT all-caps, so _looks_like_section_heading
# misses them and their content mis-roots under the previous heading (anyWAIR: 17
# atoms dumped under "SOW VERSION"). The detector below is structural-first (a
# short, terminal-punctuation-free, colon-free Title-Case line whose NEXT content
# line is a provider sentence or a table header), with a SemanticRule confirming
# the line actually NAMES a scope activity (so a stray Title-Case prose line such
# as "Athens Georgia" or "First Second" doesn't get promoted).
_SCOPE_PROVIDER_SENTENCE_RE = re.compile(
    r"^(?:PurTera|Provider|Vendor|Contractor|Customer|Subcontractor|The\s+\w+)\b.*?\b"
    r"(?:will|shall|may|is|are|provides?|installs?|completes?|performs?)\b",
    re.I,
)
# Tabular header tokens that open a scope work-item's "Type / Qty." mini-table.
_SCOPE_TABLE_HEADER_TOKENS = {
    "type", "qty", "qty.", "quantity", "qty of homeruns", "closet type",
    "equipment type", "from", "hr location", "number of idfs",
}


def _is_titlecase_heading_line(stripped: str) -> bool:
    """A short Title-Case label that reads as a heading, not a sentence/fact.

    Every alphabetic word must be Capitalized or a known joiner (and/of/the…); a
    lowercase content word means it's a sentence fragment, not a heading. Excludes
    all-caps (handled by _looks_like_section_heading), bullets, "Label: value"
    facts, and terminal-punctuation lines.
    """
    if not (2 <= len(stripped) <= 48):
        return False
    if stripped[-1] in ".!?,;:":
        return False
    if ":" in stripped:                       # "Closet Type: MDF" — a fact, not a heading
        return False
    if stripped.isupper():                    # ALL-CAPS handled elsewhere
        return False
    if _BULLET_LINE_RE.match(stripped):
        return False
    words = stripped.split()
    if not (1 <= len(words) <= 6):
        return False
    if not stripped[0].isupper():
        return False
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
    if digit_words:                           # "V1 Chase Smith", "106" rows aren't headings
        return False
    for w in alpha_words:
        first = next((c for c in w if c.isalpha()), "")
        if first.isupper():
            continue
        if w.strip("-/&.").lower() in _CLAUSE_TITLE_FUNC:
            continue
        return False                          # a lowercase content word → it's prose
    return True


def _next_content_is_scope_anchor(lines: list[str], idx: int) -> bool:
    """True when the line after ``idx`` anchors a scope work-item: a provider
    sentence ("PurTera will install …") or a "Type / Qty." table header token."""
    for j in range(idx + 1, len(lines)):
        nxt = lines[j].strip()
        if not nxt:
            continue
        low = nxt.lower().rstrip(".")
        if low in _SCOPE_TABLE_HEADER_TOKENS:
            return True
        return bool(_SCOPE_PROVIDER_SENTENCE_RE.match(nxt))
    return False


# Table summary-row / column-header words that are NEVER a scope activity name —
# "Total Drop", "Total Number", "Quantity", "Type" are tabular furniture, not
# work-item headings. Used by the offline lexical fallback (and as a hard guard).
_SCOPE_HEADING_STOPWORDS = {
    "total", "number", "count", "quantity", "qty", "qty.", "type", "location",
    "subtotal", "amount", "sum", "from", "included",
}


def _scope_heading_lexical(text: str) -> bool:
    """Offline net for the scope-activity judgment: a structurally-gated Title-Case
    line is a work-item heading UNLESS it opens with tabular-summary vocabulary
    ("Total Drop", "Quantity", "Type")."""
    words = (text or "").strip().lower().split()
    if not words:
        return False
    return words[0] not in _SCOPE_HEADING_STOPWORDS


_SCOPE_SUBHEADING_RULE = None


def _scope_subheading_rule():
    """SemanticRule: does this short Title-Case line NAME a scope work activity
    (an install / buildout / wiring task), as opposed to a stray capitalized line
    (a place name, a person, a date label)? Structural gating already constrains
    the candidates, so the lexical fallback fires whenever the structure matched."""
    global _SCOPE_SUBHEADING_RULE
    if _SCOPE_SUBHEADING_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _SCOPE_SUBHEADING_RULE = SemanticRule(
            name="scope_work_item_heading",
            positives=[
                "Unit Wiring", "Media Panel Installation", "Common Area",
                "Fiber backbone", "Unit AP Installation",
                "MDF and IDF Closet Buildout", "Camera Rough and Install",
                "Access Control Rough and Install", "Speaker Rough and Install",
                "Door Lock Installation", "Rack Buildout", "Cable Pull",
                "Access Point Installation", "Cabling and Termination",
                "Demolition and Removal", "Fiber Backbone Installation",
            ],
            negatives=[
                "Athens Georgia", "Chase Smith", "First Second",
                "Executive Summary", "Revision History", "Project Overview",
                "Total Number", "Full Name", "Job Title",
            ],
            threshold=0.55,
            lexical_fallback=_scope_heading_lexical,
        )
    return _SCOPE_SUBHEADING_RULE


def _looks_like_scope_subheading(stripped: str, lines: list[str], idx: int) -> bool:
    """A Title-Case scope work-item heading: short Title-Case label, anchored by a
    provider sentence or Type/Qty table on the next line, confirmed by the
    scope-activity SemanticRule. Universal across SOW formats."""
    if not _is_titlecase_heading_line(stripped):
        return False
    if not _scope_heading_lexical(stripped):   # hard guard: never a tabular word
        return False
    if not _next_content_is_scope_anchor(lines, idx):
        return False
    try:
        return _scope_subheading_rule().fires(stripped)
    except Exception:
        return True


#: Closed grammatical class: a clause whose Title-Case run starts with one of
#: these is a sentence with a capitalised opener, not a heading followed by a body.
_SENTENCE_OPENERS = frozenset({
    "no", "not", "all", "any", "each", "every", "both", "neither", "either",
    "this", "these", "those", "such", "the", "a", "an", "if", "unless", "when",
})


def _split_runon_numbered_clause(line: str) -> tuple[str, str] | None:
    """A numbered clause whose Title-Case heading runs straight into its body on
    ONE line — e.g. ``"8.  Contract Award and Interpretations ACE may accept …"``
    — which ``_numbered_heading`` misses (it only fires when the whole line is the
    title). Returns ``(heading, body)`` so the heading becomes its own section and
    the body its content; else ``None``.

    Boundary rule: the heading is the leading run of Title-Case words (capitalized
    words + small function words). A trailing capitalized word that is immediately
    followed by a lowercase word is the *subject of the body sentence*, not part of
    the title, so it's trimmed ("…Interpretations | ACE may accept"). Conservative:
    needs ≥2 capitalized CONTENT words of title and a ≥4-word body, so a normal
    numbered sentence item ("1. The Owner will not be responsible…") never fires.
    """
    m = _RUNON_CLAUSE_RE.match(line.strip())
    if not m:
        return None
    toks = m.group(1).split()
    if len(toks) < 5:
        return None

    def _bare(t: str) -> str:
        return t.strip(".,;:()")

    # Longest leading run of Title-Case words (capitalized or small function word).
    i = 0
    while i < len(toks):
        w = _bare(toks[i])
        if w and (w[0].isupper() or w.lower() in _CLAUSE_TITLE_FUNC):
            i += 1
        else:
            break
    # Find where the body sentence actually begins inside that run:
    #   (a) a capitalized determiner ("The"/"A"/"An") mid-run starts a new
    #       sentence ("Company Responsibility | The Company shall…"); take the
    #       LAST such — the title never contains a sentence-starting determiner;
    #   (b) else a trailing capitalized CONTENT word immediately before a
    #       lowercase word is the body's subject ("…Interpretations | ACE may…").
    cut = i
    for k in range(1, i):
        w = _bare(toks[k])
        if w[:1].isupper() and w.lower() in {"the", "a", "an"}:
            cut = k
    if cut == i and i >= 1 and i < len(toks) and toks[i][:1].islower():
        w = _bare(toks[i - 1])
        if w[:1].isupper() and w.lower() not in _CLAUSE_TITLE_FUNC:
            cut = i - 1
    title_toks = toks[:cut]
    # A run that BEGINS with a determiner, negator or quantifier is a sentence,
    # never a title: "1. No Provider Pre-Existing Materials are included…"
    # became heading "No Provider Pre-Existing" + body "Materials are
    # included…", which says the opposite of the clause (live 010300). Same
    # closed grammatical class the mid-run rule above already relies on.
    if title_toks and _bare(title_toks[0]).lower() in _SENTENCE_OPENERS:
        return None
    content_caps = [t for t in title_toks
                    if _bare(t)[:1].isupper() and _bare(t).lower() not in _CLAUSE_TITLE_FUNC]
    heading = " ".join(title_toks).strip().rstrip(".,;:")
    body = " ".join(toks[cut:]).strip()
    if (len(content_caps) < 2 or not heading or len(heading) > 60
            or heading[-1] in ".!?" or len(body.split()) < 4 or not body[:1].isupper()):
        return None
    return heading, body


_SIG_CERT_PHRASES = (
    "certificate of completion", "signer events", "envelope id",
    "signature adoption", "electronic record and signature", "carbon copy events",
    "envelope summary events", "hashed/encrypted", "autonav",
    "envelopeid stamping", "in person signer events", "certified delivery events",
    "signature timestamp", "status timestamp", "intermediary delivery events",
)


def _is_signature_certificate_page(page_text: str) -> bool:
    """A DocuSign / Adobe-Sign "Certificate of Completion" audit page appended to a
    signed document: pure signature-trail boilerplate (Signer Events, Envelope Id,
    Carbon Copy Events, Hashed/Encrypted timestamps, Notary Events …). It is NOT
    deal content — parsing it mints dozens of junk atoms (anyWAIR: ~46). Detect it
    by >=3 distinctive certificate phrases so the page collapses to ONE boilerplate
    marker. Vendor-agnostic; the phrases are e-signature-platform furniture, not
    deal language, so it never fires on a real scope/pricing page."""
    if not page_text:
        return False
    low = page_text.lower()
    return sum(1 for ph in _SIG_CERT_PHRASES if ph in low) >= 3


def _is_multi_paragraph_prose(page_text: str) -> bool:
    """True when a page reads as several blank-line-separated prose paragraphs.

    Used to route a short cover-letter / intro page (which carries a letterhead
    image, so it would otherwise hit the heavyweight layout pipeline that MERGES
    paragraphs into one glued atom) through the prose splitter instead, which
    keeps each paragraph a separate fact. A "prose paragraph" here is a chunk of
    >=40 chars containing a sentence (has spaces and ends with terminal
    punctuation, or is long); >=3 of them = clear prose page.
    """
    if not page_text:
        return False
    chunks = re.split(r"\n\s*\n", page_text)
    prose = 0
    for c in chunks:
        s = " ".join(c.split())
        if len(s) >= 40 and " " in s and (s.rstrip()[-1:] in ".!?:" or len(s) >= 120):
            prose += 1
    return prose >= 3


def _is_bulleted_list_page(page_text: str) -> bool:
    """True when a page is dominated by a BULLETED LIST — e.g. an architecture
    slide whose body is a bill-of-materials list ('• 74 x FG-30Gs', '• 24 x
    FS-1024E Core Switches', ...) beside a diagram image. Such a page carries an
    image so it would otherwise hit the heavyweight layout pipeline, which reorders
    the bullets by geometry around the diagram and splits 'N x DEVICE' into garbage
    ('x x x FS-648F-FPOE x FS-1024E ...'). The text layer is already in clean
    reading order, so route it to the prose/bullet splitter instead. A bullet line
    is a glyph alone OR a glyph + text; >=5 bullets AND bullets >= 30% of the
    non-blank lines = a bullet page (a stray bullet in a prose page won't trip it)."""
    if not page_text:
        return False
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    bullets = sum(1 for ln in lines if ln[:1] in "•▪●◦‣*-" or _BULLET_LINE_RE.match(ln))
    return bullets >= 5 and bullets >= 0.30 * len(lines)


_FORM_QUESTION_RULE = None


def _form_question_lexical(text: str) -> bool:
    """Offline net for 'is this a fill-out FORM question?': a short standalone
    prompt ending in '?'. The structural prefilter already required the '?'; this
    just rejects a long prose/legal sentence that happens to end in one."""
    s = (text or "").strip()
    return s.endswith("?") and 2 <= len(s.split()) <= 18


def _form_question_rule():
    """SemanticRule: does this line read like a fill-out FORM / field-report
    prompt ('Did you install the tablet?', 'Have you installed the NEXEO Box?')
    rather than a prose / legal / rhetorical question ('What happens in the event
    of a conflict?')? Distinguishing the two by MEANING keeps the questionnaire
    router from firing on a contract page that merely contains questions — which
    would wrongly skip that page's tables. Regex net is the offline fallback."""
    global _FORM_QUESTION_RULE
    if _FORM_QUESTION_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _FORM_QUESTION_RULE = SemanticRule(
            name="form_field_question",
            positives=[
                "Have you installed the NEXEO Box?",
                "Did you install a new tablet at the site?",
                "Is this store a 2 LANE store for drive thru?",
                "Did you have any issues with the tablet install?",
                "Was there a pre-existing audio box next to the old unit?",
                "Did you pull 2 cables to each POS?",
                "How many total cables were pulled?",
                "Are all devices powered on and online?",
                "Did you complete the closeout checklist?",
            ],
            negatives=[
                "What happens in the event of a conflict between this SOW and the MSA?",
                "Who bears the risk of loss during transit?",
                "What is the meaning of force majeure under this agreement?",
                "Why is network redundancy important for this deployment?",
                "Shall the contractor be liable for consequential damages?",
                "What are the payment terms?",
            ],
            threshold=0.52,
            lexical_fallback=_form_question_lexical,
        )
    return _FORM_QUESTION_RULE


def _is_form_page(page_text: str) -> bool:
    """True when a page is a fill-out form / field-report — a questionnaire (>=2
    form questions) OR a page carrying a photo-request ('Upload N photos showing
    X'). Either marks form TEXT that must take the text path, not the heavyweight
    layout pipeline (which splits the section header, glues the Q&A, and reorders
    by geometry)."""
    if not page_text:
        return False
    if _is_questionnaire_page(page_text):
        return True
    return any(_is_photo_request(ln.strip()) for ln in page_text.splitlines())


def _is_questionnaire_page(page_text: str) -> bool:
    """True when a page is a fill-out form / field-report questionnaire. Structural
    prefilter: >=2 standalone lines ending in '?'. Then a SemanticRule confirms
    they read like FORM field-prompts (not prose/legal/rhetorical questions), so
    this never fires on a contract page that merely contains questions (which would
    wrongly route it off the layout path and skip its tables). Offline -> the
    lexical net (short standalone question)."""
    if not page_text:
        return False
    q_lines = [ln.strip() for ln in page_text.splitlines() if ln.strip().endswith("?")]
    if len(q_lines) < 2:
        return False
    try:
        rule = _form_question_rule()
        return sum(1 for q in q_lines if rule.fires(q)) >= 2
    except Exception:
        return sum(1 for q in q_lines if _form_question_lexical(q)) >= 2


def _page_captured_text_len(page: dict[str, Any]) -> int:
    """Total chars of real textual content a built page captured (headings, prose,
    bullets, table cells, key-value pairs) — excluding image / boilerplate markers.
    Used by the coverage backstop to compare what the layout pipeline KEPT against
    the page's actual text layer, so a page that drops most of its text (not just
    all of it) is caught too."""
    total = 0

    def _walk(sections: list[dict[str, Any]]) -> None:
        nonlocal total
        for sec in sections or []:
            total += len((sec.get("heading") or "").strip())
            for b in sec.get("blocks") or []:
                if b.get("kind") in ("paragraph", "bullet_list", "table", "keyval"):
                    txt = (b.get("text") or "").strip()
                    if not txt and b.get("items"):
                        txt = " ".join(str(x) for x in b["items"])
                    if not txt and b.get("rows"):
                        txt = " ".join(str(x) for r in b["rows"] for x in r)
                    if not txt and b.get("pairs"):
                        txt = " ".join(f"{k} {v}" for k, v in b["pairs"])
                    total += len(txt.strip())
            _walk(sec.get("subsections") or [])

    _walk(page.get("sections") or [])
    return total


_MONTHS_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_WEEKDAYS_RE = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*"
_BARE_DATE_RE = re.compile(
    r"^(?:" + _WEEKDAYS_RE + r"\.?,?\s*)?"
    # ',?\s+' would miss 'May 14,2026' (comma, no space) — use '[,\s]+' so a comma
    # OR a space (or both) separates day from year.
    r"(?:" + _MONTHS_RE + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}"   # April 8, 2026 / May 14,2026
    r"|\d{1,2}\s+" + _MONTHS_RE + r"\.?[,\s]+\d{4}"                      # 8 April 2026
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"                                    # 4/8/2026
    r"|\d{4}-\d{1,2}-\d{1,2})$",                                         # 2026-04-08
    re.I,
)


def _is_bare_date_line(text: str) -> bool:
    """True when a line is JUST a date ('Wednesday, April 8, 2026', '4/8/2026').

    A standalone date at a page corner is timestamp furniture (repeated on every
    page of a form/field-report export) — it is NOT the document title and NOT a
    fact, so it must not be promoted to the section heading or emitted as an atom.
    """
    s = (text or "").strip()
    return 6 <= len(s) <= 40 and bool(_BARE_DATE_RE.match(s))


_TITLE_LINE_RULE = None


def _title_line_lexical(text: str) -> bool:
    """Offline net for 'is this line a document title?': a short, non-sentence
    label that is NOT a date, page footer, or CRM-id band. This is the structural
    fallback when the embedder is unreachable (and the deterministic test path)."""
    s = (text or "").strip()
    if not s or len(s) > 90 or s[-1] in ".!?,;:":
        return False
    # A question is never a document title — neither one with a '?' ('Is this
    # store a 2 LANE Store for Drive Thru?  No') nor a statement-phrased one
    # ('Have you train the MOD … units.  Yes', no '?'). Without this a form page's
    # first question was picked as the page title and dropped.
    if "?" in s or _FORM_INTERROG_RE.match(s):
        return False
    if _is_bare_date_line(s):
        return False
    # A bare clock time ('09:30 AM', '06:00') is page furniture (an arrival /
    # departure stamp), not a title — without this it was picked as a form page's
    # title once the real questions were (correctly) rejected.
    if re.fullmatch(r"\d{1,2}:\d{2}(\s*[AP]\.?M\.?)?", s, re.I):
        return False
    if _looks_like_page_footer(s):
        return False
    low = s.lower()
    if "hubspot" in low and re.search(r"\d{4,}", s):
        return False
    return True


def _title_line_rule():
    """SemanticRule: does this line read like a DOCUMENT/SECTION TITLE ('Burger
    King HME Install', 'Statement of Work') rather than an accessory that belongs
    in the path metadata, not the title — a date, page furniture, an id, a bare
    value? Recognising title-ness by MEANING generalises past the fixed furniture
    regexes (a date is 'obviously never a title'); the regex net is the offline
    fallback."""
    global _TITLE_LINE_RULE
    if _TITLE_LINE_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _TITLE_LINE_RULE = SemanticRule(
            name="document_title_line",
            positives=[
                "Burger King HME Install", "Statement of Work", "Project Overview",
                "Master Services Agreement", "Scope of Work", "Field Service Report",
                "HME NEXO Box Install", "Installation Checklist", "Site Roster & Facilities",
                "Network Cabling Proposal", "Deal Kit Summary", "Work Order",
                "anyWAIR UGA", "Closeout Report", "Executive Summary",
            ],
            negatives=[
                "Wednesday, April 8, 2026", "April 8, 2026", "4/8/2026", "2026-04-08",
                "Page 1 of 5", "www.purtera-it.com", "Confidential",
                "000087 - OPTBOT | HubSpot 60355665326", "06:00 AM", "Yes", "557",
                "Rev 2", "Sheet 1",
            ],
            threshold=0.52,
            lexical_fallback=_title_line_lexical,
        )
    return _TITLE_LINE_RULE


_SECTION_HEADER_RULE = None


def _section_header_lexical(line: str) -> bool:
    """Offline net for 'is this standalone line a form SECTION header?'
    (the embedder is the primary judge; this fires when it's unreachable / in
    the deterministic test + offline-labeler path). A form sub-header is a short
    Title-Case label naming a subsystem ('BK Audio', 'POS Cabling', 'Tablet
    Install', 'Drive Thru Lane 2'). Distinguish it from the other standalone
    short lines on a form page — bare answers ('Yes', '8') and question
    continuations ('Talking POS in the store', which carry lowercase words) —
    by requiring most ALPHA words to be capitalised and rejecting bare values."""
    s = (line or "").strip()
    if not s or s[-1] in "?.,;:!" or not any(c.isalpha() for c in s):
        return False
    words = s.split()
    if not (1 <= len(words) <= 6):
        return False
    if s.lower() in {"yes", "no", "n/a", "na", "tbd", "none", "true", "false"}:
        return False
    if _is_photo_request(s) or _FORM_INTERROG_RE.match(s):
        return False
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha:
        return False
    capped = sum(1 for w in alpha if w[0].isupper())
    return capped / len(alpha) >= 0.8


def _section_header_rule():
    """SemanticRule: is this standalone line a form/field-report SECTION header
    ('Tablet Install', 'BK Audio', 'POS Cabling') — a divider the following
    questions belong UNDER — rather than content (a question, a bare answer, a
    material/tool line)? Header-ness is a meaning judgment, so the embedder leads
    and ``_section_header_lexical`` is the offline net."""
    global _SECTION_HEADER_RULE
    if _SECTION_HEADER_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _SECTION_HEADER_RULE = SemanticRule(
            name="form_section_header",
            positives=[
                "Tablet Install", "BK Audio", "POS Cabling", "Drive Thru Lane 2",
                "Headset Install", "Site Information", "Network Configuration",
                "Power and Grounding", "Equipment Installation", "Menu Board",
                "Speaker Post", "Drive Thru", "Cabling", "Server Rack",
            ],
            negatives=[
                "Did you install a new tablet at the site?", "New Tablet", "Yes", "No",
                "8", "Talking POS in the store", "CAT6 jacks", "Tape measure",
                "10 ft ladder", "Cable tester", "Upload 4 photos of the unit",
                "Was there a pre-existing BK Audio Box next to the old HME unit?",
                "The contractor shall provide all materials",
            ],
            threshold=0.50,
            lexical_fallback=_section_header_lexical,
        )
    return _SECTION_HEADER_RULE


def _is_form_section_header(line: str) -> bool:
    """A standalone line on a form page that heads the questions below it. Cheap
    textual preconditions, then the semantic rule (embedding online / lexical
    offline) makes the call."""
    s = (line or "").strip()
    if not s or len(s.split()) > 6 or s[-1] in "?.,;:!":
        return False
    if _looks_like_page_footer(s) or _is_bare_date_line(s):
        return False
    # A clock time ('09:30 AM' arrival/departure stamp) is a form VALUE, not a
    # section header — and it slips past the title-case check (its only alpha
    # token is the 'AM'/'PM' unit). A real header names a subsystem, so require
    # at least one substantive word (>=3 alpha chars, not an AM/PM unit).
    if re.fullmatch(r"\d{1,2}:\d{2}(\s*[AP]\.?M\.?)?", s, re.I):
        return False
    if not any(len(w) >= 3 and w.isalpha() and w.upper() not in ("AM", "PM")
               for w in s.split()):
        return False
    try:
        return _section_header_rule().fires(s)
    except Exception:
        return _section_header_lexical(s)


def _detect_text_title(page_text: str) -> str | None:
    """First prominent line of a text page — the document's main section.

    Returns the human title ("Burger King HME Install"), skipping accessories that
    belong in path metadata rather than the title — a page-corner timestamp
    ("Wednesday, April 8, 2026"), CRM id bands, footer furniture. The judgment is
    semantic (a date is never a title; a SemanticRule that has seen real titles
    knows that), with the furniture regexes as the offline fallback. Used to root
    every atom's section_path so a sub-heading renders as a path.
    """
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line or len(line) > 90:
            continue
        # A title is a label, not a sentence.
        if line[-1] in ".!?,;:":
            continue
        try:
            if _title_line_rule().fires(line):
                return line
        except Exception:
            if _title_line_lexical(line):
                return line
    return None


def _place_tables_in_sections(
    pdf_path: Path,
    page_index: int,
    sections: list[dict[str, Any]],
    table_blocks: list[dict[str, Any]],
    table_bboxes: list[Any],
) -> None:
    """Insert recovered ruled-table blocks into the section whose heading they
    fall under (by vertical position), instead of a trailing heading-less
    section — so a roster table stays under its "Site roster" heading and
    inherits the section_path, rather than floating at the document root.
    """
    import fitz  # type: ignore[import-not-found]

    def _append_trailing() -> None:
        sections.append(
            {"heading": "", "level": 2, "blocks": list(table_blocks), "subsections": []}
        )

    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_index]
            # y0 of each section's heading on the page.
            heading_y: list[tuple[float, int]] = []
            for si, sec in enumerate(sections):
                h = (sec.get("heading") or "").strip()
                if not h:
                    continue
                try:
                    rects = page.search_for(h)
                except Exception:
                    rects = []
                if rects:
                    heading_y.append((min(r.y0 for r in rects), si))
            heading_y.sort()
            if not heading_y:
                _append_trailing()
                return
            for blk, bbox in zip(table_blocks, table_bboxes):
                try:
                    ty = float(bbox.y0)
                except Exception:
                    ty = 0.0
                # The last heading that starts above the table top owns it.
                target_si: int | None = None
                for hy, si in heading_y:
                    if hy <= ty:
                        target_si = si
                    else:
                        break
                if target_si is None:
                    sections.append(
                        {"heading": "", "level": 2, "blocks": [blk], "subsections": []}
                    )
                else:
                    sections[target_si].setdefault("blocks", []).append(blk)
    except Exception:  # pragma: no cover — never fail the parse over placement
        _append_trailing()


def _strip_title_block(sections: list[dict[str, Any]], title: str) -> None:
    """Remove the paragraph block whose text is the document title.

    Once the title is promoted to the main section (section_path root), keeping
    it as a content atom too is redundant — a heading is structure, not a fact.
    Mutates ``sections`` in place, removing only the first exact match.
    """
    want = title.strip()
    for sec in sections:
        blocks = sec.get("blocks") or []
        for bi, blk in enumerate(blocks):
            if blk.get("kind") == "paragraph" and (blk.get("text") or "").strip() == want:
                del blocks[bi]
                return


def _looks_like_section_heading(stripped: str) -> bool:
    """True when an all-caps line is a real section heading, not a sentence tail
    or an identifier code.

    A heading is a short label ("PACKET SUMMARY", "BUDGET AND APPROVAL MATRIX").
    The bare ``str.isupper()`` test also fires on all-caps identifiers like
    ``MOCK-MSA-2026-OPTBOT-001.`` (the tail of a sentence) — which would both
    truncate the preceding paragraph and stamp a garbage section on the next.
    """
    if stripped.startswith("#"):
        return True
    if not (stripped.isupper() and len(stripped) >= 3):
        return False
    # Headings don't end with sentence punctuation.
    if stripped[-1] in ".,;":
        return False
    # Reject single-token identifier codes (part/MSA/PO numbers) like
    # "MOCK-MSA-2026-OPTBOT-001": one token, with digits and hyphens.
    if " " not in stripped and "-" in stripped and any(c.isdigit() for c in stripped):
        return False
    # A clock time ('09:30 AM') passes str.isupper() (digits are uncased, 'AM' is
    # upper) but is a form VALUE, not a heading. A real heading names something —
    # require a substantive word (>=3 alpha chars, not an AM/PM unit).
    if re.fullmatch(r"\d{1,2}:\d{2}(\s*[AP]\.?M\.?)?", stripped, re.I):
        return False
    if not any(len(w) >= 3 and w.isalpha() and w.upper() not in ("AM", "PM")
               for w in stripped.split()):
        return False
    return True


# Checkbox / task-list glyphs often extract as a bare capital ``I`` (or ``l`` /
# ``|``) before an owner+verb action line. Structural — Capitalized owner +
# ``to <verb>`` — never a name list. Distinguishes "I Jacob to send…" from
# prose "I think we should…".
_CHECKBOX_OWNER_ACTION_RE = re.compile(
    r"^\s*[Il|]\s+"
    r"(?P<body>"
    r"[A-Z][\w\-']*"
    r"(?:\s+(?:[A-Z][\w\-']+|team|customer|staff|group|vendor)){0,4}"
    r"\s+to\s+[a-z].+"
    r")\s*$"
)
# Same pattern mid-blob (glued paragraph), used to split already-joined items.
_CHECKBOX_OWNER_ACTION_SPLIT_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))[Il|]\s+"
    r"(?=[A-Z][\w\-']*(?:\s+(?:[A-Z][\w\-']+|team|customer|staff|group|vendor)){0,4}"
    r"\s+to\s+[a-z])"
)
# Embedded Title-Case / ALL CAPS meeting section headers inside a glued blob.
_MEETING_HEADER_EMBEDDED_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))"
    r"(?P<header>"
    r"Executive\s+Summary|Action\s+Items?|Key\s+Decisions?|Decisions?|"
    r"Open\s+Questions?|Attendees|Participants|Next\s+Steps|Agenda|"
    r"Discussion|Notes|Follow[\s-]?Ups?"
    r")"
    r"(?=\s|$)",
    re.IGNORECASE,
)


def _is_meeting_section_heading_line(stripped: str) -> bool:
    """True when a standalone line is a meeting-summary section header.

    Uses the shared ``meeting_section_header`` SemanticRule (embeddings online,
    ``detect_section`` lexical offline). Not gated to form pages — summary
    front-matter is the primary consumer.
    """
    s = (stripped or "").strip()
    if not s:
        return False
    try:
        from app.core.semantic_rules import is_meeting_section_header

        return bool(is_meeting_section_header(s))
    except Exception:
        try:
            from app.core.normalizers import detect_section

            return detect_section(s) is not None
        except Exception:
            return False


def _canonical_meeting_section_heading(stripped: str) -> str:
    """Canonical display label for a meeting section header line."""
    try:
        from app.core.normalizers import detect_section

        return detect_section(stripped) or stripped.strip().rstrip(":")
    except Exception:
        return stripped.strip().rstrip(":")


def _checkbox_owner_action_body(stripped: str) -> str | None:
    """Return bullet body when line is a checkbox→``I Owner to verb…`` item."""
    m = _CHECKBOX_OWNER_ACTION_RE.match(stripped or "")
    if not m:
        return None
    body = (m.group("body") or "").strip()
    return body or None


def _meeting_section_connective(section_path: list[str]) -> tuple[str | None, str | None]:
    """Return ``(list_section slug, section_header)`` from the deepest meeting heading."""
    try:
        from app.core.normalizers import detect_section, meeting_section_slug
    except Exception:
        return None, None
    for part in reversed(section_path or []):
        label = detect_section(str(part)) or (
            str(part).strip() if meeting_section_slug(str(part)) else None
        )
        if not label:
            # Accept path parts that are already canonical display labels.
            slug = meeting_section_slug(str(part))
            if slug:
                return slug, str(part).strip()
            continue
        slug = meeting_section_slug(label)
        if slug:
            return slug, label
    return None, None


def _split_glued_meeting_summary_paragraph(
    text: str,
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Repair a glued meeting-summary paragraph into sectioned bullet blocks.

    Handles the common PDF failure mode where Title-Case headers
    (``Action Items``, ``Key Decisions``) and checkbox-``I`` bullets were
    joined into one prose atom.

    Returns ``(blocks, trailing_header)`` when repaired, else ``None``.
    ``trailing_header`` is set when the blob ends on a section label with no
    bullets beneath it — the caller should stamp that heading onto the next
    bullet_list block.
    """
    raw = (text or "").strip()
    if not raw or len(raw) < 40:
        return None
    headers = list(_MEETING_HEADER_EMBEDDED_RE.finditer(raw))
    action_starts = list(_CHECKBOX_OWNER_ACTION_SPLIT_RE.finditer(raw))
    if not headers and len(action_starts) < 2:
        return None
    if not headers and not re.search(
        r"\b(?:action\s+items?|key\s+decisions?|executive\s+summary)\b",
        raw,
        re.I,
    ):
        return None

    cuts: list[tuple[int, str, str | None]] = []
    for m in headers:
        cuts.append((m.start(), "header", _canonical_meeting_section_heading(m.group("header"))))
    for m in action_starts:
        cuts.append((m.start(), "action", None))
    cuts.sort(key=lambda c: c[0])
    if not cuts:
        return None

    blocks: list[dict[str, Any]] = []
    current_heading: str | None = None
    bullet_items: list[dict[str, str]] = []
    trailing_header: str | None = None

    def flush_bullets() -> None:
        nonlocal bullet_items
        if bullet_items:
            blocks.append(
                {
                    "kind": "bullet_list",
                    "items": list(bullet_items),
                    "meeting_section": current_heading,
                }
            )
            bullet_items = []

    for i, (pos, kind, label) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(raw)
        chunk = raw[pos:end].strip()
        if not chunk:
            continue
        if kind == "header":
            flush_bullets()
            current_heading = label
            trailing_header = label  # may clear if bullets follow
            rest = chunk
            if label:
                m_head = re.match(
                    re.escape(label).replace(r"\ ", r"\s+"),
                    chunk,
                    re.IGNORECASE,
                )
                if m_head:
                    rest = chunk[m_head.end():].strip()
            if not rest:
                continue
            body = _checkbox_owner_action_body(rest)
            if body is None and rest[:1].isupper():
                body = _checkbox_owner_action_body("I " + rest)
            if body:
                bullet_items.append({"text": body})
                trailing_header = None
            continue
        trailing_header = None
        body = _checkbox_owner_action_body(chunk)
        if body:
            bullet_items.append({"text": body})
        else:
            cleaned = re.sub(r"^\s*[Il|]\s+", "", chunk).strip()
            if cleaned:
                bullet_items.append({"text": cleaned})

    flush_bullets()
    n_bullets = sum(len(b.get("items") or []) for b in blocks if b.get("kind") == "bullet_list")
    if n_bullets < 2 and not trailing_header:
        return None
    if n_bullets < 2 and trailing_header and not blocks:
        # Header-only carry (e.g. blob is just chrome before a real list).
        return [], trailing_header
    return blocks, trailing_header


def _is_list_intro(text: str) -> bool:
    """A line that ends with a colon and so introduces the list below it
    ("Milestone billing schedule:", "PurTera will perform … new circuits:"). The
    trailing colon (whole line, nothing after) is the signal — length-agnostic,
    since a real intro ends with ":" whether it's a short label or a sentence. A
    "Label: value" fact does NOT end with a colon, so it's never an intro."""
    s = (text or "").strip()
    return len(s) >= 4 and s.endswith(":")


def _is_group_item(block: dict[str, Any]) -> bool:
    """A block that reads as one item of an intro's list: a bullet list, or a
    label-style paragraph ("Megger (insulation resistance): …") — a colon early
    in the line — that isn't itself a list intro."""
    kind = block.get("kind")
    if kind == "bullet_list":
        return True
    if kind != "paragraph":
        return False
    txt = (block.get("text") or "").strip()
    if not txt or _is_list_intro(txt):
        return False
    head = txt[:45]
    return ":" in head and not head.startswith("http")


def _promote_list_intros_to_subsections(sections: list[dict[str, Any]]) -> None:
    """Turn a colon-terminated intro line that sits above a group of list items
    into a sub-section wrapping that group, so the items nest under the intro
    (and the bare intro stops being a standalone atom). Handles both dash-bullet
    lists and runs of label-style paragraphs. Mutates ``sections`` in place."""
    for sec in sections:
        blocks = sec.get("blocks") or []
        kept: list[dict[str, Any]] = []
        subs = list(sec.get("subsections") or [])
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.get("kind") == "paragraph" and _is_list_intro(b.get("text") or ""):
                grouped: list[dict[str, Any]] = []
                count = 0
                j = i + 1
                while j < len(blocks) and _is_group_item(blocks[j]):
                    grouped.append(blocks[j])
                    count += len(blocks[j].get("items") or []) if blocks[j].get("kind") == "bullet_list" else 1
                    j += 1
                if count >= 2:
                    subs.append({
                        "heading": (b.get("text") or "").strip().rstrip(":").strip(),
                        "level": 3,
                        "blocks": grouped,
                        "subsections": [],
                    })
                    i = j
                    continue
            kept.append(b)
            i += 1
        sec["blocks"] = kept
        sec["subsections"] = subs


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
    pending_bullet = False  # saw a lone bullet glyph; next content line is its text
    # A field-report / questionnaire page (>=2 '?') stacks Q&A under short visual
    # sub-headers ("Tablet Install", "BK Audio", "POS Cabling"). Recognise those
    # as SECTION headers so the questions below nest under them — instead of the
    # header leaking as a junk scope_item or vanishing. Gated to form pages so an
    # ordinary prose page's short Title-Case lines aren't promoted.
    # Diarized transcript pages (dense ``Name [mm:ss]`` stamps) must NOT use
    # form-page mode: speaker labels look like Title-Case headers and rhetorical
    # questions trip the ``?`` gate, which then promotes every speaker stamp to
    # a section heading and tags blocks form_field=True (disabling short drops).
    try:
        from app.core.hybrid_summary_transcript import count_speaker_timestamp_hits

        _is_diarized_transcript = count_speaker_timestamp_hits(page_text or "") >= 2
    except Exception:
        _is_diarized_transcript = False
    _is_form_pg = (not _is_diarized_transcript) and ((page_text or "").count("?") >= 2)

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        # Drop a page-footer line ("Page 2/2 | PurPulse ...") that got swept into
        # a paragraph — it's furniture, and must not glue onto the last record.
        kept = [
            x.strip()
            for x in paragraph_lines
            if x.strip() and not _looks_like_page_footer(x.strip())
        ]
        paragraph_lines = []
        if not kept:
            return
        # An unambiguous record-list (signature roster, "Name: decision."
        # approval notes) becomes one atom per record — one fact, one atom.
        records = _split_structured_records(kept)
        if records:
            for rec in records:
                current_blocks.append({"kind": "paragraph", "text": rec, "lines": [rec]})
            return
        text = " ".join(kept).strip()
        if text:
            # Keep the per-line structure alongside the joined text: a glued
            # "key = value" metadata block needs the real line boundaries so a
            # trailing prose line isn't swallowed into the last value.
            current_blocks.append({"kind": "paragraph", "text": text, "lines": kept})

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

    skip_through = -1  # lines consumed by a multi-line photo-request instruction
    for idx, raw in enumerate(lines):
        if idx <= skip_through:
            continue
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            continue

        # A bare page-corner timestamp ("Wednesday, April 8, 2026") is furniture
        # repeated on every page of a form/field-report export — it is not the
        # title, not a section, and not a fact. Skip ONLY at the page top (the
        # corner timestamp position) so it neither glues onto the real title nor
        # becomes its own atom — a date deeper in the page is a real value (a
        # "Date:" field, a revision date) and must be kept.
        if idx <= 1 and _is_bare_date_line(line.strip()):
            flush_paragraph()
            flush_bullets()
            continue

        # A lone page-number line ("4") at the page foot is furniture — drop it
        # so it never surfaces as a junk numeric atom. Guarded to an isolated /
        # trailing short-digit line so a real numbered list item is untouched.
        if (line.strip().isdigit() and len(line.strip()) <= 3
                and (idx == len(lines) - 1 or not lines[idx + 1].strip())):
            flush_paragraph()
            flush_bullets()
            continue

        # A form sub-header ("Tablet Install", "BK Audio", "POS Cabling") on a
        # questionnaire page is a SECTION divider: start a new section so the
        # questions below nest under it, and DON'T emit it as its own atom
        # (breadcrumb only). After _regroup_form_qa answers are joined to their
        # questions, so a standalone short Title-Case line here is a header.
        if _is_form_pg and _is_form_section_header(line.strip()):
            flush_section()
            current_heading = line.strip()
            continue

        # A photo-request instruction ("Upload 4 Photos of the Nexeo installed at
        # the site.") is the LINKAGE for the images it asks for — its "answer" is
        # those photos (already captioned onto the image markers). Break it out of
        # any glued Q&A as its OWN unit so the question/answer stays clean and the
        # request reads as the images' reference, not buried text. Multi-line
        # instructions are gathered (the continuation sentence-lines that follow).
        if _is_photo_request(line.strip()) and idx > skip_through:
            flush_paragraph()
            flush_bullets()
            req = [line.strip()]
            j = idx + 1
            while j < len(lines):
                nxt = lines[j].strip()
                # A bare number is a footer page number, not a continuation — and
                # a new question / request / blank ends the wrapped instruction.
                if (not nxt or nxt.isdigit() or nxt.endswith("?")
                        or _FORM_INTERROG_RE.match(nxt) or _is_photo_request(nxt)):
                    break
                # Wrapped continuation: a lowercase-start tail, a 3+ word sentence
                # line, OR a short Title-Case fragment that completes the request
                # phrase ("Upload photo showing Battery" + "Charger Mounting").
                # Without the Title-Case case the tail leaked as a junk atom
                # ("Mounting 4") and the caption was left truncated.
                if (nxt[:1].islower() or len(nxt.split()) >= 3
                        or re.match(r"^[A-Z][\w/&-]*( [A-Z][\w/&-]*){0,2}$", nxt)):
                    req.append(nxt)
                    j += 1
                else:
                    break
            skip_through = j - 1
            paragraph_lines.extend(req)
            flush_paragraph()
            continue

        # Dotted-decimal SOW/RFP section heading ("1.0 SCOPE", "2.1 GENERAL
        # REQUIREMENTS. The Contractor shall…", "2.1.1.1 Confined Space. N / A")
        # — multi-level numbers, title often run into the body and ending in '.',
        # which the single-level rules below miss. Roots the clause under its own
        # numbered section instead of a stale carried heading ("LIST OF TABLES").
        dotted = _split_dotted_section(line)
        if dotted:
            flush_section()
            current_heading = dotted[0]
            if dotted[1]:
                paragraph_lines.append(dotted[1])
            continue

        # Numbered section heading ("1. Authoritative physical site roster") —
        # checked before the bullet rule (which would otherwise strip the
        # number and treat the title as a list item). Only when a body line,
        # not another numbered item, follows — so a numbered list stays a list.
        num_head = _numbered_heading(line)
        if num_head and _next_content_is_body(lines, idx):
            flush_section()
            current_heading = num_head
            continue

        # Run-on numbered clause: "8.  Contract Award and Interpretations ACE may
        # accept…" — the title is glued to the body on one line, so the clean
        # heading rule above misses it and the bullet rule below would strip the
        # number and bury it under the previous section. Split the title into its
        # own section so the clause (and everything under it until the next
        # number) roots correctly.
        runon = _split_runon_numbered_clause(line)
        if runon:
            flush_section()
            current_heading = runon[0]
            paragraph_lines.append(runon[1])
            continue

        # A bullet glyph alone on its line ("•") — its text is on the NEXT line(s).
        # Mark pending so the following content line becomes a bullet item (its
        # wrapped tail then joins via the lowercase-continuation rule below) — so a
        # glyph-per-line BOM list separates into items instead of gluing into one
        # mega-atom (or, with column extraction, scrambling).
        if _BARE_BULLET_RE.match(line) or _BARE_ENUM_RE.match(line):
            flush_paragraph()
            pending_bullet = True
            continue

        bullet_m = _BULLET_LINE_RE.match(line)
        if bullet_m:
            flush_paragraph()
            bullet_buffer.append(bullet_m.group(2).strip())
            pending_bullet = False
            continue

        # The content line right after a lone bullet glyph IS that bullet's text.
        if pending_bullet:
            flush_paragraph()
            bullet_buffer.append(line.strip())
            pending_bullet = False
            continue

        # Title-Case scope work-item heading ("Media Panel Installation",
        # "Camera Rough and Install") — not all-caps, so the heading rule below
        # misses it and its table/description would mis-root under the previous
        # section. Anchored by a provider sentence or Type/Qty table on the next
        # line + confirmed by the scope-activity SemanticRule.
        stripped = line.strip()
        # Meeting-summary section headers (Executive Summary / Action Items /
        # Key Decisions). Embeddings via meeting_section_header rule; lexical
        # detect_section offline. NOT form-gated — summary front matter is the
        # primary consumer, and diarized pages disable form mode.
        if _is_meeting_section_heading_line(stripped):
            flush_section()
            current_heading = _canonical_meeting_section_heading(stripped)
            continue

        # Checkbox→``I Owner to verb…`` action lines (PDF extracts ☐ as ``I``).
        action_body = _checkbox_owner_action_body(stripped)
        if action_body:
            flush_paragraph()
            bullet_buffer.append(action_body)
            pending_bullet = False
            continue

        if _looks_like_scope_subheading(stripped, lines, idx):
            flush_section()
            current_heading = stripped
            continue

        # heading guess (all caps or markdown-style #)
        if len(stripped) <= 80 and _looks_like_section_heading(stripped):
            flush_section()
            current_heading = stripped.lstrip("# ").strip()
            continue

        # A lowercase line right after a bullet is that bullet wrapped across
        # lines (the PDF broke a long item) — append it to the last bullet
        # instead of orphaning it as a separate fragment paragraph.
        if bullet_buffer and not paragraph_lines and stripped[:1].islower():
            bullet_buffer[-1] = f"{bullet_buffer[-1]} {stripped}".strip()
            continue

        # Paragraph continuation. Flush any pending bullets first so a
        # paragraph doesn't get glued onto a bullet list.
        flush_bullets()
        paragraph_lines.append(line)

    flush_section()
    # A "<label>:" line directly above a bullet list becomes a sub-section over
    # those bullets, so they nest under the label instead of floating.
    _promote_list_intros_to_subsections(sections)
    # Drop empty sections that may have been created by trailing
    # whitespace.
    return [s for s in sections if s.get("blocks") or s.get("heading") or s.get("subsections")]


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


# ───────────────── PR5 (post-v3) — PDF v2 supplements ─────────────────














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












# =====================================================================
# Boss-review (post-2-case) PDF v3 — vertical-listed tables, vertical
# workflow, and group-aware form-option states.
# =====================================================================













def _build_schematic_prepass_failure_atom(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
    exception: Exception,
    traceback: str,
) -> EvidenceAtom:
    """Surface a schematic pre-pass crash as a single warning atom.

    Without this, legacy tests stayed green even when the schematic
    pre-pass blew up — the broad ``except`` simply dropped every
    schematic atom. Boss-review fix: failures now ship as a
    ``schematic_warning`` with the truncated traceback in
    ``value['traceback']`` so the operator can see what happened.
    """
    from app.parsers.schematic_atom_emitters import emit_warning_atom
    from app.parsers.schematic_models import SchematicWarning

    detail = f"{type(exception).__name__}: {exception}"
    truncated = traceback[-1500:] if len(traceback) > 1500 else traceback
    warning = SchematicWarning.make(
        warning_type="prepass_failure",
        page_index=0,
        sheet_number=None,
        detail=f"Schematic pre-pass raised {detail}",
        extras={"failure": detail, "traceback_tail": truncated},
    )
    return emit_warning_atom(
        warning=warning,
        project_id=project_id,
        artifact_id=artifact_id,
        filename=path.name,
        parser_version=parser_version,
        page=None,
    )
















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
