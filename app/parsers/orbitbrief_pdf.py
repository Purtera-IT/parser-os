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
_PAGE_NUMBER_PATTERN = re.compile(
    # Match real "Page 3 of 12" AND the unrendered template version
    # "Page X of Y" / "Page X of N" (reportlab footers sometimes leave
    # placeholders unresolved when the doc is generated quickly).
    r"\bpage\s+(?:\d+|[xn])\s+of\s+(?:\d+|[xny])\b",
    re.IGNORECASE,
)
# Standalone "Page N" / "Page X" without "of" — only counts as a
# footer when corroborated by other footer hints in the same line.
_PAGE_NUMBER_LOOSE_PATTERN = re.compile(r"\bpage\s+(?:\d+|[xn])\b", re.IGNORECASE)
# Copyright line shape: "(c) 2026 ORG", "© 2026 ORG", "Copyright 2026 ORG"
_COPYRIGHT_PATTERN = re.compile(
    r"(?:\(c\)|©|copyright)\s*(?:19|20)\d{2}", re.IGNORECASE
)
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
    "do not redistribute",
    "do not distribute",
    "internal use only",
    "internal only",
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
    text_lower = text.lower()
    # "Page 17" alone (no "of M") on a short line that also carries an
    # RFP/footer hint is also a footer.
    if _PAGE_NUMBER_LOOSE_PATTERN.search(text):
        if any(hint in text_lower for hint in _PAGE_FOOTER_HINTS):
            # Make sure it doesn't carry quantitative info that scope
            # atoms care about.
            has_money = bool(re.search(r"\$\s*\d", text))
            has_qty = bool(re.search(r"\b\d+(?:,\d{3})*\s*(?:cameras?|aps?|drops?|outlets?|jacks?|users?|licenses?|installations?)\b", text, re.IGNORECASE))
            if not (has_money or has_qty):
                return True
    # Copyright + confidentiality marker on a short pipe-separated line
    # is universally a footer band (every page repeats it).
    if _COPYRIGHT_PATTERN.search(text):
        hint_count = sum(1 for hint in _PAGE_FOOTER_HINTS if hint in text_lower)
        if hint_count >= 1:
            return True
    # Two-or-more footer hints in a single short line — pipe-separated
    # bands like "Confidential | Page X of Y | (c) 2026 X | DO NOT
    # REDISTRIBUTE" are universally footer furniture.
    hint_count = sum(1 for hint in _PAGE_FOOTER_HINTS if hint in text_lower)
    if hint_count >= 2 and len(text) <= 200:
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
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files,
        )


# ──────────────────────── public helpers ─────────────────────────────────


def _fitz_site_roster_fallback(
    *,
    pdf_path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
    already_emitted: set[str | None] | None = None,
) -> list[EvidenceAtom]:
    """Use ``fitz.find_tables()`` to catch site rosters the structured
    pipeline missed.

    Returns a list of structured ``physical_site`` entity atoms. Never
    raises — on any error (fitz unavailable, PDF unreadable, no tables)
    returns an empty list.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        from app.parsers.site_roster_extractor import (
            extract_site_roster,
            looks_like_site_roster,
        )
    except Exception:
        return []

    already_emitted = already_emitted or set()
    out: list[EvidenceAtom] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []
    try:
        # Pull the document-level surrounding text once so the
        # extractor can spot ``kind=physical_site`` declarations.
        page_texts: list[str] = []
        try:
            for p in doc:
                try:
                    page_texts.append(p.get_text() or "")
                except Exception:
                    continue
        except Exception:
            page_texts = []
        document_text = "\n".join(page_texts)

        for page_index, page in enumerate(doc):
            try:
                tables_finder = page.find_tables()
            except Exception:
                continue
            tables = list(getattr(tables_finder, "tables", []) or [])
            if not tables:
                continue
            for table_index, table in enumerate(tables):
                try:
                    extracted = table.extract()
                except Exception:
                    continue
                if not extracted or len(extracted) < 2:
                    continue
                header = [(c or "") for c in extracted[0]]
                body = extracted[1:]
                rows: list[dict[str, Any]] = []
                for r in body:
                    if not r:
                        continue
                    rows.append({
                        header[i] if i < len(header) and header[i] else f"col_{i}": (
                            # Collapse internal whitespace in cell values so a
                            # word that wrapped across two display lines (e.g.
                            # "ATL-WEST-0\n2") renders as a single token.
                            " ".join((c or "").split())
                        )
                        for i, c in enumerate(r)
                    })
                if not rows:
                    continue
                # Build column header list, then route through
                # site_roster_extractor.
                columns = [
                    header[i] if i < len(header) and header[i] else f"col_{i}"
                    for i in range(len(header) if header else (len(rows[0]) if rows else 0))
                ]
                try:
                    is_roster = looks_like_site_roster(
                        columns=columns, rows=rows, surrounding_text=document_text
                    )
                except Exception:
                    is_roster = False
                if not is_roster:
                    continue
                try:
                    roster_rows = extract_site_roster(
                        columns=columns, rows=rows, surrounding_text=document_text
                    )
                except Exception:
                    roster_rows = []
                # Bbox from fitz table -> base locator
                try:
                    bbox = table.bbox
                    locator_base = {
                        "page": int(page_index),
                        "block_kind": "table",
                        "bbox": list(bbox),
                        "extraction": "site_roster_fitz_fallback_v1",
                    }
                except Exception:
                    locator_base = {"page": int(page_index), "extraction": "site_roster_fitz_fallback_v1"}
                for site_row in roster_rows:
                    sid = (site_row.site_id or "").strip()
                    # Normalize whitespace inside the ID (PDF wrap
                    # artifacts: "ATL-WEST-0 2" -> "ATL-WEST-02")
                    if sid and " " in sid:
                        compact = re.sub(r"\s+", "", sid)
                        # Only collapse when the compact form still
                        # looks like a site ID — keeps "Building C"
                        # type values from getting smushed.
                        from app.parsers.site_roster_extractor import _SITE_ID_SHAPE_RE
                        if _SITE_ID_SHAPE_RE.match(compact):
                            sid = compact
                    if sid in already_emitted:
                        continue
                    already_emitted.add(sid)
                    canon_id = sid or site_row.facility_name or ""
                    if not canon_id:
                        continue
                    site_text = " | ".join(
                        f"{k}: {v}"
                        for k, v in [
                            ("site_id", sid or site_row.site_id),
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
                    locator = {
                        **locator_base,
                        "row_index": site_row.row_index,
                        "table_index": table_index,
                    }
                    out.append(
                        _make_atom(
                            text=site_text or canon_id,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=pdf_path.name,
                            parser_version=parser_version,
                            atom_type=AtomType.entity,
                            authority_class=AuthorityClass.contractual_scope,
                            confidence=site_row.confidence,
                            locator=locator,
                            value={
                                "kind": "physical_site",
                                "site_id": sid or site_row.site_id,
                                "facility_name": site_row.facility_name,
                                "street_address": site_row.street_address,
                                "mdf_idf": site_row.mdf_idf,
                                "access_window": site_row.access_window,
                                "escort_owner": site_row.escort_owner,
                                "contact": site_row.contact,
                                "phone": site_row.phone,
                                "email": site_row.email,
                                "city_state": site_row.city_state,
                                "sqft": site_row.sqft,
                                "occupancy": site_row.occupancy,
                                "notes": site_row.notes,
                                "extras": dict(site_row.extra_fields),
                            },
                        )
                    )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


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

    # Page bucketing thresholds:
    #   < LOW_TEXT_PAGE_THRESHOLD       → marker page only (scanned)
    #   >= TEXT_RICH_PAGE_THRESHOLD     → text-only fast path
    #   else                             → heavyweight layout pipeline
    LOW_TEXT_PAGE_THRESHOLD = 80
    TEXT_RICH_PAGE_THRESHOLD = 1200

    def _build_low_text_page(page_index: int) -> dict[str, Any]:
        # Low-text page = likely scanned. Try the OCR chain
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
        sections = page.get("sections", []) or []
        yield from _atoms_for_sections(
            sections=sections,
            section_path=[],
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        )
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
                if _looks_like_form_field(text) or _looks_like_page_footer(text):
                    continue
                # Apply the same text-pattern classifier the normal
                # paragraph path uses — without it, SLA / decision /
                # constraint / risk shapes that arrive via the
                # fallback all get the default scope_item label.
                atom_type, authority = _classify_text_block(
                    text=text, section_path=[], kind="paragraph"
                )
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
                        "page": page_index,
                        "block_kind": "metadata_fallback",
                        "meta_index": meta_index,
                    },
                    value={"kind": "paragraph", "fallback": "page_metadata"},
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
                        atom_type=AtomType.entity,
                        authority_class=AuthorityClass.contractual_scope,
                        confidence=site_row.confidence,
                        locator={
                            **base_locator,
                            "row_index": site_row.row_index,
                            "extraction": "site_roster_v1",
                        },
                        value={
                            "kind": "physical_site",
                            "site_id": site_row.site_id,
                            "facility_name": site_row.facility_name,
                            "street_address": site_row.street_address,
                            "mdf_idf": site_row.mdf_idf,
                            "access_window": site_row.access_window,
                            "escort_owner": site_row.escort_owner,
                            "contact": site_row.contact,
                            "phone": site_row.phone,
                            "email": site_row.email,
                            "city_state": site_row.city_state,
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


# Filter list for "orphan token" harvesting — common column-header
# words and English filler that shouldn't be treated as symbols.
_LEGEND_TOKEN_BLOCKLIST: frozenset[str] = frozenset({
    "ABOVE", "AFF", "ARCH", "BACK", "CABLE", "CAT6", "CEILING",
    "CLOSET", "CMP", "COAX", "COMPONENT", "COMPONENTS", "CONDUIT",
    "CONTROL", "COOPER", "COPPER", "COUNT", "COUNTER", "COVER",
    "DESCRIPTION", "DEVICE", "DOCK", "DOOR", "DRAWING", "DRAWINGS",
    "ELECTRICAL", "ENTRY", "EQUIP", "ETC", "FINISH", "FLUSH",
    "FRAME", "FROM", "GROUP", "HARDWARE", "HEIGHT", "INSERT",
    "INSTALLATION", "JACK", "LIST", "LOAD", "LOWER", "MANUFACTURERS",
    "MOUNT", "MOUNTED", "MOUNTING", "MUD", "NIC", "NOT", "NOTE",
    "NOTES", "N/A", "NA", "NORMAL", "NUMBER", "OUTLET", "OWNER",
    "PANEL", "PART", "PATCH", "PER", "PLANS", "POE", "PORT",
    "POWER", "PROVIDE", "READER", "REFER", "REMARKS", "REQUIREMENT",
    "REQUIREMENTS", "RING", "RISER", "ROOM", "ROOMS", "ROUGH",
    "ROUGH-IN", "SCHEDULE", "SECONDARY", "SECURITY", "SEE",
    "SHIELDED", "SHOWN", "SIZE", "SPACE", "STANDARD", "STRANDED",
    "STUB", "SUITE", "SYMBOL", "SYMBOLS", "SYSTEM", "TERMINATION",
    "TYPE", "TYPES", "TYPICAL", "TYPICALLY", "UNDER", "UNLESS",
    "UPS", "USE", "USED", "VAULT", "VERIFY", "WALL", "WAREHOUSE",
    "WIRE", "WITH", "WORK", "ZONE",
    "AND", "OR", "FOR", "THE", "ARE", "WAS", "WERE", "ALL", "ANY",
    "PER", "VERIFY", "TBD",
    "A", "B", "C", "D", "E", "F", "G",
    # ----- column letters used as grid coordinates -----
    "A#", "A #",
})


def _augment_legend_with_orphan_tokens(
    *,
    legend: Any,
    per_page_legend_bbox: dict[int, tuple[float, float, float, float]],
    per_page_blocks: dict[int, list[Any]],
) -> Any:
    """Harvest short uppercase tokens from a legend's bbox region.

    The row-parser pairs blocks into (symbol, description) rows but
    occasionally misses the symbol token (multi-column legends with
    wide gaps, columns of nothing-but-icon swatches, etc.). For each
    legend, scan its bbox for short standalone uppercase tokens that
    don't already appear as ``normalized_symbol_text`` in the legend
    entries, and append a synthetic ParsedLegendEntry for each.

    Filtered by ``_LEGEND_TOKEN_BLOCKLIST`` to keep English filler /
    column-header words out of the symbol vocabulary.
    """
    from app.parsers.schematic_models import ParsedLegend, ParsedLegendEntry
    import re

    legend_bbox = per_page_legend_bbox.get(legend.page_index)
    if legend_bbox is None:
        return legend
    blocks = per_page_blocks.get(legend.page_index) or []
    if not blocks:
        return legend

    have: set[str] = set()
    for e in legend.entries:
        s = (e.normalized_symbol_text or "").strip().upper()
        if s:
            have.add(s)
    new_entries: list[ParsedLegendEntry] = list(legend.entries)
    seen_new: set[str] = set()
    # Pattern: short uppercase alphanum tokens, optionally with -, /, or digits
    pat = re.compile(r"^[A-Z][A-Z0-9/\-]{0,5}$")
    for b in blocks:
        bbox = getattr(b, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        # Must lie inside the legend bbox
        if not (
            legend_bbox[0] <= bbox[0]
            and bbox[2] <= legend_bbox[2]
            and legend_bbox[1] <= bbox[1]
            and bbox[3] <= legend_bbox[3]
        ):
            continue
        text = (getattr(b, "text", "") or "").strip()
        if not text or len(text) > 6:
            continue
        upper = text.upper()
        if upper in have or upper in seen_new:
            continue
        if upper in _LEGEND_TOKEN_BLOCKLIST:
            continue
        if not pat.match(upper):
            continue
        # Looks like a real legend symbol — synthesize an entry.
        try:
            entry = ParsedLegendEntry.make(
                page_index=legend.page_index,
                label_text=upper,
                normalized_label=upper.lower(),
                raw_symbol_text=upper,
                normalized_symbol_text=upper,
                symbol_bbox_pdf=tuple(float(x) for x in bbox),
                confidence=0.6,
            )
        except (TypeError, ValueError):
            continue
        new_entries.append(entry)
        seen_new.add(upper)

    if not seen_new:
        return legend
    # Rebuild the ParsedLegend with the new entry tuple. Use make() so
    # legend_id rolls forward to reflect the new entry set.
    return ParsedLegend.make(
        page_index=legend.page_index,
        sheet_number=legend.sheet_number,
        title=legend.title,
        scope=legend.scope,
        entries=tuple(new_entries),
        continuation_refs=legend.continuation_refs,
        source_ref_locator=dict(legend.source_ref_locator),
        confidence=legend.confidence,
        warnings=legend.warnings,
    )


def _run_schematic_pre_pass(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
    domain_pack: DomainPack | None,
) -> tuple[list[EvidenceAtom], list[dict[str, Any]]]:
    """Legend-first schematic pre-pass for a PDF (PR5).

    Returns ``(atoms, derived_files)``.  ``atoms`` is a deterministic
    list of ``schematic_*`` atoms; ``derived_files`` is a list of
    ``ParserDerivedFile`` dicts to attach to ``ParserOutput``.

    Behavior is conservative — if no legend is parsed anywhere in the
    document AND the active domain pack declares no detection
    targets, the pre-pass returns empty results so non-schematic PDFs
    are untouched (preserves the determinism + provenance contracts
    for the existing test grid).
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return [], []
    from app.parsers.schematic_atom_emitters import (
        collect_all,
        emit_detection_atom,
        emit_keyed_note_atom,
        emit_legend_atom,
        emit_line_run_atom,
        emit_room_atom,
        emit_schedule_row_atom,
        emit_sheet_metadata_atom,
        emit_target_set_atom,
        emit_warning_atom,
        intersect_with_pack,
    )
    from app.parsers.schematic_models import DetectionTarget, DetectionTargetSet, SchematicWarning
    from orbitbrief_page_os.segmentation.schematic.legend_locator import (
        locate_legend_candidates,
        page_text_blocks,
    )
    from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend
    from orbitbrief_page_os.segmentation.schematic.legend_resolver import (
        LegendResolver,
        extract_sheet_number,
    )
    from orbitbrief_page_os.segmentation.schematic.symbol_detector import detect_symbols
    from orbitbrief_page_os.segmentation.schematic.raster import is_text_poor_page
    from orbitbrief_page_os.segmentation.schematic import ocr as schematic_ocr
    from orbitbrief_page_os.segmentation.schematic.page_kind_classifier import (
        LEGEND_TABLE,
        SCHEDULE_BOM,
        SPEC_PROSE,
        SCHEMATIC_DRAWING,
        UNKNOWN as PAGE_UNKNOWN,
        classify_page_kind,
    )

    try:
        doc = fitz.open(str(path))
    except Exception:  # pragma: no cover
        return [], []

    resolver = LegendResolver()
    per_page_blocks: dict[int, list[Any]] = {}
    per_page_legend_bbox: dict[int, tuple[float, float, float, float]] = {}
    parsed_legends: list[Any] = []

    atoms: list[EvidenceAtom] = []
    legend_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []

    declared_emitted: set[tuple[str, str]] = set()
    legend_gap_emitted: set[tuple[str, str]] = set()
    pack_has_targets_for_warning = bool(domain_pack and domain_pack.detection_targets)
    try:
        for page_index in range(doc.page_count):
            try:
                page_obj = doc.load_page(page_index)
                blocks = page_text_blocks(page_obj)
            except Exception:
                blocks = []
                page_obj = None
            per_page_blocks[page_index] = blocks
            # Raster fallback: if the page has effectively no text layer
            # AND the active pack expects schematic content, try local
            # OCR to recover legend rows. When OCR is unavailable, emit
            # an ``ocr_unavailable`` warning so the page doesn't silently
            # parse as blank. When OCR IS available, convert recognized
            # words into TextBlocks in PDF-point space and feed them to
            # the rest of the legend pipeline.
            if (
                page_obj is not None
                and pack_has_targets_for_warning
                and not blocks
                and is_text_poor_page(page_obj)
            ):
                if not schematic_ocr.is_available():
                    atoms.append(
                        emit_warning_atom(
                            warning=schematic_ocr.status_warning(
                                page_index=page_index, sheet_number=None
                            ),
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=page_obj,
                        )
                    )
                else:
                    from orbitbrief_page_os.segmentation.schematic.raster import (
                        render_page_to_ndarray,
                    )
                    from app.parsers.schematic_models import SCHEMATIC_REPLAY_DPI

                    arr = render_page_to_ndarray(page_obj, dpi=SCHEMATIC_REPLAY_DPI)
                    if arr is not None:
                        words = schematic_ocr.ocr_words(arr)
                        ocr_blocks = schematic_ocr.words_to_textblocks(
                            words, page_dpi=SCHEMATIC_REPLAY_DPI
                        )
                        if ocr_blocks:
                            blocks = ocr_blocks
                            per_page_blocks[page_index] = ocr_blocks
                            atoms.append(
                                emit_warning_atom(
                                    warning=SchematicWarning.make(
                                        warning_type="ocr_recovered",
                                        page_index=page_index,
                                        sheet_number=None,
                                        detail=(
                                            f"Raster page parsed via OCR "
                                            f"({len(ocr_blocks)} text rows recovered)."
                                        ),
                                        extras={
                                            "ocr_word_count": len(words),
                                            "ocr_block_count": len(ocr_blocks),
                                        },
                                    ),
                                    project_id=project_id,
                                    artifact_id=artifact_id,
                                    filename=path.name,
                                    parser_version=parser_version,
                                    page=page_obj,
                                )
                            )
            # ── Page-kind routing (PR: Marriott multi-legend fix) ──
            # Classify the page so we (a) skip prose/schedule pages
            # and (b) extract MULTIPLE legends from legend-table
            # pages instead of bailing on the first match.
            classification = classify_page_kind(
                page_index=page_index, page=page_obj, blocks=blocks
            )
            page_kind = classification.kind

            # SPEC_PROSE + SCHEDULE_BOM pages have no schematic content;
            # skip the entire legend/symbol flow. The generic PDF parser
            # (table/text extraction) handles these pages.
            if page_kind in (SPEC_PROSE, SCHEDULE_BOM):
                # Still ingest into resolver so cross-doc state is
                # consistent (it just produces no legends/targets).
                page_bbox_for_ingest_skip: tuple[float, float, float, float] | None = None
                if page_obj is not None:
                    try:
                        r = page_obj.rect
                        page_bbox_for_ingest_skip = (
                            float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                        )
                    except Exception:  # pragma: no cover
                        page_bbox_for_ingest_skip = None
                resolver.ingest_page(
                    page_index=page_index,
                    blocks=blocks,
                    legend=None,
                    page_bbox=page_bbox_for_ingest_skip,
                )
                continue

            legend = None
            candidates = locate_legend_candidates(page_index=page_index, blocks=blocks)
            ordered = sorted(
                (c for c in candidates if c.score >= 0.45),
                key=lambda c: (-c.score, c.page_index, c.bbox[1], c.bbox[0]),
            )
            chosen_bbox: tuple[float, float, float, float] | None = None
            sheet = extract_sheet_number(blocks)

            # LEGEND_TABLE pages contain MULTIPLE legends (Marriott
            # T0.01 = Structured Cabling + Intrusion + Access Control
            # + CCTV). Extract every non-bogus candidate; promote
            # scope to ``global`` since the legend applies to all
            # subsequent drawing pages with the same domain.
            if page_kind == LEGEND_TABLE:
                page_legends: list[Any] = []
                seen_legend_ids: set[str] = set()
                seen_bbox_centers: list[tuple[float, float]] = []
                # Marriott T0.01 has FOUR legend tables (STRUCTURED
                # CABLING + INTRUSION DETECTION + ACCESS CONTROL +
                # CCTV) — the locator normalizes their headers to the
                # same string ("symbol legend"), so deduping by header
                # text used to collapse all four into one. Instead,
                # dedupe by the parsed legend_id (entry-set hash) and
                # by bbox-center proximity so distinct legends survive.
                BBOX_DUPE_PT = 36.0
                for cand in ordered:
                    cx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                    cy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                    if any(
                        abs(cx - sc[0]) <= BBOX_DUPE_PT and abs(cy - sc[1]) <= BBOX_DUPE_PT
                        for sc in seen_bbox_centers
                    ):
                        continue
                    parsed = parse_legend(
                        candidate=cand,
                        page_blocks=blocks,
                        sheet_number=sheet,
                        scope="global",
                    )
                    if parsed is None:
                        continue
                    if parsed.legend_id in seen_legend_ids:
                        continue
                    seen_legend_ids.add(parsed.legend_id)
                    seen_bbox_centers.append((cx, cy))
                    page_legends.append(parsed)
                    if chosen_bbox is None:
                        chosen_bbox = cand.bbox

                # Promote the in-loop ``legend`` to the first parsed
                # (for the ``if legend is not None`` block below); the
                # rest get appended directly to parsed_legends.
                if page_legends:
                    legend = page_legends[0]
                    parsed_legends.extend(page_legends[1:])
            else:
                # SCHEMATIC_DRAWING / COVER_TITLE / UNKNOWN — keep
                # current "first non-empty candidate wins" behavior.
                for cand in ordered:
                    scope = "global" if (cand.header_text and "symbols & legends" in cand.header_text) else "page"
                    legend = parse_legend(
                        candidate=cand,
                        page_blocks=blocks,
                        sheet_number=sheet,
                        scope=scope,  # type: ignore[arg-type]
                    )
                    if legend is not None:
                        chosen_bbox = cand.bbox
                        break
            page_bbox_for_ingest: tuple[float, float, float, float] | None = None
            if page_obj is not None:
                try:
                    r = page_obj.rect
                    page_bbox_for_ingest = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                except Exception:  # pragma: no cover
                    page_bbox_for_ingest = None
            resolver.ingest_page(
                page_index=page_index,
                blocks=blocks,
                legend=legend,
                page_bbox=page_bbox_for_ingest,
            )
            if legend is not None:
                parsed_legends.append(legend)
                if chosen_bbox is not None:
                    per_page_legend_bbox[page_index] = chosen_bbox

        pack_has_targets = bool(domain_pack and domain_pack.detection_targets)
        if not parsed_legends and not pack_has_targets:
            return [], []

        # Vision-LLM symbol detection bootstrap. Extract legend symbol
        # crops once per document so they can be reused across every
        # SCHEMATIC_DRAWING page during the per-page detection loop.
        # Opt-in via PARSER_OS_VISION_DETECT=1 so default compiles stay
        # byte-stable for the existing test grid.
        vision_legend_crops: list[Any] = []
        vision_enabled = os.environ.get("PARSER_OS_VISION_DETECT") == "1"
        vision_cache_path: Path | None = None
        if vision_enabled and parsed_legends:
            try:
                from orbitbrief_page_os.segmentation.schematic.legend_symbol_crops import (
                    extract_legend_symbol_crops,
                )
                from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
                    is_vision_endpoint_reachable,
                )
            except Exception:  # pragma: no cover
                extract_legend_symbol_crops = None  # type: ignore[assignment]
                is_vision_endpoint_reachable = None  # type: ignore[assignment]
            if extract_legend_symbol_crops is not None and is_vision_endpoint_reachable is not None:
                if is_vision_endpoint_reachable():
                    crops_out_dir = derived_dir_for(path)
                    try:
                        crops_out_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:  # pragma: no cover
                        pass
                    try:
                        vision_legend_crops = extract_legend_symbol_crops(
                            legends=parsed_legends,
                            pdf_path=path,
                            out_dir=crops_out_dir,
                        )
                    except Exception:  # pragma: no cover
                        vision_legend_crops = []
                    vision_cache_path = path.parent / ".orbitbrief_vision_detect_cache.jsonl"

        for legend in parsed_legends:
            try:
                legend_page = doc.load_page(legend.page_index)
            except Exception:  # pragma: no cover
                legend_page = None
            atoms.append(
                emit_legend_atom(
                    legend=legend,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=legend_page,
                )
            )
            legend_records.append(
                {
                    "legend_id": legend.legend_id,
                    "page": legend.page_index,
                    "sheet_number": legend.sheet_number,
                    "scope": legend.scope,
                    "entries": [
                        {
                            "entry_id": e.entry_id,
                            "symbol": e.raw_symbol_text,
                            "label": e.label_text,
                            "normalized_label": e.normalized_label,
                            "count_column": e.count_column,
                        }
                        for e in legend.entries
                    ],
                }
            )

        # Per-page resolution + target-set emission. Pages without a sheet
        # number AND without a parsed legend on them are skipped: this
        # is the discriminator that prevents non-drawing PDFs from being
        # spammed with ``missing_legend`` warnings.
        for page_index in sorted(per_page_blocks):
            blocks = per_page_blocks[page_index]
            sheet = extract_sheet_number(blocks)
            own_legend = any(l.page_index == page_index for l in parsed_legends)
            # The pack-with-targets case: even if a drawing-like page has
            # no extractable sheet number, the active domain pack
            # expects schematic context. Routing it through the resolver
            # surfaces a ``missing_legend`` warning instead of silently
            # dropping the page (boss-review fix).
            pack_expects_schematic = bool(domain_pack and domain_pack.detection_targets)
            page_text_density = sum(len((b.text or "").strip()) for b in blocks)
            # Image-only drawing detection: if the page has effectively no
            # text BUT the document has parsed legends from other pages
            # AND the active pack expects schematic content, we still want
            # to run the glyph-template matcher against the raster page so
            # symbol counts come back instead of vanishing silently.
            raster_only_page = (
                pack_expects_schematic
                and parsed_legends
                and not blocks
            )
            if sheet is None and not own_legend and not (
                pack_expects_schematic and page_text_density >= 40
            ) and not raster_only_page:
                continue
            try:
                page = doc.load_page(page_index)
            except Exception:  # pragma: no cover
                page = None
            resolved = resolver.resolve_for_page(page_index)
            for warning in resolved.warnings:
                atoms.append(
                    emit_warning_atom(
                        warning=warning,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )
            if resolved.legend is None:
                continue
            if domain_pack is not None:
                targets, gaps = intersect_with_pack(
                    legend=resolved.legend, pack=domain_pack
                )
                pack_id_for_set = domain_pack.pack_id
            else:
                targets, gaps = [], []
                pack_id_for_set = "legend_only"

            # When the active pack doesn't intersect the legend (e.g.
            # fiber pack vs telecom legend), synthesize one target per
            # legend entry so the text-tag + vision detectors have
            # something to look for. This keeps the parser universal:
            # a real DD with WN/CR/TV symbols produces detections
            # regardless of which domain pack is loaded.
            if not targets:
                # Augment the legend with orphan symbol tokens.
                # Real legends have one column of short symbol tokens
                # (WN / CR / ZN / DC / FACP-2 / MATV / etc.) but the
                # row-parser occasionally fails to pair a token with
                # its description, so the resulting entry list omits
                # the symbol. Scan the legend bbox for standalone
                # uppercase tokens and synthesize entries for any
                # that aren't already represented.
                augmented_legend = _augment_legend_with_orphan_tokens(
                    legend=resolved.legend,
                    per_page_legend_bbox=per_page_legend_bbox,
                    per_page_blocks=per_page_blocks,
                )
                synthesized: list[DetectionTarget] = []
                for entry in augmented_legend.entries:
                    key_seed = (
                        entry.normalized_symbol_text
                        or entry.normalized_label
                        or entry.entry_id
                    )
                    if not key_seed:
                        continue
                    tk = key_seed.lower().strip()
                    ek = f"device:{tk}".replace(" ", "_")
                    try:
                        synthesized.append(
                            DetectionTarget(
                                target_key=tk,
                                entity_key=ek,
                                completeness="informational",
                                expected_modalities=("text_tag", "vision_llm"),
                                legend_entry_id=entry.entry_id,
                                aliases=tuple(
                                    a for a in (
                                        entry.raw_symbol_text or "",
                                        entry.normalized_symbol_text or "",
                                        entry.label_text or "",
                                        entry.normalized_label or "",
                                    ) if a
                                ),
                            )
                        )
                    except ValueError:
                        continue
                targets = synthesized
                pack_id_for_set = "legend_only"
                # Replace resolved.legend with the augmented copy so
                # downstream code (symbol detector, atom emitters)
                # see the harvested entries too.
                import dataclasses as _dc
                try:
                    resolved = _dc.replace(resolved, legend=augmented_legend)
                except (TypeError, ValueError):  # pragma: no cover
                    pass

            target_set = DetectionTargetSet.make(
                page_index=page_index,
                sheet_number=sheet,
                pack_id=pack_id_for_set,
                legend_id=resolved.legend.legend_id,
                targets=tuple(targets),
                legend_gap_target_keys=tuple(gaps),
            )
            page_bbox: tuple[float, float, float, float] | None = None
            if page is not None:
                try:
                    rect = page.rect
                    page_bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                except Exception:  # pragma: no cover
                    page_bbox = None
            atoms.append(
                emit_target_set_atom(
                    target_set=target_set,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=page,
                    page_bbox=page_bbox,
                )
            )
            target_records.append(
                {
                    "page": page_index,
                    "sheet_number": sheet,
                    "legend_id": resolved.legend.legend_id,
                    "rationale": resolved.rationale,
                    "priority": resolved.priority,
                    "targets": [t.target_key for t in targets],
                    "legend_gap_target_keys": list(gaps),
                }
            )
            # legend_gap warnings: pack declared the target as
            # load-bearing but the resolved legend doesn't mention it.
            # Attach the legend's bbox so source_replay still verifies
            # the receipt against pixels (rather than emitting a
            # locator with only a page index).
            legend_bbox_for_gap = per_page_legend_bbox.get(resolved.legend.page_index)
            legend_page_for_gap = None
            try:
                legend_page_for_gap = doc.load_page(resolved.legend.page_index)
            except Exception:  # pragma: no cover
                pass
            for gap_key in gaps:
                # Dedupe: emit each (legend_id, target_key) gap once
                # regardless of how many drawing pages resolve to the
                # same legend.
                dedup = (resolved.legend.legend_id, gap_key)
                if dedup in legend_gap_emitted:
                    continue
                legend_gap_emitted.add(dedup)
                atoms.append(
                    emit_warning_atom(
                        warning=SchematicWarning.make(
                            warning_type="legend_gap",
                            page_index=resolved.legend.page_index,
                            sheet_number=resolved.legend.sheet_number,
                            detail=f"Pack '{domain_pack.pack_id}' declares load-bearing target '{gap_key}' but legend has no matching entry",
                            target_key=gap_key,
                            legend_id=resolved.legend.legend_id,
                            bbox_pdf=legend_bbox_for_gap,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=legend_page_for_gap,
                    )
                )

            # Symbol detection (PR6) — run when we have a resolved
            # legend AND either:
            #   (a) a non-empty pack target set, OR
            #   (b) vision-LLM detection is enabled + crops exist
            #
            # The original guard only allowed pack-matched targets,
            # which silenced vision detection whenever the domain
            # pack didn't match the legend vocabulary (e.g. running
            # the fiber pack against a security/telecom legend). The
            # vision detector matches directly against legend
            # entries, so it can fire even when no pack target
            # intersects.
            vision_can_run = bool(vision_enabled and vision_legend_crops)
            if page is None or (not target_set.targets and not vision_can_run):
                continue
            try:
                legend_page = doc.load_page(resolved.legend.page_index)
            except Exception:  # pragma: no cover
                continue
            excluded: list[tuple[float, float, float, float]] = []
            if resolved.legend.page_index in per_page_legend_bbox:
                if resolved.legend.page_index == page_index:
                    excluded.append(per_page_legend_bbox[resolved.legend.page_index])
            # Additional exclusion zones — title block, drawing index,
            # keyed notes, and schedules. Without these, a "PTZ" inside
            # "PTZ ROOM" or a schedule cell gets counted as a detection.
            from orbitbrief_page_os.segmentation.schematic.exclusion_zones import (
                detect_exclusion_zones,
            )
            from orbitbrief_page_os.segmentation.schematic.sheet_metadata import (
                parse_sheet_metadata,
            )
            from orbitbrief_page_os.segmentation.schematic.rooms import (
                Room,
                assign_detections_to_rooms,
                detect_rooms,
            )
            from orbitbrief_page_os.segmentation.schematic.keyed_notes import (
                detect_keyed_notes,
            )

            zones = detect_exclusion_zones(blocks, page_bbox=page_bbox)
            for zone in zones:
                excluded.append(zone.bbox)

            # Sheet metadata atom — one per drawing page that carries
            # an extractable title block.
            title_block_bbox = next(
                (z.bbox for z in zones if z.label == "title_block"),
                None,
            )
            try:
                sheet_meta = parse_sheet_metadata(
                    page_index=page_index,
                    blocks=blocks,
                    sheet_number=sheet,
                    title_block_bbox=title_block_bbox,
                )
            except Exception:  # pragma: no cover
                sheet_meta = None
            if sheet_meta is not None:
                # Suppress fieldless sheet_metadata atoms: a sheet
                # number alone is already captured elsewhere
                # (target_set, legend, detections). Only emit when
                # at least one substantive title-block field was
                # parsed.
                substantive = any([
                    sheet_meta.sheet_title,
                    sheet_meta.project_name,
                    sheet_meta.scale,
                    sheet_meta.issue_date,
                    sheet_meta.revision,
                    sheet_meta.drafter,
                    sheet_meta.checker,
                    sheet_meta.approver,
                    sheet_meta.client,
                ])
                if substantive:
                    atoms.append(
                        emit_sheet_metadata_atom(
                            metadata=sheet_meta,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=page,
                        )
                    )

            # Room / zone atoms — pulled from blocks outside the
            # excluded zones so we don't pick up schedule-row room IDs.
            try:
                rooms_on_page: list[Room] = detect_rooms(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                    excluded_bboxes=tuple(excluded),
                )
            except Exception:  # pragma: no cover
                rooms_on_page = []
            for room in rooms_on_page:
                atoms.append(
                    emit_room_atom(
                        room=room,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Keyed-notes atoms — both the note rows and their resolved
            # body callouts. The exclusion-zone pass already keeps the
            # block out of symbol detection; this turns the contents
            # into reviewable atoms.
            try:
                keyed_notes_on_page = detect_keyed_notes(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                keyed_notes_on_page = []
            for note in keyed_notes_on_page:
                atoms.append(
                    emit_keyed_note_atom(
                        note=note,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Construction schedule rows — door / camera / equipment /
            # fixture / panel schedules.  Each row joins to a detection
            # by tag downstream (after detect_symbols runs).
            from orbitbrief_page_os.segmentation.schematic.schedules import (
                detect_schedules,
                join_schedule_rows_to_detections,
            )

            try:
                schedule_rows_on_page = detect_schedules(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                schedule_rows_on_page = []
            for row in schedule_rows_on_page:
                atoms.append(
                    emit_schedule_row_atom(
                        row=row,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Prose-with-symbol suppression: any text block whose text
            # contains a legend symbol but ISN'T a standalone label
            # (e.g. "PTZ ROOM", "Card Reader Suite") must be added to
            # the exclusion set so the glyph_template matcher does
            # not catch the symbol's pixels inside the prose word.
            # The text-tag matcher already filters via
            # _block_text_is_standalone_symbol; glyph_template needs
            # the bboxes excluded explicitly because it operates on
            # rendered pixels.
            from orbitbrief_page_os.segmentation.schematic.symbol_detector import (
                _block_text_is_standalone_symbol,
            )

            legend_symbol_tokens: dict[str, Any] = {
                (e.normalized_symbol_text or "").upper(): e
                for e in resolved.legend.entries
                if e.normalized_symbol_text
            }
            for blk in blocks:
                text = (blk.text or "").strip()
                if not text:
                    continue
                upper = text.upper()
                if not any(
                    sym in upper.split() or sym + " " in upper or " " + sym in upper or upper == sym
                    for sym in legend_symbol_tokens
                ):
                    continue
                if not _block_text_is_standalone_symbol(text, legend_symbol_tokens):
                    excluded.append(blk.bbox)
            detections = detect_symbols(
                page=page,
                page_index=page_index,
                sheet_number=sheet,
                blocks=blocks,
                target_set=target_set,
                legend=resolved.legend,
                legend_page=legend_page,
                excluded_bboxes=tuple(excluded),
            )
            # Vision-LLM augmentation for SCHEMATIC_DRAWING pages.
            # On real schematics the symbol IS an icon, not text — the
            # text-tag detector returns 0 hits. Vision detector finds
            # icons via region proposals + qwen2.5vl match against the
            # legend symbol crops. Only runs when the endpoint is
            # reachable + at least one legend crop was extracted.
            classification_for_page = classify_page_kind(
                page_index=page_index, page=page, blocks=blocks
            ) if page is not None else None
            page_kind_for_vision = (
                classification_for_page.kind if classification_for_page else PAGE_UNKNOWN
            )
            if (
                vision_enabled
                and vision_legend_crops
                and page_kind_for_vision in (SCHEMATIC_DRAWING, PAGE_UNKNOWN)
                and page is not None
            ):
                try:
                    from orbitbrief_page_os.segmentation.schematic.region_proposals import (
                        propose_regions,
                    )
                    from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
                        detect_symbols_via_vision,
                    )
                except Exception:  # pragma: no cover
                    propose_regions = None  # type: ignore[assignment]
                    detect_symbols_via_vision = None  # type: ignore[assignment]
                if propose_regions is not None and detect_symbols_via_vision is not None:
                    try:
                        proposals = propose_regions(page=page, page_index=page_index)
                    except Exception:  # pragma: no cover
                        proposals = []
                    if proposals:
                        try:
                            vision_dets = detect_symbols_via_vision(
                                page=page,
                                page_index=page_index,
                                region_proposals=proposals,
                                legend_crops=vision_legend_crops,
                                cache_path=vision_cache_path,
                            )
                        except Exception:  # pragma: no cover
                            vision_dets = []
                        # Convert VisionDetection → SymbolDetection so the
                        # downstream emit pipeline treats them uniformly
                        # with the text_tag detections.
                        from app.parsers.schematic_models import SymbolDetection as _SymbolDetection
                        entry_by_id = {
                            e.entry_id: e
                            for l in parsed_legends
                            for e in l.entries
                        }
                        target_by_entry_id: dict[str, Any] = {}
                        for t in target_set.targets:
                            if t.legend_entry_id:
                                target_by_entry_id[t.legend_entry_id] = t
                        for vd in vision_dets:
                            entry = entry_by_id.get(vd.matched_entry_id)
                            if entry is None:
                                continue
                            target = target_by_entry_id.get(vd.matched_entry_id)
                            # When the active pack doesn't intersect the
                            # legend (e.g. running the fiber pack on a
                            # security/telecom legend), synthesize a
                            # target_key from the entry itself so the
                            # vision detection isn't dropped.
                            if target is not None:
                                target_key = target.target_key
                                entity_key = target.target_key
                            else:
                                target_key = (
                                    entry.normalized_label
                                    or (entry.normalized_symbol_text or "")
                                    or entry.entry_id
                                )
                                entity_key = f"device:{target_key}".lower().replace(" ", "_")
                            try:
                                sd = _SymbolDetection.make(
                                    page_index=page_index,
                                    sheet_number=sheet,
                                    target_key=target_key,
                                    entity_key=entity_key,
                                    legend_entry_id=entry.entry_id,
                                    bbox_pdf=vd.bbox_pdf,
                                    crop_sha256="",
                                    modality="vision_llm",
                                    confidence=vd.confidence,
                                    nearby_text=vd.matched_label_text,
                                )
                            except (TypeError, ValueError):
                                continue
                            detections.append(sd)
            # Assign each detection to its nearest room (when rooms
            # were detected on this page). The mapping is recorded
            # on the detection atom's value so downstream consumers
            # can group counts by room without re-running geometry.
            detection_room_map: dict[str, str] = {}
            if rooms_on_page:
                try:
                    detection_room_map = assign_detections_to_rooms(
                        detections, rooms_on_page
                    )
                except Exception:  # pragma: no cover
                    detection_room_map = {}

            # Mounting-height callouts — attach the nearest one to each
            # detection so a CR atom carries "48 AFF" without the
            # reviewer opening the PDF.
            from orbitbrief_page_os.segmentation.schematic.callouts import (
                attach_callouts_to_detections,
                detect_callouts,
            )

            try:
                callouts_on_page = detect_callouts(blocks, excluded_bboxes=tuple(excluded))
                detection_callout_map = attach_callouts_to_detections(
                    detections, callouts_on_page
                )
            except Exception:  # pragma: no cover
                detection_callout_map = {}

            # Mounting-height inheritance chain (PM-critical):
            #   1. nearest inline callout (set above)
            #   2. schedule row's "mounting" / "mounting_height" field
            #   3. legend entry's MOUNTING / MOUNTING HEIGHT attribute
            #   4. keyed-note default ("All devices mounted at X AFF
            #      unless noted") — derived once per page
            import re as _re

            keyed_note_default_height: str | None = None
            for note in keyed_notes_on_page:
                m = _re.search(
                    r"(?:mounted|mounting)\s+(?:at|height)?\s*"
                    r"([0-9]+(?:\.[0-9]+)?\s*(?:\"|in|inches)?\s*"
                    r"a\.?f\.?f\.?|"
                    r"[0-9]+\s*'\s*-\s*[0-9]+(?:\s*[0-9]+/[0-9]+)?\s*\"|"
                    r"ceiling|"
                    r"verify\s+w/?\s*arch)",
                    note.text,
                    _re.IGNORECASE,
                )
                if m:
                    keyed_note_default_height = m.group(1).strip()
                    break

            legend_mounting_by_entry: dict[str, str] = {}
            legend_responsibility_by_entry: dict[str, str] = {}
            legend_remarks_by_entry: dict[str, str] = {}
            for entry in resolved.legend.entries:
                attrs = dict(entry.attributes)
                m_val = (
                    attrs.get("mounting_height")
                    or attrs.get("mounting")
                )
                if m_val:
                    legend_mounting_by_entry[entry.entry_id] = m_val
                # Responsibility / by-others markers — explicit
                # ``responsibility`` column wins; otherwise scan
                # the remarks column for the conventional phrases.
                resp_val: str | None = attrs.get("responsibility")
                remarks_text = attrs.get("remarks") or ""
                if not resp_val and remarks_text:
                    upper = remarks_text.upper()
                    for marker in ("NIC", "BY OWNER", "BY GC", "BY OTHERS", "NOT IN CONTRACT"):
                        if marker in upper:
                            resp_val = marker
                            break
                if resp_val:
                    legend_responsibility_by_entry[entry.entry_id] = resp_val
                if remarks_text:
                    legend_remarks_by_entry[entry.entry_id] = remarks_text

            # Schedule-row joins — pass 1 is nearby_text tag match,
            # pass 2 is spatial join when a TAG block sits within
            # ~2 inches of the detection center.
            try:
                detection_schedule_map = join_schedule_rows_to_detections(
                    schedule_rows_on_page,
                    detections,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                detection_schedule_map = {}

            # Line runs — conduit / cable / riser polylines, snapped
            # to nearby detections. Emitted AFTER detections so the
            # snap targets are deterministic.
            from orbitbrief_page_os.segmentation.schematic.line_runs import (
                detect_line_runs,
            )

            try:
                line_runs_on_page = detect_line_runs(
                    page=page,
                    page_index=page_index,
                    sheet_number=sheet,
                    detections=detections,
                    excluded_bboxes=tuple(excluded),
                )
            except Exception:  # pragma: no cover
                line_runs_on_page = []
            for line_run in line_runs_on_page:
                atoms.append(
                    emit_line_run_atom(
                        line_run=line_run,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            for det in detections:
                room_id = detection_room_map.get(det.detection_id)
                callout = detection_callout_map.get(det.detection_id)
                schedule_row = detection_schedule_map.get(det.detection_id)
                atom = emit_detection_atom(
                    detection=det,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                )
                updates: dict[str, Any] = {}
                new_value = dict(atom.value)
                new_entity_keys = list(atom.entity_keys)
                if room_id:
                    new_value["located_in_room_id"] = room_id
                    # Look up the room's human-readable label/number
                    # so downstream consumers don't have to join on
                    # the opaque room hash.
                    room_obj = next(
                        (r for r in rooms_on_page if r.room_id == room_id),
                        None,
                    )
                    if room_obj is not None:
                        new_value["located_in_room_label"] = room_obj.label
                        if room_obj.number:
                            new_value["located_in_room_number"] = room_obj.number
                            new_value["located_in_room_display"] = (
                                f"{room_obj.label} {room_obj.number}"
                            )
                        else:
                            new_value["located_in_room_display"] = room_obj.label
                    new_entity_keys.append(f"room:{room_id}")
                # Mounting-height inheritance chain.
                resolved_height: str | None = None
                height_source: str | None = None
                if callout is not None:
                    resolved_height = callout.text
                    height_source = "inline_callout"
                    new_value["callout_bbox"] = list(callout.bbox)
                if schedule_row is not None:
                    new_value["schedule_row_id"] = schedule_row.row_id
                    new_value["schedule_tag"] = schedule_row.tag
                    new_value["schedule_kind"] = schedule_row.schedule_kind
                    new_value["schedule_fields"] = dict(schedule_row.fields)
                    new_entity_keys.append(f"schedule_tag:{schedule_row.tag}")
                    if resolved_height is None:
                        sched_height = (
                            schedule_row.fields_dict().get("mounting_height")
                            or schedule_row.fields_dict().get("mounting")
                        )
                        if sched_height:
                            resolved_height = sched_height
                            height_source = "schedule"
                # Legend column fallback.
                if resolved_height is None and det.legend_entry_id:
                    legend_height = legend_mounting_by_entry.get(det.legend_entry_id)
                    if legend_height:
                        resolved_height = legend_height
                        height_source = "legend_column"
                # Keyed-note default fallback ("X AFF unless noted").
                if resolved_height is None and keyed_note_default_height:
                    resolved_height = keyed_note_default_height
                    height_source = "keyed_note_default"
                if resolved_height is not None:
                    new_value["mounting_height"] = resolved_height
                    new_value["mounting_height_source"] = height_source

                # Responsibility / NIC markers (PM-critical for scope).
                if det.legend_entry_id:
                    resp_val = legend_responsibility_by_entry.get(det.legend_entry_id)
                    if resp_val:
                        new_value["responsibility"] = resp_val
                        new_entity_keys.append(
                            f"responsibility:{resp_val.lower().replace(' ', '_')}"
                        )
                    remarks_val = legend_remarks_by_entry.get(det.legend_entry_id)
                    if remarks_val:
                        new_value["legend_remarks"] = remarks_val
                # Trigger the update when ANY field was added or
                # ANY new entity_key was appended.  The earlier code
                # only checked the room/callout/schedule trio, which
                # silently dropped keyed-note-default heights,
                # legend-column heights, and responsibility markers
                # on detections with no room/callout/schedule.
                if new_value != atom.value or new_entity_keys != list(atom.entity_keys):
                    updates["value"] = new_value
                    updates["entity_keys"] = sorted(set(new_entity_keys))
                if updates:
                    atom = atom.model_copy(update=updates)
                atoms.append(atom)
                detection_records.append(
                    {
                        "detection_id": det.detection_id,
                        "page": det.page_index,
                        "target_key": det.target_key,
                        "modality": det.modality,
                        "bbox": list(det.bbox_pdf),
                        "crop_sha256": det.crop_sha256,
                        "confidence": det.confidence,
                        "located_in_room_id": room_id,
                        "mounting_height": callout.text if callout else None,
                        "schedule_row_id": schedule_row.row_id if schedule_row else None,
                        "schedule_tag": schedule_row.tag if schedule_row else None,
                    }
                )

            # Schematic quantity aggregation (PR7) — turn detection
            # counts into ``AtomType.quantity`` atoms and emit a
            # declared-count atom from any legend row that has a
            # count_column. Same-sheet conflicts are paired by
            # ``_build_schematic_quantity_edges`` in the graph builder.
            from app.parsers.schematic_atom_emitters import (
                emit_declared_count_atom,
                emit_detected_count_atom,
            )

            counts_by_target: dict[str, list] = {}
            for det in detections:
                counts_by_target.setdefault(det.target_key, []).append(det)
            for target in target_set.targets:
                hits = counts_by_target.get(target.target_key, [])
                detected_atom = emit_detected_count_atom(
                    page_index=page_index,
                    sheet_number=sheet,
                    target=target,
                    detections=hits,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                )
                if detected_atom is not None:
                    atoms.append(detected_atom)

                # legend_orphan: load-bearing target declared by the
                # legend but zero detections on this drawing body.
                # Boss-review fix — previously declared but never emitted.
                if (
                    not hits
                    and target.completeness == "load_bearing"
                    and target.legend_entry_id is not None
                ):
                    orphan_entry = next(
                        (e for e in resolved.legend.entries if e.entry_id == target.legend_entry_id),
                        None,
                    )
                    orphan_bbox = orphan_entry.symbol_bbox_pdf if orphan_entry else None
                    atoms.append(
                        emit_warning_atom(
                            warning=SchematicWarning.make(
                                warning_type="legend_orphan",
                                page_index=page_index,
                                sheet_number=sheet,
                                detail=(
                                    f"Legend entry for load-bearing target "
                                    f"'{target.target_key}' produced zero detections on this page."
                                ),
                                target_key=target.target_key,
                                legend_id=resolved.legend.legend_id,
                                legend_entry_id=target.legend_entry_id,
                                bbox_pdf=orphan_bbox,
                            ),
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=legend_page,
                        )
                    )

                if target.legend_entry_id is None:
                    continue
                # Walk the legend for the declared count for this entry.
                # Emit the declared atom only once per (target, legend_entry)
                # pair — without this guard the same declared count would
                # be re-emitted for every drawing page that resolves to
                # the same legend.
                dedup_key = (target.target_key, target.legend_entry_id)
                if dedup_key in declared_emitted:
                    continue
                for entry in resolved.legend.entries:
                    if entry.entry_id != target.legend_entry_id:
                        continue
                    if entry.count_column is None:
                        continue
                    declared = emit_declared_count_atom(
                        page_index=resolved.legend.page_index,
                        sheet_number=resolved.legend.sheet_number,
                        target=target,
                        declared_count=entry.count_column,
                        entry=entry,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=legend_page,
                    )
                    if declared is not None:
                        atoms.append(declared)
                        declared_emitted.add(dedup_key)
                    else:
                        # Provenance gate refused (no symbol bbox or no
                        # crop hash available). Emit a low-confidence
                        # warning so the count isn't silently lost.
                        atoms.append(
                            emit_warning_atom(
                                warning=SchematicWarning.make(
                                    warning_type="weak_declared_count_provenance",
                                    page_index=resolved.legend.page_index,
                                    sheet_number=resolved.legend.sheet_number,
                                    detail=(
                                        f"Legend declared count={entry.count_column} for target "
                                        f"'{target.target_key}' but the row had no replayable bbox; "
                                        f"declared-count atom suppressed."
                                    ),
                                    target_key=target.target_key,
                                    legend_id=resolved.legend.legend_id,
                                    legend_entry_id=target.legend_entry_id,
                                ),
                                project_id=project_id,
                                artifact_id=artifact_id,
                                filename=path.name,
                                parser_version=parser_version,
                                page=legend_page,
                            )
                        )
                    break

            # ``unknown_symbol`` warnings: tokens that look like
            # legend-style symbol tags but matched no legend entry.
            atoms.extend(
                _unknown_symbol_warnings(
                    blocks=blocks,
                    page_index=page_index,
                    sheet=sheet,
                    legend=resolved.legend,
                    excluded_bboxes=excluded,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=page,
                )
            )
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover
            pass

    derived_relative = derived_dir_for(path).name
    derived_files: list[dict[str, Any]] = [
        {
            "relative_path": f"{derived_relative}/schematic_legends.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.legends.v1", "legends": legend_records},
        },
        {
            "relative_path": f"{derived_relative}/schematic_targets.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.targets.v1", "pages": target_records},
        },
        {
            "relative_path": f"{derived_relative}/schematic_detections.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.detections.v1", "detections": detection_records},
        },
    ]
    # Optional debug-overlay sidecars. The flag is opt-in via the
    # ``PARSER_OS_SCHEMATIC_OVERLAYS`` env var so default compiles
    # still produce byte-identical output. When set, one PNG per
    # drawing page is written under ``<stem>.derived/overlays/`` and
    # an ``schematic_overlays.json`` manifest is added so downstream
    # consumers (OrbitBrief envelope renderer, debug viewer) can
    # find them deterministically.
    if os.environ.get("PARSER_OS_SCHEMATIC_OVERLAYS") == "1" and parsed_legends:
        try:
            from orbitbrief_page_os.segmentation.schematic.debug_overlay import render_overlay
        except Exception:
            render_overlay = None  # type: ignore[assignment]
        if render_overlay is not None:
            try:
                overlay_doc = fitz.open(str(path))
            except Exception:  # pragma: no cover
                overlay_doc = None
            overlay_manifest: list[dict[str, Any]] = []
            target_pages = sorted({rec["page"] for rec in target_records})
            if overlay_doc is not None:
                try:
                    for page_index in target_pages:
                        try:
                            overlay_page = overlay_doc.load_page(page_index)
                        except Exception:  # pragma: no cover
                            continue
                        page_detections = [
                            d for d in detection_records if d.get("page") == page_index
                        ]
                        legends_here = [
                            l for l in parsed_legends if l.page_index == page_index
                        ]
                        # debug_overlay.render_overlay expects SymbolDetection
                        # records, not raw dicts — rebuild lightweight stand-ins.
                        from app.parsers.schematic_models import SymbolDetection

                        dets: list[SymbolDetection] = []
                        for d in page_detections:
                            bbox = d.get("bbox") or [0, 0, 1, 1]
                            try:
                                dets.append(
                                    SymbolDetection.make(
                                        page_index=int(d.get("page", page_index)),
                                        sheet_number=None,
                                        target_key=str(d.get("target_key", "")),
                                        entity_key=str(d.get("target_key", "")),
                                        legend_entry_id=None,
                                        bbox_pdf=(
                                            float(bbox[0]),
                                            float(bbox[1]),
                                            float(bbox[2]),
                                            float(bbox[3]),
                                        ),
                                        crop_sha256=str(d.get("crop_sha256") or ""),
                                        modality=d.get("modality") or "text_tag",
                                        confidence=float(d.get("confidence") or 0.0),
                                    )
                                )
                            except ValueError:
                                continue
                        out_rel = f"{derived_relative}/overlays/page_{page_index:04d}.png"
                        out_path = path.parent / out_rel.replace("/", os.sep)
                        result = render_overlay(
                            page=overlay_page,
                            legends_on_page=legends_here,
                            detections=dets,
                            out_path=out_path,
                        )
                        if result is not None:
                            overlay_manifest.append(
                                {
                                    "page": page_index,
                                    "relative_path": out_rel,
                                    "legend_count": result.legend_count,
                                    "detection_count": result.detection_count,
                                    "width": result.width,
                                    "height": result.height,
                                }
                            )
                finally:
                    try:
                        overlay_doc.close()
                    except Exception:  # pragma: no cover
                        pass
            derived_files.append(
                {
                    "relative_path": f"{derived_relative}/schematic_overlays.json",
                    "content_kind": "json",
                    "content_json": {
                        "schema_version": "schematic.overlays.v1",
                        "overlays": overlay_manifest,
                    },
                }
            )
    return collect_all(atoms), derived_files


# Tokens that look symbol-shaped but are conventionally noise on
# construction drawings — column-grid bubbles (single letters), simple
# integer keyed-note numbers (handled separately by the keyed-notes
# pass when present), the page's own sheet number, and a small set of
# common page metadata tokens.  Boss-review fix: previously every
# repeated short ALL-CAPS token became an unknown_symbol.
_UNKNOWN_TOKEN_IGNORES = {
    "NIC",
    "NTS",
    "NA",
    "TBD",
    "REF",
    "REV",
    "SEE",
    "MAX",
    "MIN",
    "TYP",
    "EQ",
    "AFF",
    "OC",
    "DWG",
    "SHT",
    "GC",
    "EC",
    "MC",
    "PC",
    "AV",
    "FA",
    "AC",
    "SC",
    "BMS",
    "AHU",
    "VAV",
    "PDU",
    "UPS",
    "ATS",
    "MDF",
    "IDF",
    "TR",
    "ER",
    "MEP",
}


def _unknown_symbol_warnings(
    *,
    blocks: list[Any],
    page_index: int,
    sheet: str | None,
    legend: Any,
    excluded_bboxes: list[tuple[float, float, float, float]],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> list[EvidenceAtom]:
    """Emit ``unknown_symbol`` warnings for legend-style tokens with no match.

    Conservative: only short ALL-CAPS tokens (length 2-5) that appear
    repeatedly on the page.  The boss review caught that the previous
    implementation flagged ordinary drawing furniture — sheet numbers,
    grid bubbles, keyed-note integers, common drawing abbreviations —
    as unknown symbols, drowning the real warnings.  This version
    suppresses each of those classes.
    """
    import re as _re

    from app.parsers.schematic_atom_emitters import emit_warning_atom
    from app.parsers.schematic_models import SchematicWarning

    known: set[str] = {
        (e.normalized_symbol_text or "").upper()
        for e in legend.entries
        if e.normalized_symbol_text
    }
    sheet_token = (sheet or "").upper()

    def _looks_like_grid_bubble(tok: str) -> bool:
        # A single letter or single digit is a grid label, not a symbol.
        return len(tok) == 1

    def _looks_like_keyed_note_integer(tok: str) -> bool:
        # Bare 1-3 digit integers are typically keyed-note markers.
        return tok.isdigit() and 1 <= len(tok) <= 3

    def _looks_like_sheet_number(tok: str) -> bool:
        # The page's own sheet number repeats in the title block / index.
        return tok == sheet_token or _re.match(r"^[A-Z]{1,3}\d+(?:\.\d+)?$", tok) is not None

    counts: dict[str, int] = {}
    first_bbox: dict[str, tuple[float, float, float, float]] = {}
    for blk in blocks:
        if any(_bbox_intersects(blk.bbox, ex) for ex in excluded_bboxes):
            continue
        for m in _re.finditer(r"\b[A-Z0-9][A-Z0-9\-]{1,4}\b", blk.text):
            tok = m.group(0).upper()
            if tok in known:
                continue
            if tok in _UNKNOWN_TOKEN_IGNORES:
                continue
            if _looks_like_grid_bubble(tok):
                continue
            if _looks_like_keyed_note_integer(tok):
                continue
            if _looks_like_sheet_number(tok):
                continue
            counts[tok] = counts.get(tok, 0) + 1
            first_bbox.setdefault(tok, blk.bbox)
    out: list[EvidenceAtom] = []
    for tok, n in sorted(counts.items()):
        if n < 3:  # ignore noise — only flag clearly repeated tokens
            continue
        out.append(
            emit_warning_atom(
                warning=SchematicWarning.make(
                    warning_type="unknown_symbol",
                    page_index=page_index,
                    sheet_number=sheet,
                    detail=f"Token {tok!r} appears {n} times on page but is not in the resolved legend.",
                    bbox_pdf=first_bbox[tok],
                    extras={"token": tok, "count": n},
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                page=page,
            )
        )
    return out


def _bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


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
