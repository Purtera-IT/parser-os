"""Primitives shared by every PDF decoding concern.

Atom construction, table-row repair, and the predicates that decide whether a line is page furniture, a form field, or a photo request. Every module in this package needs them, which is precisely why they cannot stay owned by any one consumer -- a shared primitive living inside its largest caller is how a 10,000-line module gets built. ``orbitbrief_pdf`` re-exports them, so it keeps using its own names.
"""

from __future__ import annotations

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import ArtifactType
from app.core.schemas import AtomType
from app.core.schemas import AuthorityClass
from app.core.schemas import EvidenceAtom
from app.core.schemas import ReviewStatus
from app.core.schemas import SourceRef
from typing import Any
import re


EXTRACTION_METHOD = "orbitbrief_pdf_color_driven_v1"

TABLE_ROW_CONFIDENCE = 0.92  # tables are the most-trustworthy structure on a page

def _table_rows_repaired(page: Any, table: Any) -> list[list[Any]]:
    """``table.extract()`` with transposed glyphs repaired from the page text.

    PyMuPDF's table extractor re-reads each cell and can emit its glyphs out of
    order: the Xtra Lease install spec came back with "Initail document",
    "order of executoin" and "add lifgtate operatoin" where the page itself
    says Initial / execution / liftgate. 9 of 54 cells (17%) were corrupted,
    and nothing downstream can tell -- the words are plausible, just wrong, and
    they flow into the SOW.

    ``page.get_text(clip=cell)`` reads the same rectangle correctly, but it is
    not a blanket replacement: a clip boundary can slice a glyph off the edge
    (one cell here lost its leading "f", giving "ication:" for "fication:"), and
    a loose bbox can pull in a neighbour. So substitute ONLY when the clipped
    text is an anagram of what extract() returned -- same characters, different
    order. That is precisely the transposition bug and nothing else: a repair
    can reorder glyphs but can never add, drop, or change one.
    """
    import fitz  # type: ignore[import-not-found]

    rows = table.extract()
    try:
        cell_rows = list(getattr(table, "rows", []) or [])
    except Exception:
        return rows
    for ri, row in enumerate(cell_rows):
        if ri >= len(rows):
            break
        for ci, cell in enumerate(getattr(row, "cells", []) or []):
            if cell is None or ci >= len(rows[ri]):
                continue
            original = rows[ri][ci]
            if not original:
                continue
            a = " ".join(str(original).split())
            try:
                b = " ".join((page.get_text("text", clip=fitz.Rect(cell)) or "").split())
            except Exception:
                continue
            if not b or a == b:
                continue
            if sorted(a.replace(" ", "")) == sorted(b.replace(" ", "")):
                rows[ri][ci] = b
    return rows

_FORM_INTERROG_RE = re.compile(
    r"^(?:did|is|are|was|were|have|has|had|do|does|can|could|will|would|should|"
    r"how|what|where|when|why|which|who)\b", re.I,
)
_FORM_INSTRUCTION_RE = re.compile(
    r"^(?:upload|attach|provide|take\b|see\b|please\b|enter\b|select\b|note:|"
    r"photo of|photos of|showing\b)", re.I,
)

_FIELD_LABEL_WORDS = {
    "name", "date", "time", "title", "number", "no", "by", "email", "phone",
    "address", "id", "manager", "tech", "technician", "store", "site", "rep",
}
_FIELD_LABEL_RULE = None

def _field_label_lexical(text: str) -> bool:
    s = (text or "").strip()
    if not s or s.endswith((".", "!", "?")) or len(s.split()) > 4:
        return False
    if s.endswith(":") or s.endswith("#"):
        return True
    last = s.split()[-1].strip(":#").lower()
    return last in _FIELD_LABEL_WORDS

def _field_label_rule():
    """SemanticRule: is this a form FIELD LABEL whose value follows on the next
    line ('Managers Name' -> 'Diedra Kennedy', 'Completed By', 'Store #', 'Date'),
    so the two should merge into one 'label: value' fact — rather than a value, a
    heading, or a sentence? Label-vs-value is a meaning judgment; the suffix net is
    the offline fallback."""
    global _FIELD_LABEL_RULE
    if _FIELD_LABEL_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _FIELD_LABEL_RULE = SemanticRule(
            name="form_field_label",
            positives=[
                "Managers Name", "Site Name", "Completed By", "Date", "Store #",
                "Arrival Time", "Technician Name", "Email", "Phone Number", "Site #",
            ],
            negatives=[
                "Diedra Kennedy", "Tablet Install", "BK Audio", "POS Cabling", "Yes",
                "New Tablet", "CAT6 jacks", "The contractor shall provide all materials",
                "Upload 4 photos of the unit",
            ],
            threshold=0.52,
            lexical_fallback=_field_label_lexical,
        )
    return _FIELD_LABEL_RULE

def _is_value_field_label(text: str) -> bool:
    """A short field label that expects a TEXT value on the next line (so they
    merge). Cheap precondition, then the semantic rule decides."""
    s = (text or "").strip()
    if not s or len(s.split()) > 4 or s.endswith((".", "!", "?")):
        return False
    if _FORM_INTERROG_RE.match(s) or _FORM_INSTRUCTION_RE.match(s) or _is_photo_request(s):
        return False
    try:
        return _field_label_rule().fires(s)
    except Exception:
        return _field_label_lexical(s)

def _page_is_form_lexical(raw: list[str]) -> bool:
    """Structural net for 'is this a form / field-report page?' — the offline
    fallback for the semantic judge below. A form is: >=2 '?'  OR  >=2 form
    signals (question-start / instruction / photo-request) on a page that is
    mostly short lines (so a prose page of long sentences with an occasional
    'Provide …' is NOT a form). Detects forms whose questions are phrased as
    statements ('Have you train the MOD … units.', zero '?')."""
    nonblank = [l.strip() for l in raw if l.strip()]
    if len(nonblank) < 3:
        return False
    if sum(1 for l in nonblank if l.endswith("?")) >= 2:
        return True
    signals = sum(
        1 for l in nonblank
        if l.endswith("?") or _FORM_INTERROG_RE.match(l)
        or _FORM_INSTRUCTION_RE.match(l) or _is_photo_request(l)
    )
    short = sum(1 for l in nonblank if len(l.split()) <= 12)
    return signals >= 2 and short >= 0.75 * len(nonblank)

_FORM_PAGE_RULE = None

def _form_page_rule():
    """SemanticRule: does this page READ like a form / field-report (questions +
    labelled fields + photo requests + signatures) rather than flowing prose, a
    contract clause, or a rate-card/BOM table? Form-vs-prose is a MEANING judgment
    — and real pages are hybrids (a field report mixes Q&A, a photo, a signature,
    sometimes a paragraph) — so the embedder leads and the structural heuristic is
    the offline net. Worth-regrouping is what we're really asking."""
    global _FORM_PAGE_RULE
    if _FORM_PAGE_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _FORM_PAGE_RULE = SemanticRule(
            name="form_field_report_page",
            positives=[
                "Have you installed the NEXEO Box? Yes  Upload 4 Photos of the unit installed.",
                "Did you pull 2 cables to each POS? Yes  How many total Cables were pulled? 8",
                "Is this store a 2 LANE Store for Drive Thru? No  Tablet Install  Did you install a new Tablet?",
                "Name:  Store #:  Site #:  Arrival Time: 06:00 AM  Departure Time: 09:30 AM",
                "Upload Photo of Tablet installed.  Signature  Managers Name  Diedra Kennedy",
                "Was there a pre-existing BK Audio Box? No  POS Cabling  Upload Photo showing the label",
            ],
            negatives=[
                "The Contractor shall furnish all materials, tools, and labor necessary to complete the installation in accordance with the project specifications and applicable codes.",
                "This Project Services Statement of Work is made by and between Norvet MSP and PurTera LLC and shall become effective on the date of last signature.",
                "Payment terms are net thirty (30) days from the date of invoice; late payments accrue interest at 1.5% per month.",
                "Country: Albania | Networking L1 Technician 2 hr min: 83.3 | Networking L2 Technician: 110.0",
                "ID #: 11 | Material Description: 24-Port CAT6 Patch Panel | Quantity: 5 | Unit Cost: $120",
                "2.2.10 Cable Pathways. The Contractor shall install cable support hardware such as cable trays, J-hooks, and conduit as required.",
            ],
            threshold=0.50,
            # offline / embedder-down: fall back to the structural heuristic
            # (re-split the joined representation back into lines).
            lexical_fallback=lambda s: _page_is_form_lexical(
                [x for x in s.split("  ") if x.strip()]
            ),
        )
    return _FORM_PAGE_RULE

def _page_is_form(raw: list[str]) -> bool:
    """Is this page a form / field-report worth Q&A-regrouping + column-skip +
    short-field retention — vs flowing prose / a contract / a rate-card table?
    Embedding-led (handles hybrid pages by MEANING), structural heuristic offline."""
    nonblank = [l.strip() for l in raw if l.strip()]
    if len(nonblank) < 3:
        return False
    # Diarized meeting transcripts are never forms — speaker stamps + rhetorical
    # questions would otherwise trip form mode and shatter the dialogue.
    try:
        from app.core.hybrid_summary_transcript import count_speaker_timestamp_hits

        if count_speaker_timestamp_hits("\n".join(nonblank)) >= 2:
            return False
    except Exception:
        pass
    rep = "  ".join(nonblank[:40])[:800]
    try:
        return _form_page_rule().fires(rep)
    except Exception:
        return _page_is_form_lexical(raw)

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

# A "Label: Value" pair whose value is real text (≥3 letters), not blanks/
# underscores — e.g. "Business Sponsor: Jordan Ames". Used to tell a filled
# roster apart from an empty form template.
_FILLED_FIELD_RE = re.compile(r"[A-Za-z][A-Za-z ]{2,40}:\s*[A-Za-z][A-Za-z.\-]{2,}")

# A signature / sign-off line: a role label, blanks to sign on, and a Date
# field — "OPTBOT - Director, Workplace Technology: ____  Date: ____". Even
# unsigned, the ROLE is governance content (who approves), so it's kept.
_SIGNOFF_RE = re.compile(
    r"[A-Za-z][A-Za-z ,/&'\-]{4,60}:\s*_{2,}.*\bDate\b\s*:", re.IGNORECASE
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
    # A filled roster line (signature / approval block: "Role: Name |
    # Signature: ___ | Date: ___") carries real content — the role→name pair —
    # even though its Signature/Date fields are blank. One filled label:value
    # pair is enough: a genuinely blank vendor template has none, so this never
    # rescues those. (Threshold is 1, not 2, so a single split-out signature
    # line still survives.)
    if len(_FILLED_FIELD_RE.findall(text)) >= 1:
        return False
    # A signature / sign-off line ("OPTBOT - Director, Workplace Technology:
    # ____  Date: ____") carries the approver role even unsigned — keep it. A
    # blank vendor data-collection template has no role+Date sign-off line.
    if _SIGNOFF_RE.search(text):
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
    # placeholders unresolved when the doc is generated quickly), plus the
    # slash form "Page 1/1" common in tool-generated footers.
    r"\bpage\s+(?:(?:\d+|[xn])\s+of\s+(?:\d+|[xny])|\d+\s*/\s*\d+)\b",
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

_BARE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/?", re.I)

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
    # A standalone URL / bare domain ('www.purtera-it.com', 'WWW.X.COM',
    # 'https://x.com') is a footer / letterhead band, never deal content — strip
    # it so it can't glue onto the start of a real clause.
    if _BARE_URL_RE.fullmatch(text.strip()):
        return True
    if len(text) > 240:
        return False  # Real footers are short; long blocks are scope.
    # A signature / sign-off line ("Role: ____  Date: ____") is governance
    # content, not page furniture — its blanks + "Date" must not read as a
    # footer band.
    if _SIGNOFF_RE.search(text):
        return False
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

_PHOTO_REQUEST_RULE = None
_PHOTO_REQUEST_RE = re.compile(
    r"\b(?:upload|attach|take|provide|include)\b.*\bphotos?\b"
    r"|\bphotos?\b.*\b(?:showing|of|install)", re.I,
)

def _photo_request_lexical(text: str) -> bool:
    return bool(_PHOTO_REQUEST_RE.search(text or ""))

def _photo_request_rule():
    """SemanticRule: is this line a PHOTO-REQUEST instruction ('Upload 4 Photos of
    the Nexeo', 'Upload a photo showing the rack', 'Take a photo of the install')?
    Used to caption each extracted image with what it SHOULD show — the form
    instruction the photo answers — so a reviewer / the vision pass sees expected
    content, not a bare 'awaiting OCR'. Embedding generalises past the keyword net;
    regex is the offline fallback."""
    global _PHOTO_REQUEST_RULE
    if _PHOTO_REQUEST_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _PHOTO_REQUEST_RULE = SemanticRule(
            name="photo_request_instruction",
            positives=[
                "Upload 4 Photos of the Nexeo installed at the site.",
                "Upload a photo showing all of the cables terminated and labeled.",
                "Take a photo of the rack showing the equipment mounted.",
                "Upload photos showing the IB7000 installed in its location.",
                "Upload a photo of the drive thru director showing it is working.",
                "Attach a picture of the completed install.",
            ],
            negatives=[
                "PurTera will install low voltage cabling.",
                "The vendor shall provide standardized reports upon completion.",
                "Total Base Hrs: 1148.81", "Have you installed the NEXEO Box?",
                "BK Store Number: 557", "Network design and configuration.",
            ],
            threshold=0.55,
            lexical_fallback=_photo_request_lexical,
        )
    return _PHOTO_REQUEST_RULE

def _is_photo_request(text: str) -> bool:
    s = (text or "").strip()
    if not s or "photo" not in s.lower():
        return False
    try:
        return _photo_request_rule().fires(s)
    except Exception:
        return _photo_request_lexical(s)

_IMAGE_FIELD_LABEL_RULE = None

def _image_field_label_rule():
    """SemanticRule: is this a short label whose value is the IMAGE directly below
    it ('Signature', 'Floor Plan', 'Rack Elevation', 'Wiring Diagram', 'Before /
    After') — so the image is captioned by it rather than a far-off photo request
    higher on the page? A meaning judgment (generalises past Title-Case to 'floor
    plan', 'cable test results'); the structural check below is the offline net."""
    global _IMAGE_FIELD_LABEL_RULE
    if _IMAGE_FIELD_LABEL_RULE is None:
        from app.core.semantic_rules import SemanticRule
        _IMAGE_FIELD_LABEL_RULE = SemanticRule(
            name="image_field_label",
            positives=[
                "Signature", "Floor Plan", "Rack Elevation", "Site Diagram",
                "Wiring Diagram", "Equipment Photo", "Network Closet", "Before",
                "After", "Cable Test Results", "Site Map", "Rack Layout",
            ],
            negatives=[
                "Diedra Kennedy", "Yes", "No", "Did you install a new tablet?",
                "The contractor shall furnish all materials and labor.",
                "2.2.10 Cable Pathways", "Country: Albania", "New Tablet",
            ],
            threshold=0.50,
            lexical_fallback=lambda s: _is_image_field_label(s, _lexical_only=True),
        )
    return _IMAGE_FIELD_LABEL_RULE

def _is_image_field_label(text: str, _lexical_only: bool = False) -> bool:
    """A short labelled field whose value is the image directly below it on a form
    ('Signature' -> the signature image). Used so such an image is captioned by
    the field right above it, not by a far-off photo request higher on the page.
    Conservative: a short Title-Case label / colon-label that is not a question,
    instruction, sentence, or bare number. Embedding-led; the lexical body below
    is the offline net (and runs directly when _lexical_only)."""
    s = (text or "").strip()
    if not _lexical_only:
        try:
            return _image_field_label_rule().fires(s)
        except Exception:
            pass
    if not s or s.endswith((".", "!", "?")):
        return False
    words = s.split()
    if not (1 <= len(words) <= 4) or not any(c.isalpha() for c in s):
        return False
    if _FORM_INTERROG_RE.match(s) or _FORM_INSTRUCTION_RE.match(s) or _is_photo_request(s):
        return False
    if s.endswith(":"):
        return True
    if s.lower() in {"yes", "no", "n/a", "na", "tbd", "none"}:
        return False
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    return bool(alpha) and sum(1 for w in alpha if w[0].isupper()) / len(alpha) >= 0.8

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
    review_flags: list[str] | None = None,
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
        review_flags=list(review_flags or []),
        parser_version=parser_version,
    )

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
