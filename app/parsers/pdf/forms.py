"""Form pages: checkboxes, header fields, and question/answer regrouping.

A filled form is not prose. Its meaning lives in which box carries an X and which label sits beside which value, so it needs its own reading rather than the paragraph path.
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
from app.parsers.pdf._shared import _FORM_INSTRUCTION_RE
from app.parsers.pdf._shared import _FORM_INTERROG_RE
from app.parsers.pdf._shared import _is_photo_request
from app.parsers.pdf._shared import _is_value_field_label
from app.parsers.pdf._shared import _page_is_form
import re


# Connector words: if a question line ENDS on one, the next line completes the
# question ("...were pulled to" + "POS"), so it's a continuation, not the answer.
_CONNECTOR_WORDS = {
    "to", "of", "for", "in", "on", "at", "the", "a", "an", "and", "or", "with",
    "each", "by", "from", "into", "per", "your", "their", "any", "all",
}

def _is_discrete_answer(line: str) -> bool:
    """A DISCRETE form answer — a self-contained value: 'Yes', 'No', '8', '$300',
    'New Tablet'. Distinct from a question's wrapped continuation ('Talking POS in
    the store'), which is a sentence fragment carrying lowercase function words.
    Used to find where a multi-line question (one with no '?') ends and its answer
    begins, so the answer doesn't get mistaken for question text (and dropped)."""
    s = (line or "").strip()
    if not s or s.endswith("?") or _FORM_INTERROG_RE.match(s) or _FORM_INSTRUCTION_RE.match(s):
        return False
    words = s.split()
    if len(words) > 4:
        return False
    if s.lower() in {"yes", "no", "n/a", "na", "tbd", "none", "true", "false"}:
        return True
    if re.fullmatch(r"[-+]?[\d.,$%]+\w{0,6}", s):  # 8, 950, $300, 12.5, 950ft
        return True
    # a short noun-phrase value with NO lowercase function words is an answer
    # ('New Tablet'); one WITH them ('Talking POS in the store') is continuation.
    if any(w.lower() in _CONNECTOR_WORDS for w in words):
        return False
    return len(words) <= 3

def _regroup_form_qa(text: str) -> str:
    """On a questionnaire / field-report page, join each question with its answer
    so 'Have you installed the NEXEO Box?\\nYes' becomes ONE 'Q?  A' unit instead
    of a blob that glues the question, its answer, and the next instruction. Gated
    to pages carrying >=2 question lines, so ordinary prose/scope pages pass
    through untouched. A multi-line question (\"Is this store a 2 LANE Store for
    Drive\" + \"Thru?\") is reassembled; instruction lines ('Upload 4 Photos…') stay
    on their own so they become their own atoms."""
    raw = [ln.strip() for ln in (text or "").splitlines()]
    if not _page_is_form(raw):
        return text
    units: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        ln = raw[i]
        if not ln:
            i += 1
            continue
        if _is_photo_request(ln):
            # A photo request wraps across lines ("Upload Photo of Tablet
            # installed and" / "showing the correct screen loaded to" / "show
            # activity."). Gather the whole thing as ONE unit. The generic
            # instruction branch below breaks on a continuation that happens to
            # start with an instruction word ('showing' is in _FORM_INSTRUCTION_RE)
            # — splitting the request — so photo requests get their own gather
            # that joins any non-question wrapped tail and stops at sentence end
            # (so it never swallows the following section header).
            parts = [ln]
            i += 1
            while i < n and raw[i]:
                nxt = raw[i]
                # a new question / photo request / footer page-number ends it
                if (nxt.endswith("?") or _FORM_INTERROG_RE.match(nxt)
                        or _is_photo_request(nxt) or nxt.isdigit()):
                    break
                # wrapped tail: lowercase start, a 3+ word sentence line, OR a
                # short Title-Case fragment that completes the phrase ('Headset
                # Holder' + 'Mounting', 'Battery' + 'Charger Mounting') — same rule
                # the text splitter uses, so regroup doesn't leave fragments it can't.
                if (nxt[0].islower() or len(nxt.split()) >= 3
                        or re.match(r"^[A-Z][\w/&-]*( [A-Z][\w/&-]*){0,2}$", nxt)):
                    parts.append(nxt)
                    i += 1
                    if nxt.rstrip().endswith((".", "!", ":")):
                        break  # sentence complete — don't absorb the next header
                else:
                    break
            units.append(" ".join(parts).strip())
        elif _FORM_INTERROG_RE.match(ln) or ln.endswith("?"):
            # Assemble a (possibly multi-line) question, then its answer. The
            # question may wrap WITHOUT a '?' ("Did you pull 2 cables to each Cash"
            # + "Talking POS in the store" + "Yes"): keep joining continuation
            # lines until a DISCRETE answer ('Yes'/'8'/'New Tablet'). A line that
            # looks discrete is still a continuation when the question dangles on a
            # connector word ("...pulled to" + "POS"). Without this the question's
            # tail was taken as the answer and the real answer was orphaned + lost.
            parts = [ln]
            i += 1
            joined = 0
            while (not parts[-1].endswith("?") and i < n and raw[i] and joined < 4
                   and not _FORM_INTERROG_RE.match(raw[i])
                   and not _FORM_INSTRUCTION_RE.match(raw[i])):
                last_word = parts[-1].rstrip().split()[-1].lower() if parts[-1].split() else ""
                dangling = last_word in _CONNECTOR_WORDS
                if _is_discrete_answer(raw[i]) and not dangling:
                    break
                parts.append(raw[i])
                i += 1
                joined += 1
            question = " ".join(parts).strip()
            answer = ""
            if i < n and _is_discrete_answer(raw[i]):
                answer = raw[i]
                i += 1
            units.append(f"{question}  {answer}".strip())
        elif _FORM_INSTRUCTION_RE.match(ln):
            # an instruction ("Upload 4 Photos…") wraps across lines — join the
            # sentence-continuation lines (until the next question / instruction /
            # short header) so it is one atom, not three fragments.
            parts = [ln]
            i += 1
            while (i < n and raw[i]
                   and not raw[i].endswith("?") and not _FORM_INTERROG_RE.match(raw[i])
                   and not _FORM_INSTRUCTION_RE.match(raw[i])
                   and (raw[i][0].islower() or len(raw[i].split()) >= 4)):
                parts.append(raw[i])
                i += 1
            units.append(" ".join(parts).strip())
        elif (_is_value_field_label(ln) and i + 1 < n and raw[i + 1]
              and not _is_value_field_label(raw[i + 1])
              and not raw[i + 1].endswith("?") and not _FORM_INTERROG_RE.match(raw[i + 1])
              and not _FORM_INSTRUCTION_RE.match(raw[i + 1]) and not _is_photo_request(raw[i + 1])):
            # A field label whose value is on the next line ('Managers Name' +
            # 'Diedra Kennedy' -> 'Managers Name: Diedra Kennedy') — merge into one
            # label:value fact. Only when the label expects a TEXT value (not
            # 'Signature', whose value is the image below it) and the next line is
            # a value, not another label / question / instruction.
            units.append(f"{ln.rstrip(':')}: {raw[i + 1]}")
            i += 2
        else:
            # standalone line — a section header ("HME NEXO Box Install", "BK
            # Audio", "POS Cabling") or stray value; keep as its own unit.
            units.append(ln)
            i += 1
    # blank-line-separate every unit so the prose splitter emits each Q&A,
    # instruction and header as its OWN atom instead of gluing them together.
    return "\n\n".join(units)

# Heuristic: a "checkbox cluster" line has ≥2 capitalized labels and
# ≥1 literal x prefix.
_CHECKBOX_LITERAL_LINE_RE = re.compile(r"\bx\s+[A-Z]", re.UNICODE)

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
