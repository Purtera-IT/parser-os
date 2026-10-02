"""HubSpot CRM note parser — short timeline notes, not transcripts.

Exported ``*-hs-note-*.txt`` files carry deal facts in a few lines (ROM,
hardware counts, config-only scope) but lack speaker/timestamp structure.
Routing them through :class:`TranscriptParser` yields ``ok_empty`` because
utterance regexes miss ``aps``/``configuration`` and ISO dates masquerade as
timestamps.

This parser reads the note body directly (no utterance segmentation) and emits
typed atoms with ``artifact_id`` provenance for the Files UI.
"""

from __future__ import annotations

from app.core.textio import read_text

import html as _html
import re
from pathlib import Path
from typing import Any

from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.greetings import starts_with_greeting
from app.core.ids import stable_id
from app.core.internal_author import (
    apply_internal_author_elevation,
    classify_author_affiliation,
)
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
from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser
from app.parsers.structured_projection import (
    derived_files_for,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    stamp_section_and_block_ids,
)

STRUCTURED_SCHEMA_HUBSPOT_NOTE = "orbitbrief.hubspot_note.structured.v1"
HUBSPOT_NOTE_RELATION = "hubspot_note_extraction"

_HS_NOTE_FILENAME_RE = re.compile(r"-hs-note-", re.I)
_HS_NOTE_HEADER_RE = re.compile(r"^HubSpot Note\s*:", re.I | re.M)
_HS_NOTE_ID_RE = re.compile(r"^HubSpot Note ID:\s*(\S+)", re.I | re.M)
_HS_DATE_RE = re.compile(r"^Date:\s*(.+)$", re.I | re.M)
_HS_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.I | re.M)
_HS_AUTHOR_EMAIL_RE = re.compile(r"^Author-Email:\s*(.+)$", re.I | re.M)

_ROM_RE = re.compile(
    r"\b(?:ROM|rough\s+order\s+of\s+magnitude)\b|"
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b|"
    r"\b(\d{1,3}(?:,\d{3})*)\s*(?:k|K)\b",
    re.I,
)
_SCOPE_SIGNAL_RE = re.compile(
    r"\b(install|configur|ubiquiti|unifi|udm|nvr|unvr|camera|badge|okta|otka|"
    r"vlan|meraki|remote|onsite|white\s+glove|equipment|switch|router|ap\b|aps\b|"
    r"reader|enterprise|rom|good\s+2\s+go)\b",
    re.I,
)
#: Titles the CRM (or a hurried PM) gives a note that says nothing: on its own
#: such a note is empty, not content.
_PLACEHOLDER_NOTE_TITLES = frozenset({
    "note", "notes", "new note", "call", "call notes", "meeting", "meeting notes",
    "email", "update", "fyi", "test", "untitled", "n/a", "na", "-",
})


def _automated_sender_lines_as_chatter(atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    """A pasted e-sign notice's chrome is the robot's, never a person's.

    "From: Adobe Sign <echosign@echosign.com>" and "please add
    echosign@echosign.com to your address book" were a customer_instruction
    and a scope line (010003). Each such line stays an atom, typed
    ``deal_metadata`` and flagged admission chatter, so a labeler can reject
    it and no head reads it; one line read as several types keeps one atom.
    """
    from app.core.admission_chatter import mark_admission_chatter
    from app.parsers.email_parser import _names_automated_sender

    out: list[EvidenceAtom] = []
    seen: set[str] = set()
    for atom in atoms:
        text = str(getattr(atom, "raw_text", "") or "")
        at = getattr(atom.atom_type, "value", atom.atom_type)
        if at in ("stakeholder", "physical_site") or not _names_automated_sender(text):
            out.append(atom)
            continue
        if text in seen:
            continue
        seen.add(text)
        atom.atom_type = AtomType.deal_metadata
        mark_admission_chatter(atom, "automated_sender")
        out.append(atom)
    return out


#: Author names a CRM gives a note no person wrote: the HubSpot system user,
#: an integration or API user, a workflow. Whole-name match only.
_SYSTEM_AUTHOR_RE = re.compile(
    r"^(?:hubspot(?:\s+(?:system|integration|automation|api|workflows?|bot|app)(?:\s+user)?)?"
    r"|(?:system|integration|automation|api|workflow|bot|service)(?:\s+(?:user|account))?"
    r"|workflows?|zapier|make\.com)$",
    re.I,
)

#: What the note export writes when HubSpot resolved NO owner for a note
#: (platform-infra `mapHubspotNoteProperties`). Every API-created note carries
#: it -- the OxBlue intake notification among them -- but so does a person's
#: note whose owner did not resolve (000061's meeting recap). So on its own it
#: decides nothing; with a notification-shaped body it is the system user.
_UNRESOLVED_AUTHORS = frozenset({"hubspot user"})

_FIELD_LINE_RE = re.compile(r"^[A-Za-z][\w /&().-]{0,40}:\s*\S")

#: A field value that is an identifier or a date: one token carrying a digit
#: ("OXR-1042", a UUID, "2026-08-14"). The fact an automated note may be the
#: only place to state; its prose is a restatement.
_ID_OR_DATE_VALUE_RE = re.compile(
    r"^(?=[^\s]*\d)[A-Za-z0-9][A-Za-z0-9#/_.:+-]*$"
)


def _is_automated_note_author(author: str, author_email: str) -> bool:
    """Is the author unmistakably a machine -- the CRM's system user, an
    integration, a no-reply address -- rather than a person on the deal?"""
    from app.core.automated_senders import is_automated_address, is_automated_sender

    name = " ".join(str(author or "").split()).strip(" .")
    if author_email and is_automated_address(author_email):
        return True
    if not name or name.lower() in _UNRESOLVED_AUTHORS:
        return False
    return bool(_SYSTEM_AUTHOR_RE.match(name)) or is_automated_sender("", name=name)


#: The announcing line of a system notification names an event that happened
#: to a record ("intake received", "form submitted", "deal created"). A person
#: titling a form ("Site survey details", "Walkthrough") names a topic.
_NOTIFICATION_EVENT_RE = re.compile(
    r"\b(?:received|submitted|created|generated|imported|synced|logged|"
    r"auto[- ]?(?:created|generated|logged)|new\s+(?:submission|intake|form|lead|request))\b",
    re.I,
)

#: A person writing to colleagues says "I", "we", "you"; a system does not.
_PERSONAL_PRONOUN_RE = re.compile(r"\b(?:I|I'm|I've|we|we're|our|us|my|me|you|your)\b")


def _is_notification_shaped(parsed: dict[str, Any]) -> bool:
    """One short announcing line naming a system event, then nothing but
    ``Label: value`` fields -- the shape a system writes, never a person's
    recap. A person's own form under an unresolved owner ("Site survey
    scheduled" / "Address: ..." / "Contact: ...") has the field shape too, so
    the announcing line must read as an event and nobody may speak in it."""
    lines = [str(ln).strip() for ln in (parsed.get("body_lines") or []) if str(ln).strip()]
    if len(lines) < 3 or len(lines[0].split()) > 12 or _FIELD_LINE_RE.match(lines[0]):
        return False
    if not _NOTIFICATION_EVENT_RE.search(lines[0]):
        return False
    if any(_PERSONAL_PRONOUN_RE.search(ln) for ln in lines):
        return False
    return all(_FIELD_LINE_RE.match(ln) for ln in lines[1:])


def _note_is_automated(parsed: dict[str, Any]) -> bool:
    author = str(parsed.get("author") or "")
    email = str(parsed.get("author_email") or "")
    if _is_automated_note_author(author, email):
        return True
    unresolved = " ".join(author.split()).lower() in _UNRESOLVED_AUTHORS and not email
    return unresolved and _is_notification_shaped(parsed)


def _place_unlined_atoms(atoms: list[EvidenceAtom], raw_lines: list[str], body_at: int) -> None:
    """Give every atom of a note the line it was read from.

    Only body prose was located; the header atom, field values, signature
    people and address sites carried the bare note locator, so they tied at
    line 0 and read in whatever order the passes emitted them -- live 010087,
    a pasted email's signature above the note's own header. The header sits
    at line 0, above the body; any other atom on the first body line holding
    its whole text. Never moves an atom
    that already has a line.
    """
    normed = [" ".join(str(ln).split()).lower() for ln in raw_lines]
    for atom in atoms:
        refs = list(getattr(atom, "source_refs", None) or [])
        if not refs:
            continue
        loc = dict(getattr(refs[0], "locator", None) or {})
        if loc.get("line_start") is not None:
            continue
        v = atom.value if isinstance(atom.value, dict) else {}
        line = None
        if v.get("kind") == "hubspot_note_meta":
            # Composed text: line 0 places it above the body without claiming
            # a line it would fail to replay against (``source_lines`` holds
            # the real header lines).
            line = 0
        else:
            # Only where the line holds the atom's whole text, so the source
            # replay still verifies it; composed atoms keep no line.
            probe = " ".join(str(atom.raw_text or "").split()).lower()
            if probe:
                line = next((i + 1 for i in range(body_at, len(normed)) if probe in normed[i]), None)
        if line is None:
            continue
        loc.update({"line_start": line, "line_end": line})
        refs[0] = refs[0].model_copy(update={"locator": loc})
        atom.source_refs = refs


_PASTED_FROM_RE = re.compile(r"^\s*From\s*:\s*(?P<v>\S.*)$", re.I)
_PASTED_HDR_RE = re.compile(r"^\s*(?:Sent|Date|To|Cc|Subject)\s*:", re.I)
_SIGNOFF_LINE_RE = re.compile(
    r"^\s*(?:thanks?|thank\s+you|thx|many\s+thanks|(?:kind(?:est)?|best|warm(?:est)?)\s+regards?|"
    r"regards?|cheers|sincerely|best|all\s+the\s+best|respectfully)\s*[,.!-]*\s*$",
    re.I,
)
_BARE_NAME_RE = re.compile(r"^[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2}$")


def _pasted_email_sender(parsed: dict[str, Any]) -> tuple[dict[str, str], int] | None:
    """``({name, email}, first body line of the paste)`` when the note body is
    someone else's email pasted in, else None.

    Live 010087: Trent's note opened "Hi Trent," -- Stephanie's email to him,
    pasted -- and every line of it was credited to Trent. Two shapes:

    * a pasted header block: a ``From:`` line followed by ``Sent:`` / ``To:`` /
      ``Subject:``; the sender is the ``From:`` value and the paste runs from
      that line;
    * a body that opens by greeting the note's own author by name and closes
      with a signature (or a sign-off and a name) naming someone else; the
      whole body is the paste.
    """
    from email.utils import parseaddr

    lines = [str(ln).strip() for ln in (parsed.get("body_lines") or []) if str(ln).strip()]
    if not lines:
        return None
    author = _name_key(str(parsed.get("author") or ""))
    for i, ln in enumerate(lines):
        m = _PASTED_FROM_RE.match(ln)
        if not m or not any(_PASTED_HDR_RE.match(x) for x in lines[i + 1:i + 5]):
            continue
        nm, addr = parseaddr(m.group("v"))
        nm = " ".join((nm or "").split()) or (m.group("v").split("<")[0].strip(' "') if not addr else "")
        if not nm and not addr:
            continue
        if (author and _name_key(nm) == author) or (
            addr and addr.lower() == str(parsed.get("author_email") or "").lower()
        ):
            continue
        return {"name": nm, "email": addr if "@" in addr else ""}, i
    first_name = author.split()[0] if author and author not in _UNRESOLVED_AUTHORS else ""
    from app.core.greetings import LEADING_GREETING_RE

    g = LEADING_GREETING_RE.match(lines[0])
    if not first_name or not g or first_name not in _name_key(lines[0][: g.end()]).split():
        return None
    sender: dict[str, str] = {}
    try:
        from app.parsers.signature_block import people_from_signature_lines

        for rec in people_from_signature_lines(lines[1:]):
            if _name_key(rec.get("name", "")) not in (author, first_name):
                sender = {"name": str(rec.get("name") or ""), "email": str(rec.get("email") or "")}
    except Exception:  # pragma: no cover - signature reading is best effort
        sender = {}
    if not sender:
        for k in range(len(lines) - 2, 0, -1):
            if _SIGNOFF_LINE_RE.match(lines[k]):
                cand = re.sub(r"^[-~\s]+", "", lines[k + 1]).strip(" .,")
                if _BARE_NAME_RE.match(cand) and _name_key(cand) not in (author, first_name):
                    sender = {"name": cand, "email": ""}
                break
    return (sender, 0) if sender.get("name") or sender.get("email") else None


def _credit_pasted_email(atoms: list[EvidenceAtom], parsed: dict[str, Any]) -> None:
    """Credit the atoms of a pasted email to its sender, not the note author.

    The note author stays on each atom as ``note_author``: they posted it, the
    sender said it. An internal author's trust boost does not carry over to
    an outside sender's words."""
    found = _pasted_email_sender(parsed)
    if not found:
        return
    sender, first = found
    raw_lines = [str(ln) for ln in (parsed.get("raw_lines") or [])]
    body_at = int(parsed.get("body_line_index") or 0)
    body = [str(ln).strip() for ln in (parsed.get("body_lines") or []) if str(ln).strip()]
    start_line = 0  # 1-based line of the paste in the file; 0 = whole body
    if first > 0:
        target = body[first]
        start_line = next((i + 1 for i in range(body_at, len(raw_lines))
                           if raw_lines[i].strip() == target), 0) or 10 ** 9
    affiliation = classify_author_affiliation(sender.get("name"), author_email=sender.get("email") or None)
    for atom in atoms:
        val = atom.value if isinstance(atom.value, dict) else None
        if val is None or val.get("kind") in ("hubspot_note_meta", "automated_note"):
            continue
        if start_line:
            refs = list(getattr(atom, "source_refs", None) or [])
            loc = dict(getattr(refs[0], "locator", None) or {}) if refs else {}
            ln = loc.get("line_start")
            if ln is None or int(ln) < start_line:
                continue
        val = dict(val)
        val.setdefault("note_author", str(parsed.get("author") or ""))
        val.setdefault("note_author_email", str(parsed.get("author_email") or ""))
        val["author"] = sender.get("name") or sender.get("email") or ""
        val["author_email"] = sender.get("email") or ""
        val["author_affiliation"] = affiliation
        val["pasted_email"] = True
        atom.value = val
        if affiliation != "internal" and "internal_author" in (atom.review_flags or []):
            atom.review_flags = [f for f in atom.review_flags
                                 if f not in ("internal_author", "trusted_internal_source")]
            atom.confidence = min(atom.confidence, 0.84)


def _is_placeholder_note_title(text: str) -> bool:
    return " ".join(str(text or "").lower().split()).strip(" .:!-") in _PLACEHOLDER_NOTE_TITLES


#: A finite verb makes a clause, and a clause is a statement: "parking is not
#: free" says something, "psow from current partner" names something.
_FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|am|be|been|has|have|had|do|does|did|will|would|can|could|"
    r"should|shall|may|might|must|isn't|aren't|wasn't|don't|doesn't|didn't|won't|can't|"
    r"believe|think|want|wants|know|knows)\b",
    re.I,
)


def _is_upload_caption(text: str) -> bool:
    """A label for an attached file ("SOW", "psow from current partner"), not
    a statement: short, no figures, no sentence punctuation, no ask, no verb,
    nobody greeted. "PO!!", "Need Troy and Wilmington sites removed." and
    "Hello, parking is not free" are statements."""
    t = " ".join(str(text or "").split())
    return bool(
        t
        and not starts_with_greeting(t)
        and len(t.split()) <= 8
        and not re.search(r"\d|\$", t)
        and not re.search(r"[.!?]", t)
        and not _INSTRUCTION_RE.search(t)
        and not _FINITE_VERB_RE.search(t)
        and not re.search(r"\b(?:need|needs|remove|removed|add|added|cancel|confirm|send)\b", t, re.I)
    )


#: A line that revises what the author said just before it, in the same note:
#: "Update on that they do have wall mounts and parking" (010003).
_UPDATE_CUE_RE = re.compile(
    # "Update the firmware on all APs" is an instruction, not a revision: a
    # bare "update" counts only when punctuated as a lead-in.
    r"^\s*(?:update\s+on\s+(?:that|this)\b|update\s*[:,\-]|correction\s*[:,\-]|"
    r"actually\s*,|scratch\s+that\b)",
    re.I,
)


_INSTRUCTION_RE = re.compile(
    r"\b(please|need to|must|should|customer would like|get full list|"
    r"good\s+2\s+go|approved|hold off|go ahead)\b",
    re.I,
)


#: A field label inside running text: one to four capitalised words (a
#: "|"-joined pair like "Date | Time" or a slash pair like "Tech/Engineer"
#: counts as one), followed by a colon and a space. The shape, not the words.
_INLINE_LABEL_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s.;]))(?P<label>[A-Z][A-Za-z/&]*(?:\s*\|\s*[A-Z][A-Za-z/&]*)?(?:\s+[A-Za-z][A-Za-z/&]*){0,2})\s*:\s+(?=\S)"
)
_TIME_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|minutes?|mins?)\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(?:hour|day|week|month)s?\b", re.I)
_COUNT_ROLE_RE = re.compile(r"^\s*\d+\s+[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s*$")


#: A label that owns its own line: "Address:", "Onsite work at 1517:",
#: "Here's the scope:", or "Project: <value>". Judged by shape -- a short run of
#: words starting with a capital, no sentence punctuation, then a colon -- so a
#: store number or an apostrophe inside the label does not hide it.
_LINE_LABEL_RE = re.compile(r"^(?P<label>[A-Z][^:.?!]{0,58}?)\s*:\s*(?P<rest>.*)$")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


def _line_label(line: str) -> tuple[str, str] | None:
    m = _LINE_LABEL_RE.match(line.strip())
    if not m or re.match(r"\s*//", m.group("rest")):
        return None
    label = " ".join(m.group("label").split())
    if len(label.split()) > 6:
        return None
    return label, m.group("rest").strip()


def _split_line_fields(body: str) -> list[tuple[str, str]]:
    """Fields of a note typed one label per line, values on the lines below.

    000020 Binghamton is the shape: "Address:" then two address lines, then
    "Onsite work at 1517:" then eleven one-line tasks. The inline splitter joined
    every line into one string first, so the label with a store number in it was
    never seen, and each task list was cut at its commas ("label it", "store as
    a backup Reset Ubiquiti gateway"). When the author gave us lines, the line
    is the item. Values keep their line breaks; ``_split_list_value`` splits on
    them. Needs two or more label lines, otherwise this is not a form."""
    lines = [ln.strip() for ln in str(body or "").splitlines()]
    marks = [(i, _line_label(ln)) for i, ln in enumerate(lines)]
    marks = [(i, lab) for i, lab in marks if lab]
    if len(marks) < 2:
        return []
    fields, _free = _split_line_fields_and_free_lines(body)
    return fields


def _split_line_fields_and_free_lines(body: str) -> tuple[list[tuple[str, str]], list[str]]:
    """``(fields, free lines)`` for a one-label-per-line note.

    A label whose own line already carries a complete statement ("Next Steps:
    Client to provide floor plans; survey will finalize scope ...") is closed by
    it; the lines under it are not its items. 000061 MBrany's meeting recap is
    the shape: one "Next Steps:" line followed by the whole recap -- dozens of
    sentences and section headings -- which all became items of "Next Steps".
    A short inline value ("Adress: CHECKOUT SAN FRANCISCO") still continues on
    the lines below, which is how an address is typed. Lines that belong to no
    field come back as free lines for the ordinary prose path."""
    lines = [ln.strip() for ln in str(body or "").splitlines()]
    marks = [(i, _line_label(ln)) for i, ln in enumerate(lines)]
    marks = [(i, lab) for i, lab in marks if lab]
    if len(marks) < 2:
        return [], []
    out: list[tuple[str, str]] = []
    free: list[str] = [ln for ln in lines[: marks[0][0]] if ln]
    for k, (i, (label, rest)) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        below = [ln for ln in lines[i + 1:end] if ln]
        closed = bool(rest) and (len(rest.split()) >= 6 or bool(_SENTENCE_END_RE.search(rest)) or rest.endswith("\u2026"))
        if closed:
            out.append((label, rest))
            free.extend(below)
        else:
            out.append((label, "\n".join(([rest] if rest else []) + below)))
    return out, free


def _form_preamble(body_text: str, flat_body: str, by_lines: bool) -> str:
    """The prose an author wrote BEFORE the first field label.

    000036 San Fran TV mount: "Hi Trent, ... My customer, Checkout LLC, is
    renovating their office ... looking for a service partner to install their
    new Samsung 65' display." then "Adress:", then a signature with "Direct:".
    Both splitters started at the first label, so the request itself -- the
    only statement of the work -- produced no atom at all."""
    if by_lines:
        lines = [ln.strip() for ln in str(body_text or "").splitlines()]
        head: list[str] = []
        for ln in lines:
            if _line_label(ln):
                break
            if ln:
                head.append(ln)
        return " ".join(head).strip()
    text = " ".join(str(flat_body or "").split())
    marks = list(_INLINE_LABEL_RE.finditer(text))
    return text[: marks[0].start()].strip() if marks else ""


def _split_trailing_prose(value: str) -> tuple[str, str]:
    """(list part, trailing sentences) for a multi-line value.

    The last lines of a pasted work order are often sign-off prose ("The tech
    will have to work with the A1 engineer throughout the WO." / "If you have
    any questions, feel free to reach out.") under the final list. Items in the
    list do not end in a full stop; trailing lines that do are a note, not two
    more tasks. Only split when most of the list lines lack terminal punctuation,
    so a list whose every line is a sentence stays whole."""
    lines = [ln for ln in value.split("\n") if ln.strip()]
    if len(lines) < 3:
        return value, ""
    cut = len(lines)
    while cut > 0 and _SENTENCE_END_RE.search(lines[cut - 1]):
        cut -= 1
    head = lines[:cut]
    if cut == len(lines) or len(head) < 2:
        return value, ""
    if sum(1 for ln in head if _SENTENCE_END_RE.search(ln)) * 2 >= len(head):
        return value, ""
    return "\n".join(head), " ".join(lines[cut:])


def _split_inline_fields(body: str) -> list[tuple[str, str]]:
    """'Address: X Date | Time: Y Duration: Z' -> [(Address, X), (Date | Time, Y), (Duration, Z)].
    Requires two or more labels; otherwise the body is prose and is left alone."""
    text = " ".join(str(body or "").split())
    marks = list(_INLINE_LABEL_RE.finditer(text))
    if len(marks) < 2:
        return []
    out: list[tuple[str, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        label = " ".join(m.group("label").split())
        value = text[m.end():end].strip(" ;,")
        out.append((label, value))
    return out


# " -Relay" / " -Mag Lock Cable": a dash glued to the item, after a space.
_INLINE_DASH_ITEM_RE = re.compile(r"(?:^|\s)-(?=[A-Za-z0-9(])")


def _split_list_value(value: str) -> list[str]:
    if "\n" in value:
        lines = [ln.strip(" .;") for ln in value.split("\n") if ln.strip(" .;")]
        return [ln for ln in lines if len(re.sub(r"[^A-Za-z0-9]", "", ln)) >= 3]
    parts = [p.strip(" .;") for p in re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+(?=[a-z])", value) if p.strip(" .;")]
    return [p for p in parts if len(re.sub(r"[^A-Za-z0-9]", "", p)) >= 3]


def _field_value_shape(value: str) -> str:
    lines = [ln.strip() for ln in value.strip().split("\n") if ln.strip()]
    v = ", ".join(lines)
    if v.endswith("?"):
        return "question"
    if re.search(r"\b\d{1,5}[A-Za-z]?(?:/\d+[A-Za-z]?)?\b", v) and re.search(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+)\b.*\b(?:[A-Z][a-z]+|[A-Z]{2,})\b", v) and (
        re.search(r"\b(?:FLR|Floor|Suite|Ste|Campus|Road|Rd|Street|St|Avenue|Ave|Park|Tower|Building|Bldg|Plaza|Drive|Dr|Lane|Ln|Blvd|Highway|Hwy)\b", v, re.I)
        or re.search(r"\b[A-Z]{2}\s+\d{5}\b", v)
        or v.count(",") >= 2 and len(v.split()) >= 6 and not re.search(r"\b(?:of|and|the|with|for)\b", v.split(",")[0], re.I)
    ):
        # The shape test above accepts "a digit, two capitals, and a comma",
        # which is most of English prose. A CRM note is prose, so it minted
        # sites named "Job Tasks Console Access Provisioning Connect" whose
        # address was the whole scope narrative, and "Se Atlanta Office" whose
        # address was an email sign-off. Confirm the value actually has the
        # grammar of a street line before calling it one.
        from app.core.address_parse import looks_like_street_address

        if looks_like_street_address(v):
            return "address"
        if len(lines) < 2:
            return "other"
        # Several lines that are not a street address are a list the author
        # typed one per line (000020's "Hardware involved:" block), not prose.
    if _COUNT_ROLE_RE.match(v):
        return "staffing"
    if _TIME_UNIT_RE.search(v) and len(v.split()) <= 8:
        return "duration"
    if len(lines) >= 2 or len(_split_list_value(v)) >= 3:
        return "list"
    return "other"


def _address_facility(value: str) -> str:
    """The named place inside an address: the comma segment with the most
    capitalised words that is not the street number line."""
    best = ""
    for seg in value.split(","):
        run: list[str] = []
        for w in seg.split():
            if w[:1].isupper() and re.fullmatch(r"[A-Za-z&'.-]+", w):
                run.append(w)
            else:
                if len(" ".join(run)) > len(best):
                    best = " ".join(run)
                run = []
        if len(" ".join(run)) > len(best):
            best = " ".join(run)
    return best[:80]


def is_hubspot_note_path(path: Path, sample_text: str | None = None) -> bool:
    name = path.name.lower()
    if _HS_NOTE_FILENAME_RE.search(name):
        return True
    text = (sample_text or "")[:2000]
    return bool(_HS_NOTE_HEADER_RE.search(text))


def _name_key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' -]", " ", str(text or "").lower()).split())


def _is_author_line(line: str, author: str, author_email: str) -> bool:
    """Is this line nothing but the note author's name ("Megan Blevins",
    "- Megan", "Megan Blevins," as a sign-off)?

    Live 010353: Megan Blevins' Oct 1 note opened with her bare name and it
    became a stakeholder atom. Matched against the note's own author metadata
    (the ``Author:`` name, else the ``Author-Email`` local part), never
    guessed from capitalisation alone, so "Camden Site" or a customer's name
    in a body stays content."""
    t = str(line or "").strip()
    t = re.sub(r"^[-~\u2013\u2014\s]+", "", t).strip(" .,;:!")
    if not t or len(t.split()) > 4 or re.search(r"[\d@:/]", t):
        return False
    key = _name_key(t)
    names: list[str] = []
    a = _name_key(author)
    if a and a not in _UNRESOLVED_AUTHORS:
        names.append(a)
    local = str(author_email or "").split("@", 1)[0]
    if "@" in str(author_email or "") and re.search(r"[._-]", local):
        names.append(_name_key(re.sub(r"[._-]+", " ", local)))
    for n in names:
        parts = n.split()
        if key == n or (len(parts) >= 2 and key == parts[0]):
            return True
    return False


def _strip_author_lines(body_lines: list[str], author: str, author_email: str) -> tuple[list[str], str]:
    """``(body lines without the author's name line, that line)``. Only the
    first or last line, and never the only line."""
    lines = list(body_lines)
    found = ""
    for idx in (0, -1):
        if len(lines) < 2:
            break
        if _is_author_line(lines[idx], author, author_email):
            found = found or lines[idx].strip()
            lines.pop(idx)
    return lines, found


def parse_hubspot_note_text(raw: str) -> dict[str, Any]:
    """Split HubSpot export headers from the note body.

    Two artifact shapes exist in the wild:

    1. A full HubSpot export with ``HubSpot Note:`` / ``HubSpot Note ID:`` /
       ``Date:`` / ``Author:`` header lines followed by the body.
    2. A header-less ``title`` + blank line + ``body`` (how the deal-uploads
       pipeline actually writes notes to blob) — or a bare body.

    Shape 2 has no ``HubSpot Note:`` header, so the header state machine below
    never leaves its pre-body state and drops every line, yielding zero atoms
    (``ok_empty``) and silently losing address/scope facts (this is why the
    address note's city never reached ``site_facility_head``). Detect the
    header-less shape up front and treat the content directly as the body,
    keeping the first line as the title.
    """
    # HubSpot exports notes with HTML entities ("5 6 7 &amp; 8 FLR", live
    # 010297); the text is what the author typed, not its encoding.
    raw = _html.unescape(raw or "")
    lines = [ln.rstrip() for ln in (raw or "").splitlines()]
    if not _HS_NOTE_HEADER_RE.search(raw or ""):
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        title = non_empty[0] if non_empty else ""
        body = " ".join(non_empty[1:]).strip() if len(non_empty) > 1 else title
        return {
            "title": title,
            "note_id": "",
            "date_raw": "",
            "author": "",
            "author_email": "",
            "body": body,
            "raw_lines": lines,
            "body_line_index": 0,
        }
    title = ""
    note_id = ""
    date_raw = ""
    author = ""
    author_email = ""
    body_lines: list[str] = []
    in_body = False
    body_line_index = len(lines)
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped and not in_body:
            continue
        if in_body and stripped and body_line_index == len(lines):
            body_line_index = line_index
        if not in_body:
            m = re.match(r"^HubSpot Note:\s*(.*)$", stripped, re.I)
            if m:
                title = m.group(1).strip()
                continue
            m = _HS_NOTE_ID_RE.match(stripped)
            if m:
                note_id = m.group(1).strip()
                continue
            m = _HS_DATE_RE.match(stripped)
            if m:
                date_raw = m.group(1).strip()
                continue
            m = _HS_AUTHOR_EMAIL_RE.match(stripped)
            if m:
                author_email = m.group(1).strip()
                continue
            m = _HS_AUTHOR_RE.match(stripped)
            if m:
                author = m.group(1).strip()
                continue
            if not stripped and title:
                in_body = True
                continue
            if title and not note_id and not date_raw and not author and not author_email:
                body_lines.append(stripped)
                body_line_index = min(body_line_index, line_index)
                in_body = True
                continue
            if title:
                in_body = True
        if in_body and stripped:
            body_line_index = min(body_line_index, line_index)
            body_lines.append(stripped)
    body_lines, author_line = _strip_author_lines(body_lines, author, author_email)
    body = " ".join(body_lines).strip()
    if not body and title:
        body = title
    return {
        "title": title,
        "note_id": note_id,
        "date_raw": date_raw,
        "author": author,
        "author_email": author_email,
        # The author's own name typed as the body's first or last line: it
        # says who wrote the note -- metadata of the note, not a statement.
        "author_line": author_line,
        "body": body,
        # The flattened ``body`` above is what every prose consumer wants, but
        # joining on spaces destroys a pasted TABLE: a roster's rows and columns
        # become one run of words. Carry the lines too, so a delimited table can
        # still be recovered. Additive -- ``body`` is unchanged.
        "body_lines": list(body_lines),
        # Where the body begins in the file, so an atom can point at the body
        # line and not at the identical title in the ``HubSpot Note:`` header.
        "raw_lines": lines,
        "body_line_index": body_line_index,
    }


class HubspotNoteParser(BaseParser):
    parser_name = "hubspot_note"
    parser_version = "hubspot_note_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".txt"],
        supported_artifact_types=[ArtifactType.txt],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.customer_instruction,
            AtomType.deal_metadata,
            AtomType.commercial_total,
            AtomType.physical_site,
            AtomType.constraint,
            AtomType.open_question,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del domain_pack
        if path.suffix.lower() != ".txt":
            return ParserMatch(
                parser_name=self.parser_name,
                confidence=0.0,
                reasons=["not_txt"],
                artifact_type=ArtifactType.txt,
            )
        reasons: list[str] = []
        confidence = 0.0
        # These two were inverted: the NAME scored 0.97 and the parser's own
        # CONTENT signal scored 0.94, so "-hs-note-" in a filename outranked
        # an actual "HubSpot Note:" header -- and 0.97 was the highest
        # confidence anywhere in the registry, above RFC-5322 headers (0.91)
        # and document structure (0.90).
        #
        # Swept over 2500 real artifacts: renaming any .txt to
        # "acme-hs-note-99213.txt" moved it here at 0.97 regardless of what
        # was in it, including files whose content routes to TranscriptParser.
        #
        # Content now ranks above the name. The name stays ABOVE
        # MATCH_THRESHOLD, deliberately and unlike the other filename priors
        # in this router: "-hs-note-" is emitted by HubSpot's own exporter
        # rather than chosen by a person, and no local corpus contains a
        # single hs-note file, so there is no evidence that header-less
        # exports do not exist. Demoting it below threshold on no data could
        # silently drop a whole class of artifact. It is placed at 0.90 --
        # under this parser's header signal and under every content signal in
        # the registry.
        #
        # It sits at the EXTENSION tier (0.58) rather than the filename tier,
        # which is the honest classification: "-hs-note-" is a convention --
        # checkable, forgeable, machine-generated -- exactly what an extension
        # is. That placement keeps a header-less export claimable (0.58 clears
        # MATCH_THRESHOLD, so no class of artifact is dropped on no data)
        # while losing to every real content signal, so a transcript that
        # merely carries the token in its name stays a transcript.
        text = sample_text or ""
        if _HS_NOTE_HEADER_RE.search(text[:2000]):
            confidence = 0.94
            reasons.append("header:hubspot_note")
        if _HS_NOTE_FILENAME_RE.search(path.name):
            confidence = max(confidence, 0.58)
            reasons.append("filename:hs-note")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.txt,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact(project_id="unknown_project", artifact_id=artifact_id, path=artifact_path)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        return self.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        raw = read_text(path)
        parsed = parse_hubspot_note_text(raw)
        atoms = self._atoms_from_note(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            parsed=parsed,
        )
        atoms = _automated_sender_lines_as_chatter(atoms)
        if _note_is_automated(parsed):
            atoms = self._automated_note_as_chatter(
                atoms, project_id=project_id, artifact_id=artifact_id,
                filename=path.name, parsed=parsed,
            )
        _place_unlined_atoms(atoms, [str(x) for x in (parsed.get("raw_lines") or [])],
                             int(parsed.get("body_line_index") or 0))
        if not _note_is_automated(parsed):
            _credit_pasted_email(atoms, parsed)
        structured_doc = self._build_structured_doc(filename=path.name, parsed=parsed)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _automated_note_as_chatter(
        self,
        atoms: list[EvidenceAtom],
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        parsed: dict[str, Any],
    ) -> list[EvidenceAtom]:
        """A note an automated author wrote is a notification, not evidence.

        Live 010353: "OxBlue customer portal intake received.", posted by the
        HubSpot system user when the portal created the deal, became eight
        deal_metadata atoms restating the intake that the intake JSON already
        states. The labelling checklist is explicit: an automated sender is
        always a reject and never a stakeholder. So the note becomes ONE
        chatter atom flagged ``automated_sender`` and rejected. Two things
        survive beside it: the note's own header metadata (non_deal, flagged
        too), and a field whose value is an identifier or a date -- an intake
        id may be stated nowhere else.
        """
        from app.core.admission_chatter import mark_admission_chatter

        author = str(parsed.get("author") or parsed.get("author_email") or "").strip()
        kept: list[EvidenceAtom] = []
        for atom in atoms:
            val = atom.value if isinstance(atom.value, dict) else {}
            kind = str(val.get("kind") or "")
            at = getattr(atom.atom_type, "value", atom.atom_type)
            keep = kind == "hubspot_note_meta" or (
                kind == "note_field"
                and at != "stakeholder"
                and bool(_ID_OR_DATE_VALUE_RE.match(
                    str(atom.raw_text or "").split(":", 1)[-1].strip()
                ))
            )
            if not keep:
                continue
            val = dict(val)
            val["automated_sender"] = author
            atom.value = val
            if "automated_sender" not in (atom.review_flags or []):
                atom.review_flags = [*(atom.review_flags or []), "automated_sender"]
            kept.append(atom)

        title = str(parsed.get("title") or "").strip()
        body = "\n".join(str(ln) for ln in (parsed.get("body_lines") or []) if str(ln).strip())
        body = body or str(parsed.get("body") or "").strip()
        text = body
        if title and not _is_placeholder_note_title(title) and not body.lower().startswith(title.lower()):
            text = f"{title}\n{body}"
        if not text:
            return kept
        chatter = self._mint_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            atom_type=AtomType.deal_metadata, text=text,
            value={
                "kind": "automated_note", "non_deal": True,
                "hubspot_note_id": str(parsed.get("note_id") or ""),
                "author": str(parsed.get("author") or ""),
                "author_email": str(parsed.get("author_email") or ""),
                "date": str(parsed.get("date_raw") or ""),
                "title": title, "source": "hubspot_note",
                "automated_sender": author,
            },
            source_ref=self._base_source_ref(artifact_id, filename),
            confidence=0.5,
            review_flags=["automated_sender"],
        )
        mark_admission_chatter(chatter, "automated_sender")
        chatter.review_status = ReviewStatus.rejected
        return [chatter, *kept]

    def _base_source_ref(self, artifact_id: str, filename: str) -> SourceRef:
        return SourceRef(
            id=stable_id("src", artifact_id, "hubspot_note"),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=filename,
            locator={"kind": "hubspot_note_body"},
            extraction_method="hubspot_note_parser",
            parser_version=self.parser_version,
        )

    def _atoms_from_pasted_roster(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        parsed: dict[str, Any],
        source_ref: SourceRef,
        affiliation: str = "unknown",
    ) -> list[EvidenceAtom]:
        """``physical_site`` atoms for a site roster pasted into the note.

        Returns [] when the note holds no delimited table, or when the table is
        not a roster by ``looks_like_site_roster``'s own judgment. The identity
        of each site comes from the roster's OWN columns -- the name the PM
        wrote down -- never from geography reassembled afterwards.
        """
        try:
            from app.parsers.note_site_roster import (
                site_entity_key,
                site_roster_from_note_lines,
            )
        except Exception:  # pragma: no cover - never break a note parse
            return []

        lines = parsed.get("body_lines") or []
        if not lines:
            return []
        title = str(parsed.get("title") or "")
        try:
            roster_rows, columns, rows = site_roster_from_note_lines(
                lines, surrounding_text=title, deal_id=project_id
            )
        except Exception:  # pragma: no cover
            return []
        if not roster_rows:
            return []

        out: list[EvidenceAtom] = []
        for site_row in roster_rows:
            key = site_entity_key(site_row)
            if not key:
                continue
            canon = (
                (getattr(site_row, "site_id", "") or "")
                or (getattr(site_row, "facility_name", "") or "")
                or (getattr(site_row, "street_address", "") or "")
            )
            text_parts = []
            for label, val in (
                ("facility", getattr(site_row, "facility_name", "")),
                ("address", getattr(site_row, "street_address", "")),
                ("city", getattr(site_row, "city", "")),
                ("state", getattr(site_row, "state", "")),
                ("zip", getattr(site_row, "zip", "")),
            ):
                if val:
                    text_parts.append(f"{label}: {val}")
            row_text = " | ".join(text_parts) or canon
            idx = getattr(site_row, "row_index", None)
            cells = {}
            if isinstance(idx, int) and 0 <= idx < len(rows):
                cells = {c: v for c, v in zip(columns, rows[idx])}
            out.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.physical_site,
                    text=row_text,
                    value={
                        "kind": "physical_site",
                        "id": canon,
                        "site_id": getattr(site_row, "site_id", "") or canon,
                        "name": getattr(site_row, "facility_name", "") or canon,
                        "facility_name": getattr(site_row, "facility_name", ""),
                        "address": getattr(site_row, "street_address", ""),
                        "street_address": getattr(site_row, "street_address", ""),
                        "city": getattr(site_row, "city", ""),
                        "state": getattr(site_row, "state", ""),
                        "city_state": getattr(site_row, "city_state", ""),
                        "zip": getattr(site_row, "zip", ""),
                        "contact": getattr(site_row, "contact", ""),
                        "phone": getattr(site_row, "phone", ""),
                        "email": getattr(site_row, "email", ""),
                        "notes": getattr(site_row, "notes", ""),
                        # Every column survives, not just the canonical ones.
                        "cells": cells,
                        "source": "hubspot_note_pasted_roster",
                    },
                    source_ref=source_ref,
                    confidence=0.86,
                    entity_keys=[key],
                    review_flags=["note_site_roster_v1"],
                    author_affiliation=affiliation,
                )
            )
        return out

    def _mint_atom(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        atom_type: AtomType,
        text: str,
        value: dict[str, Any],
        source_ref: SourceRef,
        confidence: float = 0.8,
        review_flags: list[str] | None = None,
        entity_keys: list[str] | None = None,
        author_affiliation: str = "unknown",
    ) -> EvidenceAtom:
        flags = list(review_flags or [])
        if "hubspot_note_parser" not in flags:
            flags.append("hubspot_note_parser")
        val = dict(value)
        conf = confidence
        if author_affiliation == "internal":
            conf, flags, val = apply_internal_author_elevation(
                confidence=conf,
                review_flags=flags,
                value=val,
            )
        elif author_affiliation in {"external", "unknown"}:
            val.setdefault("author_affiliation", author_affiliation)
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, atom_type.value, text[:160]),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=text[:4000],
            normalized_text=normalize_text(text),
            value=val,
            entity_keys=list(entity_keys or []),
            source_refs=[source_ref],
            authority_class=AuthorityClass.meeting_note,
            confidence=conf,
            review_status=ReviewStatus.auto_accepted,
            review_flags=flags,
            parser_version=self.parser_version,
        )

    def _atoms_for_note_field(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        label: str,
        value: str,
        note_id: str,
        title: str,
        author: str,
        author_email: str,
        affiliation: str,
        source_ref: SourceRef,
    ) -> list[EvidenceAtom]:
        """One field of an inline-labelled note -> atoms typed by the VALUE's
        shape (never by the label's words): an address shape is a site, a
        number with a time unit is a duration constraint, a count with a
        capitalised role is staffing, a comma list of three or more phrases is
        scope (one item each, plus the whole), anything else a labelled fact."""
        out: list[EvidenceAtom] = []
        base = {
            "kind": "note_field", "field_name": label, "hubspot_note_id": note_id,
            "title": title, "source": "hubspot_note", "author": author,
            "author_email": author_email, "author_affiliation": affiliation,
        }
        remainder = ""
        # An address value runs until the next label, which in a pasted email
        # is often the sender's signature ("SAN FRANCISCO, CA 94111-2201 Sarah
        # Halpern Account Manager | CDW 72 Madison Avenue | New York, NY 10016",
        # 000036). The address ends at its first City, ST ZIP; what follows is
        # something else, and read as part of the address it published a second
        # site in New York.
        from app.core.address_parse import _CITY_STATE_ZIP_RE, looks_like_street_address as _is_street

        flat = ", ".join(ln.strip() for ln in value.split("\n") if ln.strip())
        _m = _CITY_STATE_ZIP_RE.search(flat)
        if _m and flat[_m.end():].strip(" ,|") and _is_street(flat[: _m.end()]):
            value, remainder = flat[: _m.end()].strip(" ,"), flat[_m.end():].strip(" ,|")
        # A pasted email loses its line breaks in the note: "Provided by us:
        # -PC with Access Control Software -Relay -Local and Remote Extender"
        # (010289). Two or more dash-led items are a bullet list, one per line.
        if "\n" not in value and len(_INLINE_DASH_ITEM_RE.findall(" " + value)) >= 2:
            value = "\n".join(p.strip() for p in _INLINE_DASH_ITEM_RE.split(" " + value) if p.strip())
        shape = _field_value_shape(value)
        trailer = ""
        if shape == "list" and "\n" in value:
            value, trailer = _split_trailing_prose(value)
        if shape == "address":
            value = ", ".join(ln.strip() for ln in value.split("\n") if ln.strip())
        text = f"{label}: {value}"
        if trailer:
            out.append(self._mint_atom(
                project_id=project_id, artifact_id=artifact_id, filename=filename,
                # A condition on the work ("the tech will have to work with the
                # A1 engineer throughout"), not a task: typed so the learned
                # typer does not promote the sign-off into the labor roster.
                atom_type=AtomType.constraint, text=trailer,
                value={**base, "kind": "note_field_trailer", "parent_field": label},
                source_ref=source_ref, confidence=0.8, author_affiliation=affiliation,
            ))
        if shape == "address":
            slug = re.sub(r"[^a-z0-9]+", "_", _address_facility(value).lower()).strip("_") or re.sub(r"[^a-z0-9]+", "_", value.lower())[:40].strip("_")
            # A street line with no ZIP ("1179 Vestal Ave, Suite 1, Binghamton, NY")
            # never matched the City, ST ZIP anchor, so the site published with no
            # city or state. The last two comma segments are where they live.
            from app.core.address_parse import split_city_state_strict

            from app.core.address_parse import _CITY_STATE_ZIP_RE, _parsed_from_city_state_zip_match

            segs = [s.strip() for s in value.split(",") if s.strip()]
            locality: dict[str, Any] = {}
            zm = _CITY_STATE_ZIP_RE.search(value)
            parsed_zip = _parsed_from_city_state_zip_match(value, zm) if zm else None
            if parsed_zip and parsed_zip.city and parsed_zip.state:
                locality = {k: v for k, v in (("city", parsed_zip.city), ("state", parsed_zip.state), ("zip", parsed_zip.zip)) if v}
                street_line = value[: zm.start(1)].strip(" ,")
            else:
                city, state = split_city_state_strict(", ".join(segs[-2:])) if len(segs) >= 2 else (None, None)
                if city and state:
                    locality = {"city": city, "state": state}
                street_line = ", ".join(segs[:-2])
            out.append(self._mint_atom(
                project_id=project_id, artifact_id=artifact_id, filename=filename,
                atom_type=AtomType.physical_site, text=text,
                value={**base, "kind": "physical_site", "id": slug, "site_id": slug,
                       "name": _address_facility(value), "facility_name": _address_facility(value),
                       "address": value, "street_address": street_line if locality and street_line else value,
                       "inferred": False, **locality},
                source_ref=source_ref, confidence=0.86, entity_keys=[f"site:{slug}"],
                review_flags=["hubspot_note_physical_site"], author_affiliation=affiliation,
            ))
            if remainder:
                out.append(self._mint_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=filename,
                    atom_type=AtomType.deal_metadata, text=remainder,
                    value={**base, "kind": "note_field_remainder", "parent_field": label},
                    source_ref=source_ref, confidence=0.7, author_affiliation=affiliation,
                ))
            return out
        if shape == "list":
            # A one-line list keeps its whole-field atom (the items are cut from
            # it). A list the author typed one item per line does not: that
            # atom is every line again, and the typer promoted it to a task
            # that duplicated all eleven of 000020's onsite tasks. Each item
            # carries ``parent_field``, so the section is not lost.
            if "\n" not in value:
                out.append(self._mint_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=filename,
                    atom_type=AtomType.scope_item, text=text, value={**base, "items": _split_list_value(value)},
                    source_ref=source_ref, confidence=0.84, review_flags=["hubspot_note_training_row"],
                    author_affiliation=affiliation,
                ))
            for item in _split_list_value(value):
                out.append(self._mint_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=filename,
                    atom_type=AtomType.scope_item, text=item,
                    value={**base, "kind": "note_field_item", "parent_field": label,
                           "list_item": True, "list_label": f"{label}:"},
                    source_ref=source_ref, confidence=0.82, author_affiliation=affiliation,
                ))
            return out
        atom_type = {
            "duration": AtomType.constraint,
            "staffing": AtomType.requirement,
            "question": AtomType.open_question,
        }.get(shape, AtomType.deal_metadata)
        extra: dict[str, Any] = {}
        # "Diagram: <500 characters of Outlook/Proofpoint wrapper>" is the
        # vendor's drawing (010289). Carry the real URL so the PM sees the
        # picture instead of a link they cannot read.
        if re.match(r"^https?://", value.strip()) and " " not in value.strip():
            from app.core.link_unwrap import is_image_url, unwrap_link

            direct = unwrap_link(value.strip())
            extra = {"url": direct, "wrapped_url": value.strip() if direct != value.strip() else None}
            if is_image_url(direct):
                extra.update({"kind": "note_field_image", "image_url": direct, "media_type": "image"})
        out.append(self._mint_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            atom_type=atom_type, text=text, value={**base, "shape": shape, "value": value, **extra},
            source_ref=source_ref, confidence=0.84 if shape != "other" else 0.8,
            author_affiliation=affiliation,
        ))
        return out

    def _atoms_from_note(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        parsed: dict[str, Any],
    ) -> list[EvidenceAtom]:
        body = str(parsed.get("body") or "").strip()
        title = str(parsed.get("title") or "").strip()
        note_id = str(parsed.get("note_id") or "").strip()
        author = str(parsed.get("author") or "").strip()
        author_email = str(parsed.get("author_email") or "").strip()
        date_raw = str(parsed.get("date_raw") or "").strip()
        affiliation = classify_author_affiliation(author, author_email=author_email or None)
        source_ref = self._base_source_ref(artifact_id, filename)
        atoms: list[EvidenceAtom] = []
        train_rows: list[TrainingRow] = []

        meta_bits = []
        if note_id:
            meta_bits.append(f"note_id={note_id}")
        if author:
            meta_bits.append(f"author={author}")
        if author_email:
            meta_bits.append(f"author_email={author_email}")
        if affiliation != "unknown":
            meta_bits.append(f"author_affiliation={affiliation}")
        if date_raw:
            meta_bits.append(f"date={date_raw}")
        if meta_bits:
            meta_text = " | ".join(meta_bits)
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.deal_metadata,
                    text=meta_text,
                    value={
                        # Who wrote the note and when -- about the note, never
                        # about the deal. Marked so no downstream re-typer can
                        # promote it: on dev (000020, compile e2718776) span
                        # admission re-typed this very atom to `task` and Deal
                        # Kit would have proposed "note_id=... author=...".
                        "kind": "hubspot_note_meta",
                        "non_deal": True,
                        "field_name": "hubspot_note_meta",
                        "hubspot_note_id": note_id,
                        "author": author,
                        "author_email": author_email,
                        "author_affiliation": affiliation,
                        "date": date_raw,
                        "title": title,
                        "source": "hubspot_note",
                    },
                    source_ref=source_ref,
                    confidence=0.88,
                    author_affiliation=affiliation,
                )
            )
            # "note_id=... | author=..." is composed, not written: flag it and
            # point at the export's own header lines ("HubSpot Note ID: ...",
            # "Author: ...") so a reviewer highlights those instead.
            from app.parsers.synthetic_text import find_source_lines, mark_synthetic

            _labels = [lab for lab, have in (("HubSpot Note", title), ("HubSpot Note ID", note_id),
                                             ("Date", date_raw), ("Author", author),
                                             ("Author-Email", author_email)) if have]
            _body_at = int(parsed.get("body_line_index") or 0)
            mark_synthetic(atoms[-1], find_source_lines(
                [str(ln) for ln in (parsed.get("raw_lines") or [])][: _body_at or None], _labels))

        def _mint_prose(prose: str) -> None:
            # A paragraph is several statements. Minting it whole made 000036's
            # request one atom whose text began "Hope all is well!", so the Deal
            # Kit task was named by the greeting and every head judged the
            # greeting, the context and the ask together. Each sentence is its
            # own evidence, the way email and transcript lines already are;
            # the paragraph travels on each as context.
            from app.core.sentences import split_inline_dash_list, split_sentences, split_trigger_clause

            # The author's own line breaks come first: "Hi Trent," on its own
            # line is a greeting, not the start of the request beneath it
            # (010095), and no sentence segmenter splits after a comma.
            sentences: list[str] = []
            for line in str(prose or "").splitlines() or [str(prose or "")]:
                # A bullet list flattened onto one line ("Support for ... -
                # Support for ... - 24/7 on-call availability", live 000132)
                # is one item per " - " segment, not one atom.
                dash_items = split_inline_dash_list(line)
                if dash_items:
                    sentences.extend(p for p in dash_items if p.strip())
                    continue
                for _s in split_sentences(line):
                    sentences.extend(p for p in split_trigger_clause(_s.strip()) if p)
            # A note file repeats its title as the body's first line; a body
            # that IS the title ("Need Troy and Wilmington sites removed.",
            # live 000132) is one statement, not two.
            sentences = list(dict.fromkeys(sentences))
            if len(sentences) > 1:
                for sentence in sentences:
                    _mint_prose_one(sentence, paragraph=" ".join(str(prose or "").split()))
                return
            if len(sentences) == 1:
                _mint_prose_one(" ".join(sentences[0].split()), paragraph=None)
                return
            _mint_prose_one(" ".join(str(prose or "").split()), paragraph=None)

        raw_lines = [str(ln) for ln in (parsed.get("raw_lines") or [])]
        body_line_index = int(parsed.get("body_line_index") or 0)
        from app.core.city_site_list import find_city_site_lists

        _city_sites = {
            " ".join(cs.text.split()).lower(): cs
            for cs in find_city_site_lists([str(ln) for ln in (parsed.get("body_lines") or [])])
        }
        # The statements of this note in order, so an "Update on that" line can
        # name the one it revises.
        statements: list[list[EvidenceAtom]] = []

        def _prose_source_ref(prose: str) -> SourceRef:
            """The BODY line(s) this prose came from. A note's first line is
            also its title, so matching the text anywhere in the file finds the
            ``HubSpot Note:`` header first and the viewer highlights that."""
            words = " ".join(str(prose or "").split())
            if not words or not raw_lines:
                return source_ref
            probe = words[: min(len(words), 40)]
            # The line that holds the WHOLE text wins over the first line that
            # merely starts the same way: a note repeats its title, truncated
            # ("Maintenance and support of their technical IT infrastructure
            # covering network…"), as the body's first line, and the 40-char
            # probe found that title for the full sentence two lines below.
            normed = [" ".join(raw_lines[i].split()) for i in range(len(raw_lines))]
            whole = next((i for i in range(body_line_index, len(raw_lines)) if words in normed[i]), None)
            for i in ([whole] if whole is not None else range(body_line_index, len(raw_lines))):
                joined = normed[i]
                col = joined.find(words if whole is not None else probe)
                if col < 0:
                    continue
                # A sentence the author wrapped runs onto following lines.
                end = i
                acc = joined[col:]
                while len(acc) < len(words) and end + 1 < len(raw_lines) and raw_lines[end + 1].strip():
                    end += 1
                    acc = f"{acc} {' '.join(raw_lines[end].split())}"
                locator = dict(source_ref.locator or {})
                # Where on the line it starts, too. A bullet list flattened
                # onto one line (" - "-separated, live 000132) is ten atoms on
                # ONE line; with only the line number they tie, and every
                # reader broke the tie by atom id -- a hash -- so the list
                # read in random order ("Troubleshooting" above "Support for
                # physical server"). The column is the source order.
                locator.update({"line_start": i + 1, "line_end": end + 1, "region": "body",
                                "char_start": col})
                return source_ref.model_copy(update={
                    "id": stable_id("src", artifact_id, "hubspot_note", str(i + 1), probe),
                    "locator": locator,
                })
            return source_ref

        def _mint_prose_one(prose: str, paragraph: str | None) -> None:
            is_title = bool(
                prose and title
                and " ".join(prose.lower().split()) == " ".join(title.lower().split())
            )
            first_atom = len(atoms)
            _mint_prose_typed(prose, paragraph, is_title)
            minted = atoms[first_atom:]
            if not minted:
                return
            # "Update on that they do have wall mounts and parking": the author
            # revising the statement just before, within this note. Both ends
            # carry the link so neither is read without the other.
            if statements and _UPDATE_CUE_RE.match(prose):
                earlier = statements[-1]
                for a in minted:
                    a.value["supersedes"] = earlier[0].id
                for a in earlier:
                    a.value["superseded_by"] = minted[0].id
            statements.append(minted)

        def _mint_prose_typed(prose: str, paragraph: str | None, is_title: bool) -> None:
            prose_ref = _prose_source_ref(prose)
            if is_title and _is_placeholder_note_title(prose):
                # "Note", "Call" -- the CRM's default title with nothing under
                # it. Genuinely empty; the header atom already records the note.
                return
            city_site = None if is_title else _city_sites.get(" ".join(str(prose or "").split()).lower())
            if city_site is not None:
                # A line of a "City, ST" list is a job site (live 000132's
                # "Locations" list), not a sentence of scope.
                from app.core.city_site_list import city_site_value

                atoms.append(self._mint_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=filename,
                    atom_type=AtomType.physical_site, text=city_site.text,
                    value=city_site_value(
                        city_site, hubspot_note_id=note_id, title=title, source="hubspot_note",
                        author=author, author_email=author_email, author_affiliation=affiliation,
                    ),
                    source_ref=prose_ref, confidence=0.8, entity_keys=[city_site.entity_key],
                    review_flags=["hubspot_note_physical_site", "city_site_list"],
                    author_affiliation=affiliation,
                ))
                return
            atom_types: list[AtomType] = []
            chatter_reason: str | None = None
            from app.core.sentences import sentence_kind as _sentence_kind

            if is_title and _is_upload_caption(prose):
                # The note's prose IS its title: a caption on an upload ("SOW",
                # "psow from current partner"), not scope. Live 010300: three such
                # notes each became a scope_item. It is still the note's content,
                # so it is minted -- this branch used to set the type and then
                # mint nothing, which left every title-only note ("PO!!", "Need
                # Troy and Wilmington sites removed." on 000132) with no atom.
                atom_types = [AtomType.deal_metadata]
            elif prose and not is_title and _sentence_kind(prose) == "banter":
                # "Hope you had a great 4th of July!", "Thanks!": a pleasantry,
                # not scope. Kept as an admission-chatter atom, the way the
                # email parser keeps the same line, so the admission head sees
                # the negative and no other head reads it.
                atom_types = [AtomType.deal_metadata]
                chatter_reason = "banter"
            elif prose:
                atom_types = [AtomType.scope_item]
                if _INSTRUCTION_RE.search(prose):
                    atom_types.append(AtomType.customer_instruction)
                if _ROM_RE.search(prose):
                    atom_types.append(AtomType.commercial_total)
                if prose.endswith("?"):
                    atom_types.append(AtomType.open_question)
                # No word list decides that prose is a constraint. "remote" and
                # "onsite" appear in almost every request for work, so 000036's
                # "Most of their IT team is remote ... install their new Samsung
                # 65' display" was typed a constraint and deduped out of scope.
                # What a sentence is gets learned downstream (typed classifier,
                # taught corrections), not matched here.

            if prose and atom_types:
                deduped: list[AtomType] = []
                for at in atom_types:
                    if at not in deduped:
                        deduped.append(at)

                for at in deduped:
                    val: dict[str, Any] = {
                        "text": prose,
                        "kind": "hubspot_note_body",
                        "hubspot_note_id": note_id,
                        "title": title,
                        "source": "hubspot_note",
                        "author": author,
                        "author_email": author_email,
                        "author_affiliation": affiliation,
                    }
                    if paragraph:
                        val["paragraph"] = paragraph
                    if at == AtomType.commercial_total:
                        amounts = re.findall(r"\$\s*(\d[\d,]*(?:\.\d{2})?)", prose)
                        k_amounts = re.findall(r"\b(\d{1,3}(?:,\d{3})*)\s*[kK]\b", prose)
                        val.update(
                            {
                                "category": "ROM",
                                "currency": "USD",
                                "amounts": amounts,
                                "k_amounts": k_amounts,
                                "rom_text": prose,
                            }
                        )
                    atoms.append(
                        self._mint_atom(
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=filename,
                            atom_type=at,
                            text=prose,
                            value=val,
                            source_ref=prose_ref,
                            confidence=0.84 if at == AtomType.scope_item else 0.8,
                            review_flags=["hubspot_note_training_row"] if at == AtomType.scope_item else [],
                            author_affiliation=affiliation,
                        )
                    )
                    if chatter_reason:
                        from app.core.admission_chatter import mark_admission_chatter

                        mark_admission_chatter(atoms[-1], chatter_reason)

                if AtomType.scope_item not in deduped:
                    return
                train_rows.append(
                    TrainingRow(
                        relation=HUBSPOT_NOTE_RELATION,
                        label="scope_item",
                        raw_text=prose[:4000],
                        label_kind="judgment",
                        teacher=TEACHER_STORE,
                        confidence=0.84,
                        deal_id=project_id,
                        project_id=project_id,
                        provenance={
                            "note_id": note_id,
                            "title": title,
                            "source": "hubspot_note_parser",
                            "author_affiliation": affiliation,
                        },
                    )
                )

        # A pasted TABLE is a roster, not prose. When the note body carries a
        # delimited table, hand it to the same gate + extractor the spreadsheet
        # path uses, so a site roster is read the same way whichever door it
        # came in through. Deal 010310 is why: a five-column site table reached
        # this parser as one line and published four sites for three rows, with
        # the site named "Malport" -- the only name not also a city -- lost.
        roster_atoms = self._atoms_from_pasted_roster(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parsed=parsed,
            source_ref=source_ref,
            affiliation=affiliation,
        )
        if roster_atoms:
            atoms.extend(roster_atoms)

        # A note typed as a run of "Label: value" fields on one line ("Address:
        # ... Date | Time: ... Duration: ... Tech/Engineer: ... Scope of work:
        # ...", live 010297) is a form, not a sentence: one atom per field, or
        # the whole note becomes a single "site" whose address is the entire
        # text and the scope inside it is never seen.
        # ``body`` is the note flattened to one line; ``body_lines`` keeps the
        # author's line breaks, which are the only record of where one task
        # ends and the next begins.
        body_text = "\n".join(str(ln) for ln in (parsed.get("body_lines") or []))
        line_fields, free_lines = _split_line_fields_and_free_lines(body_text) if body else ([], [])
        fields = line_fields or (_split_inline_fields(body) if body else [])
        if len(fields) >= 2:
            if line_fields:
                # Prose the author wrote outside any field: before the first
                # label, or under a label its own line already closed.
                from app.parsers.value_shapes import classify_value

                block: list[str] = []

                def _flush() -> None:
                    text = " ".join(block).strip()
                    block.clear()
                    if len(text.split()) >= 4:
                        _mint_prose(text)

                for ln in free_lines:
                    if classify_value(ln) in ("email", "phone", "postal", "state"):
                        _flush()
                        atoms.append(self._mint_atom(
                            project_id=project_id, artifact_id=artifact_id, filename=filename,
                            atom_type=AtomType.deal_metadata, text=ln,
                            value={"kind": "note_line_metadata", "hubspot_note_id": note_id, "title": title,
                                   "source": "hubspot_note", "shape": classify_value(ln)},
                            source_ref=source_ref, confidence=0.7, author_affiliation=affiliation,
                        ))
                        continue
                    if _SENTENCE_END_RE.search(ln) or len(ln.split()) >= 6:
                        block.append(ln)
                        _flush()
                    else:
                        _flush()
                        if len(ln.split()) >= 4:
                            _mint_prose(ln)
                _flush()
            else:
                preamble = _form_preamble(body_text, body, by_lines=False)
                if len(preamble.split()) >= 4:
                    _mint_prose(preamble)
            for f_label, f_value in fields:
                if not f_value:
                    continue
                atoms.extend(
                    self._atoms_for_note_field(
                        project_id=project_id, artifact_id=artifact_id, filename=filename,
                        label=f_label, value=f_value, note_id=note_id, title=title,
                        author=author, author_email=author_email, affiliation=affiliation,
                        source_ref=source_ref,
                    )
                )
            if train_rows:
                log_rows(train_rows)
            return atoms

        _mint_prose(body_text if body_text.strip() else body)

        # Physical sites from address-bearing notes. A note's title is its
        # first line, so prepending it repeats that line: on 010043 the site
        # came out as "3 Verkada cameras intsall. 3 Verkada cameras intsall.
        # North Carolina office: 6125 Tyvola Centre Drive ...".
        _t = " ".join((title or "").split())
        corpus = body if _t and " ".join(body.split()).startswith(_t) else f"{title}\n{body}"
        for parsed_addr in find_us_addresses_in_text(corpus):
            if (
                not parsed_addr.city
                or not parsed_addr.state
                or parsed_addr.state not in US_STATES
                or not parsed_addr.street_address
            ):
                continue
            slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                f"{parsed_addr.city}_{parsed_addr.state}_{parsed_addr.zip or parsed_addr.street_address}".lower(),
            ).strip("_")
            display = (
                f"{parsed_addr.street_address}, {parsed_addr.city}, "
                f"{parsed_addr.state} {parsed_addr.zip or ''}"
            ).strip()
            aliases = list(dict.fromkeys(parsed_addr.aliases))
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.physical_site,
                    text=display,
                    value={
                        "kind": "physical_site",
                        "id": slug,
                        "site_id": slug,
                        "name": display,
                        "names": list(dict.fromkeys([display, parsed_addr.city, *aliases])),
                        "aliases": aliases,
                        "street_address": parsed_addr.street_address,
                        "address": parsed_addr.street_address,
                        "city": parsed_addr.city,
                        "state": parsed_addr.state,
                        "zip": parsed_addr.zip,
                        "inferred": True,
                        "source_context": corpus[:600],
                        "author_affiliation": affiliation,
                    },
                    source_ref=source_ref,
                    confidence=0.76,
                    entity_keys=[f"site:{slug}"],
                    review_flags=["hubspot_note_physical_site"],
                    author_affiliation=affiliation,
                )
            )

        if not atoms and (title or body):
            fallback = body or title
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.deal_metadata,
                    text=fallback,
                    value={
                        "field_name": "hubspot_note_body",
                        "text": fallback,
                        "hubspot_note_id": note_id,
                        "source": "hubspot_note",
                        "author_affiliation": affiliation,
                    },
                    source_ref=source_ref,
                    confidence=0.7,
                    author_affiliation=affiliation,
                )
            )

        if train_rows:
            log_rows(train_rows)
        return atoms

    def _build_structured_doc(self, *, filename: str, parsed: dict[str, Any]) -> dict[str, Any]:
        body = str(parsed.get("body") or "").strip() or "(empty note)"
        author = str(parsed.get("author") or "").strip()
        author_email = str(parsed.get("author_email") or "").strip()
        affiliation = classify_author_affiliation(author, author_email=author_email or None)
        meta = [
            f"note_id: {parsed.get('note_id') or ''}",
            f"author: {author}",
            f"author-email: {author_email}",
            f"author_affiliation: {affiliation}",
            f"date: {parsed.get('date_raw') or ''}",
        ]
        page = make_page(
            page=0,
            title=str(parsed.get("title") or filename),
            metadata=meta,
            sections=[
                {
                    **make_section(
                        heading="HubSpot Note",
                        level=2,
                        blocks=[make_paragraph(body)],
                    ),
                    # Who wrote it is the section's header, not a block of it.
                    **({"author": author} if author else {}),
                    **({"author_line": parsed["author_line"]} if parsed.get("author_line") else {}),
                }
            ],
        )
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_HUBSPOT_NOTE,
            filename=filename,
            artifact_type=ArtifactType.txt.value,
            title=filename,
            metadata=meta,
            pages=[page],
        )


__all__ = [
    "HUBSPOT_NOTE_RELATION",
    "HubspotNoteParser",
    "is_hubspot_note_path",
    "parse_hubspot_note_text",
]
