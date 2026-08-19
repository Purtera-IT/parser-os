"""Universal hybrid meeting-summary + full-transcript handling.

Many discovery / kickoff exports are a single PDF (or text dump) whose
**first page(s)** are a structured meeting summary and whose **remainder**
is a diarized transcript (``Speaker Name [mm:ss] …``). Routing the whole
file through the generic PDF prose atomizer turns transcript turns into
glued ``scope_item`` blobs that poison neural / embedding heads.

This module is UNIVERSAL — it keys off filename/title/content STRUCTURE
only (the word ``transcript``, summary markers, speaker+timestamp density).
It never hardcodes a deal, customer, or person name.

Public API
----------
- ``detect_hybrid_summary_transcript`` — filename/title/content → plan
- ``split_speaker_timestamp_turns`` — atomize ``Name [mm:ss]`` turns
- ``classify_transcript_turn_role`` — greeting/intro/logistics/acknowledgment/filler vs deal
- ``rewrite_hybrid_pdf_atoms`` — replace transcript-page prose atoms
- ``retag_conversational_to_meta`` — scope fluff → ``deal_metadata`` /
  ``kind=conversation_meta`` so heads ignore it while audit keeps it
- ``stamp_conversation_reply_adjacency`` — short acks ("Good.") carry
  ``in_reply_to`` / ``previous_*`` so they are never orphan tokens
- ``collapse_greeting_clusters`` — optional greeting→ack merge into one meta atom
- ``stitch_cross_page_speaker_turns`` — merge turns split at PDF page breaks
- ``is_head_excluded_atom`` — heads must skip ``non_deal`` / conversation_meta
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# ── title / filename signals (closed-class document chrome) ──
_TRANSCRIPT_TOKEN_RE = re.compile(r"\btranscripts?\b", re.IGNORECASE)
_SUMMARY_TOKEN_RE = re.compile(r"\b(?:meeting\s+)?summary\b", re.IGNORECASE)
_FULL_TRANSCRIPT_MARKER_RE = re.compile(
    r"(?:^|\n)\s*full\s+transcripts?\b",
    re.IGNORECASE,
)
_EXEC_SUMMARY_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:executive\s+)?summary\b",
    re.IGNORECASE,
)

# Diarized turn: "Alex Rivera [03:15]" or "Alex Rivera [03:15]:" or
# "Alex Rivera [03:15]: body". Also tolerates an optional colon after the stamp.
# STRUCTURAL — Capitalized name tokens + bracketed clock — never a name list.
# Name tokens allow hyphens/apostrophes and single-letter initials ("J."), but
# NOT a trailing period on a multi-letter word — otherwise "Hey.\nTrent … [00:56]"
# greedily becomes speaker="Hey.\\nTrent …".
# CRITICAL: only [ \t] between name tokens — never newlines. Sticky PDF
# section headers ("Key Decisions\nFull Transcript Alex … [00:04]") must NOT
# be absorbed into the speaker name.
_SPEAKER_TS_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s]))"
    r"(?P<speaker>[A-Z](?:[A-Za-z'\-]+|\.)(?:[ \t]+[A-Z](?:[A-Za-z0-9'\-]+|\.)){0,4})"
    r"[ \t]*\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\][ \t]*:?[ \t]*",
)

# Sticky / inline meeting-summary chrome that text-rich PDF extracts often
# repeat onto every transcript page as a heading (or glue before the first
# speaker stamp). Closed-class structural labels only — never deal vocabulary.
_MEETING_SECTION_CHROME_RE = re.compile(
    r"(?im)^[ \t]*(?:key\s+decisions?|action\s+items?|executive\s+summary|"
    r"open\s+questions?|attendees?|participants?|next\s+steps?|agenda|"
    r"discussion|notes|follow[\s-]?ups?|full\s+transcripts?|"
    r"meeting\s+summary(?:\s+and\s+full\s+transcripts?)?)"
    r"[ \t]*:?[ \t]*$"
)
_FULL_TRANSCRIPT_INLINE_RE = re.compile(
    r"(?im)(?:^|(?<=\n))[ \t]*full\s+transcripts?[ \t]*:?[ \t]+(?=[A-Z])"
)

# Greeting / intro / logistics — closed-class social language. A turn that
# ALSO carries deal substance is classified as deal (substance wins).
_GREETING_RE = re.compile(
    r"(?:"
    r"(?:^|\.\s*)(?:hi|hey|hello|good\s+(?:morning|afternoon|evening))\b|"
    r"(?:how\s+(?:are|you)|been\s+a\s+while|long\s+time\s+no|hope\s+life)|"
    r"(?:nice\s+to\s+meet|good\s+to\s+meet|pleased\s+to\s+meet|nice\s+meeting)|"
    r"(?:good\s+to\s+(?:see|hear)|great\s+to\s+(?:meet|see))"
    r")",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(
    r"(?:"
    r"(?:i(?:'m|\s+am)\s+(?:one\s+of\s+the\s+)?co[-\s]?founders?\b)|"
    r"(?:i(?:'m|\s+am)\s+\w[\w\-']*.{0,48}"
    r"(?:co[-\s]?founders?|engineer|manager|director|specialist|aes\b))|"
    r"(?:introduce\s+yourself|introductions?\s+real\s+quick|knock\s+those\s+out)|"
    r"(?:start\s+some\s+introductions)|"
    r"(?:i(?:'m|\s+am)\s+a\s+\w+)|"
    r"(?:looking\s+forward\s+to\s+digging\s+in)|"
    r"(?:joining\s+will\s+be\b)"
    r")",
    re.IGNORECASE,
)
_LOGISTICS_RE = re.compile(
    r"(?:"
    r"(?:we(?:'re| are)\s+(?:waiting|expecting|ready))|"
    r"(?:joining\s+now|ran\s+over|on\s+another\s+call)|"
    r"(?:can\s+you\s+(?:hear|repeat)|i(?:'m| am)\s+not\s+hearing)|"
    r"(?:sorry|apolog|no\s+worries|excuse\s+me)|"
    r"(?:call\s+him\s+on\s+(?:his\s+)?cell|team'?s?\s+message)|"
    # Call / meeting-room chrome — not deal scope.
    r"(?:forwarded\s+to\s+voicemail|voicemail)|"
    r"(?:having\s+trouble\s+with\s+the\s+(?:link|call|audio|video))|"
    r"(?:trouble\s+with\s+the\s+(?:link|call|audio|video))|"
    r"(?:i(?:'m| am)\s+sending\s+it\s+to\s+(?:him|her|them)\s+this\s+way)|"
    r"(?:ping\s+(?:him|her|them)|let\s+me\s+ping)|"
    r"(?:hop(?:ping)?\s+(?:in|on)|should\s+be\s+joining)"
    r")",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"(?:"
    r"(?:^|\.\s*)(?:thanks|thank\s+you|appreciate(?:\s+it)?|bye|goodbye|take\s+care)\b|"
    r"(?:talk\s+soon|catch\s+you\s+later|have\s+a\s+good\s+(?:one|day|night))"
    r")",
    re.IGNORECASE,
)
# Short acknowledgments / backchannel — must never stand alone without
# reply-to adjacency. Subset of filler; classified as ``acknowledgment``.
_ACK_ONLY_RE = re.compile(
    r"^(?:yeah|yep|yup|yes|no|nope|ok|okay|sure|right|cool|nice|great|"
    r"good|fine|perfect|absolutely|definitely|exactly|totally|agreed|"
    r"thanks|thank\s+you|got\s+it|gotcha|sounds\s+good|alright|"
    r"mm-?hmm|uh-?huh|mhm|true|correct|indeed|welcome)"
    r"[\s\.\,\!]*$",
    re.IGNORECASE,
)
_FILLER_ONLY_RE = _ACK_ONLY_RE  # back-compat alias; ack is the refined class

# Deal-substance signals (universal closed-class — not a vendor/deal list).
_DEAL_SUBSTANCE_RE = re.compile(
    r"\b(?:install|configure|deploy|integrat|setup|set\s+up|provision|"
    r"require|exclud|build|survey|upgrade|replace|implement|onboard|"
    r"walk\s+(?:through|him|her|them)|white\s+glove|badge\s+zone|"
    r"vlan|ssid|radius|access\s+control|badging|camera|firewall|"
    r"switch|router|access\s+point|reader|doorbell|equipment|hardware|"
    r"parts?\s+list|sow|quote|statement\s+of\s+work|scope|"
    r"idp|identity\s+provider|sso|saml|scim|okta|unifi|uid\s+enterprise|"
    r"nvr|dream\s+machine|hub|sensor|mount)\b",
    re.IGNORECASE,
)

TurnRole = Literal[
    "greeting",
    "intro",
    "logistics",
    "filler",
    "acknowledgment",
    "greeting_cluster",
    "chrome",
    "deal",
]

# Max prior-turn text kept on reply-to stamps (audit + head safety).
_REPLY_TO_TEXT_MAX = 160

# Roles that are conversational chrome — never deal substance for heads.
CONVERSATION_META_ROLES: frozenset[str] = frozenset(
    {
        "greeting",
        "intro",
        "logistics",
        "filler",
        "acknowledgment",
        "chrome",
        "greeting_cluster",
    }
)

CONVERSATION_META_KIND = "conversation_meta"
NON_DEAL_META_KINDS: frozenset[str] = frozenset(
    {
        CONVERSATION_META_KIND,
        "email_addressee",
        "email_body_context",
        "email_header",
    }
)

# Pure social roles eligible for greeting-cluster collapse.
_CLUSTERABLE_ROLES: frozenset[str] = frozenset(
    {"greeting", "intro", "acknowledgment", "filler"}
)


@dataclass(frozen=True)
class HybridPlan:
    """How to split a hybrid summary+transcript document."""

    kind: Literal["hybrid", "transcript_only"]
    transcript_start_page: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str | None
    timestamp: str | None
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class StitchedSpeakerTurn:
    """A speaker turn after cross-page stitch (may span pages)."""

    page_idx: int
    turn: SpeakerTurn
    page_end: int | None = None  # set when body continues onto a later page


def looks_like_transcript_filename(filename: str | None) -> bool:
    name = (filename or "").replace("_", " ").replace("-", " ")
    return bool(_TRANSCRIPT_TOKEN_RE.search(name))


def looks_like_summary_filename(filename: str | None) -> bool:
    name = (filename or "").replace("_", " ").replace("-", " ")
    return bool(_SUMMARY_TOKEN_RE.search(name))


def count_speaker_timestamp_hits(text: str) -> int:
    if not text:
        return 0
    return sum(1 for _ in _SPEAKER_TS_RE.finditer(text))


def detect_hybrid_summary_transcript(
    *,
    filename: str | None = None,
    title: str | None = None,
    text: str | None = None,
    page_texts: list[str] | None = None,
) -> HybridPlan | None:
    """Return a split plan when this artifact is a hybrid or pure transcript.

    Signals (any strong combination):
    - filename/title contains ``transcript``
    - body contains a ``Full Transcript`` marker after summary chrome
    - body has dense ``Name [mm:ss]`` speaker stamps
    """
    reasons: list[str] = []
    blob_title = " ".join(x for x in (filename or "", title or "") if x)
    if looks_like_transcript_filename(filename) or _TRANSCRIPT_TOKEN_RE.search(blob_title or ""):
        reasons.append("title_or_filename_transcript")

    pages = list(page_texts or [])
    if text and not pages:
        pages = [text]
    joined = "\n".join(pages)
    if not joined and not reasons:
        return None

    speaker_hits = count_speaker_timestamp_hits(joined)
    if speaker_hits >= 3:
        reasons.append(f"speaker_timestamp_density:{speaker_hits}")
    elif speaker_hits >= 1 and reasons:
        reasons.append(f"speaker_timestamp_sparse:{speaker_hits}")

    has_full_marker = bool(_FULL_TRANSCRIPT_MARKER_RE.search(joined))
    if has_full_marker:
        reasons.append("full_transcript_marker")

    has_summary = bool(
        looks_like_summary_filename(filename)
        or _SUMMARY_TOKEN_RE.search(blob_title or "")
        or _EXEC_SUMMARY_MARKER_RE.search(joined[:2500] if joined else "")
    )
    if has_summary:
        reasons.append("summary_signal")

    # Need at least a transcript signal (filename/marker/density).
    transcriptish = any(
        r.startswith(("title_or_filename", "full_transcript", "speaker_timestamp"))
        for r in reasons
    )
    if not transcriptish:
        return None

    start_page = _infer_transcript_start_page(pages, has_full_marker=has_full_marker)
    if has_summary and start_page > 0:
        return HybridPlan(
            kind="hybrid",
            transcript_start_page=start_page,
            reasons=tuple(reasons + [f"transcript_start_page:{start_page}"]),
        )
    if has_summary and has_full_marker and start_page == 0:
        # Marker on page 0 after summary bullets — still hybrid; transcript
        # portion begins at the marker (handled by text split), page gate = 0
        # means "rewrite turns on every page that carries speaker stamps".
        return HybridPlan(
            kind="hybrid",
            transcript_start_page=0,
            reasons=tuple(reasons + ["transcript_start_page:0_marker_on_summary_page"]),
        )
    # Pure transcript PDF/text (no summary half).
    return HybridPlan(
        kind="transcript_only",
        transcript_start_page=0,
        reasons=tuple(reasons + ["transcript_only"]),
    )


def _infer_transcript_start_page(
    pages: list[str],
    *,
    has_full_marker: bool,
) -> int:
    if not pages:
        return 0
    if has_full_marker:
        for idx, page in enumerate(pages):
            if _FULL_TRANSCRIPT_MARKER_RE.search(page or ""):
                return idx
    # First page with ≥2 speaker stamps, else first with ≥1 after page 0.
    for idx, page in enumerate(pages):
        hits = count_speaker_timestamp_hits(page or "")
        if hits >= 2:
            return idx
    for idx, page in enumerate(pages):
        if idx == 0:
            continue
        if count_speaker_timestamp_hits(page or "") >= 1:
            return idx
    return 0


def split_speaker_timestamp_turns(text: str) -> list[SpeakerTurn]:
    """Split a prose blob into diarized turns on ``Name [mm:ss]`` boundaries.

    Leading text before the first stamp (e.g. a section header) is emitted as
    a speaker-less turn so callers can drop chrome separately.
    """
    if not text or not text.strip():
        return []
    matches = list(_SPEAKER_TS_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return (
            [SpeakerTurn(speaker=None, timestamp=None, text=stripped, char_start=0, char_end=len(text))]
            if stripped
            else []
        )

    turns: list[SpeakerTurn] = []
    first = matches[0]
    lead = text[: first.start()].strip()
    if lead:
        turns.append(
            SpeakerTurn(
                speaker=None,
                timestamp=None,
                text=lead,
                char_start=0,
                char_end=first.start(),
            )
        )
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        # Keep empty bodies only when the stamp itself is chrome (caller drops).
        turns.append(
            SpeakerTurn(
                speaker=match.group("speaker").strip(),
                timestamp=match.group("ts"),
                text=body,
                char_start=match.start(),
                char_end=end,
            )
        )
    return turns


def stitch_cross_page_speaker_turns(
    page_turns: list[tuple[int, list[SpeakerTurn]]],
) -> list[StitchedSpeakerTurn]:
    """Merge speaker turns truncated at a PDF page break into one turn.

    Universal hybrid/transcript rule: when page N ends mid-utterance and
    page N+1 opens with speaker-less continuation text (no new
    ``Name [mm:ss]`` stamp), append that text to the prior turn. Do not
    emit two atoms for one spoken turn.

    ``page_turns`` must be in ascending page order. Returns a flat list
    preserving document order with page provenance.
    """
    flat: list[StitchedSpeakerTurn] = []
    for page_idx, turns in page_turns:
        for turn in turns:
            flat.append(StitchedSpeakerTurn(page_idx=page_idx, turn=turn, page_end=None))

    if not flat:
        return []

    out: list[StitchedSpeakerTurn] = []
    i = 0
    while i < len(flat):
        cur = flat[i]
        # Pull forward any following speaker-less page-lead continuations.
        while i + 1 < len(flat):
            nxt = flat[i + 1]
            if nxt.page_idx <= cur.page_idx:
                break
            # Continuation: next page opens without a new speaker stamp.
            if nxt.turn.speaker is not None or nxt.turn.timestamp is not None:
                break
            # Need a real prior speaker to attribute the continuation to.
            if not cur.turn.speaker:
                break
            cont = (nxt.turn.text or "").strip()
            if not cont:
                i += 1
                continue
            prior_body = (cur.turn.text or "").strip()
            merged = f"{prior_body} {cont}".strip() if prior_body else cont
            cur = StitchedSpeakerTurn(
                page_idx=cur.page_idx,
                turn=SpeakerTurn(
                    speaker=cur.turn.speaker,
                    timestamp=cur.turn.timestamp,
                    text=merged,
                    char_start=cur.turn.char_start,
                    char_end=nxt.turn.char_end,
                ),
                page_end=nxt.page_idx,
            )
            i += 1  # consume continuation
        out.append(cur)
        i += 1
    return out


def strip_transcript_section_chrome(text: str) -> str:
    """Remove sticky meeting-summary headers and inline ``Full Transcript`` labels.

    Text-rich PDF extracts often repeat the last summary heading (e.g.
    ``Key Decisions``) onto every subsequent transcript page, and glue
    ``Full Transcript`` onto the first speaker stamp. Both poison speaker
    splits and pollute atom text. Universal / structural only.
    """
    if not text:
        return ""
    # Drop whole-line section chrome first.
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not _MEETING_SECTION_CHROME_RE.match(ln)
    ]
    cleaned = "\n".join(lines)
    # Strip an inline ``Full Transcript`` prefix glued onto a speaker stamp.
    cleaned = _FULL_TRANSCRIPT_INLINE_RE.sub(
        lambda m: "\n" if ("\n" in m.group(0)) else "",
        cleaned,
    )
    return cleaned.strip()


_SOFT_SOCIAL_RE = re.compile(
    r"(?:"
    r"(?:i\s+don'?t\s+remember\s+what\s+that\s+was)|"
    r"(?:i\s+don'?t\s+(?:recall|remember)\b)|"
    r"(?:what\s+was\s+that\s+again)|"
    r"(?:we(?:'ll| will)\s+touch\s+on\s+this)|"
    r"(?:long\s+time\s+no\s+(?:talk|see|speak))"
    r")",
    re.IGNORECASE,
)


def _utterance_body(text: str) -> str:
    """Strip leading ``Name [mm:ss]`` chrome so token-length heuristics
    score the spoken words, not the speaker stamp."""
    probe = (text or "").strip()
    if not probe:
        return ""
    return _SPEAKER_TS_RE.sub("", probe, count=1).strip() or probe


def classify_transcript_turn_role(text: str) -> TurnRole:
    """Classify a single utterance as greeting/intro/logistics/filler/deal.

    Substance always wins: a turn that mentions install/configure/equipment
    etc. is ``deal`` even if it opens with a greeting.

    Short backchannel ("Good.", "Yeah.", "Thanks.") is ``acknowledgment`` —
    never deal substance, and always eligible for reply-to adjacency stamps.
    """
    # Classify the spoken body — speaker stamps inflate token counts and
    # flip soft social ("I don't remember…") into false ``deal``.
    probe = _utterance_body(text)
    if not probe:
        return "filler"
    if _DEAL_SUBSTANCE_RE.search(probe):
        return "deal"
    if _ACK_ONLY_RE.match(probe):
        return "acknowledgment"
    # Greeting wins over soft-social when both fire — e.g. "We'll touch on
    # this after the call… How are you?" is still a greeting turn for audit.
    if _GREETING_RE.search(probe):
        return "greeting"
    if _SOFT_SOCIAL_RE.search(probe):
        return "filler"
    if _INTRO_RE.search(probe):
        return "intro"
    if _LOGISTICS_RE.search(probe):
        return "logistics"
    # Closing thanks / bye without deal substance → filler (not scope).
    if _SIGNOFF_RE.search(probe):
        return "filler"
    # Short social turns without substance → filler
    tokens = [t for t in re.findall(r"[a-z0-9]+", probe.lower()) if t]
    if len(tokens) < 8:
        return "filler"
    return "deal"


def is_conversation_meta_atom(atom: Any) -> bool:
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return False
    return str(val.get("kind") or "") == CONVERSATION_META_KIND


def is_non_deal_meta_atom(atom: Any) -> bool:
    """True for communication chrome that must not feed neural/scope heads."""
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return False
    if str(val.get("kind") or "") in NON_DEAL_META_KINDS:
        return True
    # Belt-and-suspenders: explicit head-exclude flag on any atom.
    if val.get("non_deal") is True or val.get("head_exclude") is True:
        return True
    return False


def is_head_excluded_atom(atom: Any) -> bool:
    """True when neural/embedding/scope heads must skip this atom.

    Optimal model: keep conversation_meta auditable in the envelope, but
    never feed it (or any ``non_deal`` / ``head_exclude`` atom) into heads.
    """
    return is_non_deal_meta_atom(atom) or is_conversation_meta_atom(atom)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_page(atom: Any) -> int | None:
    refs = getattr(atom, "source_refs", None) or []
    if not refs:
        return None
    loc = getattr(refs[0], "locator", None) or {}
    if not isinstance(loc, dict):
        return None
    page = loc.get("page")
    if page is None:
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _atom_text(atom: Any) -> str:
    text = (
        getattr(atom, "raw_text", None)
        or getattr(atom, "normalized_text", None)
        or ""
    )
    if not text:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            text = str(val.get("text") or val.get("name") or "")
    return str(text).strip()


def page_texts_from_structured_doc(structured_doc: dict[str, Any] | None) -> list[str]:
    """Flatten structured PDF pages into plain text (for detection + rewrite)."""
    if not structured_doc:
        return []
    pages_out: list[str] = []
    for page in structured_doc.get("pages") or []:
        chunks: list[str] = []
        for section in page.get("sections") or []:
            heading = (section.get("heading") or "").strip()
            if heading:
                chunks.append(heading)
            for block in section.get("blocks") or []:
                kind = block.get("kind")
                if kind == "paragraph":
                    t = (block.get("text") or "").strip()
                    if t:
                        chunks.append(t)
                elif kind == "bullet_list":
                    intro = (block.get("intro") or "").strip()
                    if intro:
                        chunks.append(intro)
                    for item in block.get("items") or []:
                        if isinstance(item, dict):
                            it = (item.get("text") or "").strip()
                        else:
                            it = str(item or "").strip()
                        if it:
                            chunks.append(it)
                elif kind == "note":
                    t = (block.get("text") or "").strip()
                    if t:
                        chunks.append(t)
        # Do NOT fold page.metadata into rewrite text. Pipeline chrome like
        # "[text-rich page — …]" would glue onto turns, inflate token counts,
        # and flip filler/greeting into false "deal" classifications.
        pages_out.append("\n".join(chunks))
    return pages_out


def retag_conversational_to_meta(atoms: list[Any]) -> tuple[list[Any], int]:
    """Retype conversational ``scope_item`` turns to ``deal_metadata`` /
    ``kind=conversation_meta`` instead of dropping them.

    Keeps the envelope auditable while preventing scope/neural heads from
    treating greetings as deal facts. Returns (atoms, retag_count).
    """
    try:
        from app.core.schemas import AtomType
    except Exception:
        return atoms, 0

    retagged = 0
    for atom in atoms:
        if _atom_type_str(atom) not in {"scope_item", "entity", "note"}:
            continue
        if is_non_deal_meta_atom(atom):
            continue
        text = _atom_text(atom)
        # Bare speaker chrome ("Alex Rivera [00:48]") — retag as meta.
        if re.match(
            r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}\s*"
            r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*$",
            text,
        ):
            _apply_conversation_meta(atom, AtomType, role="chrome", text=text)
            retagged += 1
            continue
        role = classify_transcript_turn_role(text)
        if role == "deal":
            continue
        # Only retag when this looks like a transcript turn (speaker stamp,
        # page≥1 hybrid body, or explicit conversational role).
        page = _atom_page(atom)
        has_speaker = bool(_SPEAKER_TS_RE.search(text))
        if role in {"greeting", "intro", "logistics", "filler", "acknowledgment"} and (
            has_speaker or (page is not None and page >= 1) or role != "filler"
        ):
            # Avoid retagging short non-transcript scope fragments on page 0.
            if page == 0 and not has_speaker and role in {"filler", "acknowledgment"}:
                continue
            _apply_conversation_meta(atom, AtomType, role=role, text=text)
            retagged += 1
    # Stamp reply-to adjacency across transcript turns (universal).
    stamp_conversation_reply_adjacency(atoms)
    return atoms, retagged


def _apply_conversation_meta(atom: Any, AtomType: Any, *, role: str, text: str) -> None:
    atom.atom_type = AtomType.deal_metadata
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        val = {}
    val = dict(val)
    val["kind"] = CONVERSATION_META_KIND
    val["role"] = role
    val["text"] = text
    val["non_deal"] = True
    val["head_exclude"] = True
    atom.value = val
    flags = list(getattr(atom, "review_flags", None) or [])
    if "conversation_meta" not in flags:
        flags.append("conversation_meta")
    if "head_exclude" not in flags:
        flags.append("head_exclude")
    atom.review_flags = flags


def _truncate_reply_text(text: str, limit: int = _REPLY_TO_TEXT_MAX) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _turn_sort_key(atom: Any) -> tuple:
    page = _atom_page(atom)
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else {}
    if not isinstance(loc, dict):
        loc = {}
    utt = loc.get("utterance_index")
    try:
        utt_i = int(utt) if utt is not None else 10**9
    except (TypeError, ValueError):
        utt_i = 10**9
    ts = str(loc.get("timestamp_start") or "")
    return (page if page is not None else 10**9, utt_i, ts, getattr(atom, "id", "") or "")


def _prior_turn_snapshot(atom: Any) -> dict[str, Any]:
    """Compact prior-turn payload for in_reply_to / previous_* stamps."""
    val = getattr(atom, "value", None)
    val = val if isinstance(val, dict) else {}
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else {}
    if not isinstance(loc, dict):
        loc = {}
    speaker = val.get("speaker") or loc.get("speaker")
    timestamp = val.get("timestamp") or loc.get("timestamp_start")
    body = _utterance_body(_atom_text(atom)) or _atom_text(atom)
    snap: dict[str, Any] = {
        "speaker": speaker,
        "text": _truncate_reply_text(body),
        "timestamp": timestamp,
    }
    utt = loc.get("utterance_index")
    if utt is not None:
        snap["utterance_index"] = utt
    page = _atom_page(atom)
    if page is not None:
        snap["page"] = page
    role = val.get("role")
    if role:
        snap["role"] = role
    return snap


def stamp_conversation_reply_adjacency(atoms: list[Any]) -> int:
    """Stamp ``in_reply_to`` / ``previous_*`` on short acknowledgments.

    Universal: walks transcript turns in page + utterance order. Any
    ``acknowledgment`` (or short filler) that follows another speaker turn
    inherits that prior turn's speaker + text so heads/audit never see an
    orphan token like ``Good.`` with no conversational context.

    Returns the number of atoms stamped.
    """
    # Order all atoms that look like transcript turns.
    ordered = sorted(atoms, key=_turn_sort_key)
    stamped = 0
    prev: Any | None = None
    for atom in ordered:
        # Only consider turns with a page locator (transcript / hybrid).
        if _atom_page(atom) is None and not (
            isinstance(getattr(atom, "value", None), dict)
            and (atom.value.get("speaker") or atom.value.get("kind") == CONVERSATION_META_KIND)
        ):
            # Still advance prev for speaker-stamped text without page.
            text = _atom_text(atom)
            if _SPEAKER_TS_RE.search(text) or (
                isinstance(getattr(atom, "value", None), dict) and atom.value.get("speaker")
            ):
                prev = atom
            continue

        val = getattr(atom, "value", None)
        val = val if isinstance(val, dict) else {}
        role = str(val.get("role") or "")
        body = _utterance_body(_atom_text(atom))
        needs_reply = (
            role == "acknowledgment"
            or (role == "filler" and bool(_ACK_ONLY_RE.match(body or "")))
            or (
                is_conversation_meta_atom(atom)
                and bool(_ACK_ONLY_RE.match(body or ""))
            )
        )
        if needs_reply and prev is not None and prev is not atom:
            snap = _prior_turn_snapshot(prev)
            # Don't self-reply when prior is empty chrome.
            if snap.get("text") or snap.get("speaker"):
                new_val = dict(val)
                new_val["in_reply_to"] = snap
                new_val["previous_speaker"] = snap.get("speaker")
                new_val["previous_text"] = snap.get("text")
                new_val["previous_timestamp"] = snap.get("timestamp")
                # Promote bare filler ack → acknowledgment once context exists.
                if new_val.get("role") == "filler" and _ACK_ONLY_RE.match(body or ""):
                    new_val["role"] = "acknowledgment"
                new_val["non_deal"] = True
                new_val["head_exclude"] = True
                atom.value = new_val
                flags = list(getattr(atom, "review_flags", None) or [])
                if "reply_adjacency" not in flags:
                    flags.append("reply_adjacency")
                atom.review_flags = flags
                stamped += 1
        # Advance previous turn pointer for any real utterance.
        text = _atom_text(atom)
        if text.strip():
            prev = atom
    return stamped


def collapse_greeting_clusters(atoms: list[Any]) -> tuple[list[Any], int]:
    """Collapse greeting → acknowledgment adjacency into one ``greeting_cluster``.

    Conservative: only merges a clusterable social turn immediately followed
    by an ``acknowledgment`` on the same page (utterance_index + 1). Longer
    social runs stay as individual meta atoms (each with reply-to stamps).
    Returns (atoms, collapse_count).
    """
    if len(atoms) < 2:
        return atoms, 0

    # Index by (page, utterance) for O(1) adjacency lookup.
    by_key: dict[tuple[int, int], int] = {}
    for i, atom in enumerate(atoms):
        if not is_conversation_meta_atom(atom):
            continue
        page = _atom_page(atom)
        refs = getattr(atom, "source_refs", None) or []
        loc = getattr(refs[0], "locator", None) if refs else {}
        if not isinstance(loc, dict) or page is None:
            continue
        try:
            utt = int(loc["utterance_index"]) if loc.get("utterance_index") is not None else None
        except (TypeError, ValueError):
            utt = None
        if utt is None:
            continue
        by_key[(page, utt)] = i

    drop: set[int] = set()
    collapsed = 0
    for (page, utt), i in list(by_key.items()):
        if i in drop:
            continue
        j = by_key.get((page, utt + 1))
        if j is None or j in drop:
            continue
        a, b = atoms[i], atoms[j]
        va = getattr(a, "value", None) or {}
        vb = getattr(b, "value", None) or {}
        if not isinstance(va, dict) or not isinstance(vb, dict):
            continue
        role_a = str(va.get("role") or "")
        role_b = str(vb.get("role") or "")
        # Only greeting/intro/filler → acknowledgment pairs.
        if role_a not in {"greeting", "intro", "filler"}:
            continue
        if role_b != "acknowledgment":
            continue
        turns = []
        texts = []
        for atom, val in ((a, va), (b, vb)):
            body = _utterance_body(_atom_text(atom)) or _atom_text(atom)
            turn_rec: dict[str, Any] = {
                "speaker": val.get("speaker"),
                "timestamp": val.get("timestamp"),
                "role": val.get("role"),
                "text": body,
            }
            if val.get("in_reply_to"):
                turn_rec["in_reply_to"] = val["in_reply_to"]
            turns.append(turn_rec)
            texts.append(_atom_text(atom))
        head_val = dict(va)
        head_val["kind"] = CONVERSATION_META_KIND
        head_val["role"] = "greeting_cluster"
        head_val["turns"] = turns
        head_val["text"] = " | ".join(t for t in texts if t)
        head_val["non_deal"] = True
        head_val["head_exclude"] = True
        # The acknowledgment's reply-to is the point of the cluster.
        if vb.get("in_reply_to"):
            head_val["in_reply_to"] = vb["in_reply_to"]
            head_val["previous_speaker"] = vb.get("previous_speaker")
            head_val["previous_text"] = vb.get("previous_text")
            head_val["previous_timestamp"] = vb.get("previous_timestamp")
        a.value = head_val
        a.raw_text = head_val["text"]
        a.normalized_text = head_val["text"]
        flags = list(getattr(a, "review_flags", None) or [])
        for f in ("conversation_meta", "greeting_cluster", "head_exclude"):
            if f not in flags:
                flags.append(f)
        a.review_flags = flags
        drop.add(j)
        collapsed += 1

    if not drop:
        return atoms, 0
    return [a for i, a in enumerate(atoms) if i not in drop], collapsed


def rewrite_hybrid_pdf_atoms(
    *,
    atoms: list[Any],
    structured_doc: dict[str, Any] | None,
    filename: str,
    project_id: str,
    artifact_id: str,
    parser_version: str,
) -> list[Any]:
    """Replace poorly segmented transcript-page prose with per-turn atoms.

    Summary pages (before ``transcript_start_page``) are left untouched.
    Transcript-region ``scope_item`` / ``entity`` / ``note`` paragraphs are
    removed and rebuilt from speaker-timestamp splits. Typed atoms already
    carrying deal structure (quantity, decision, …) are kept.
    """
    pages = page_texts_from_structured_doc(structured_doc)
    title = None
    if structured_doc:
        title = (structured_doc.get("document") or {}).get("title")
    plan = detect_hybrid_summary_transcript(
        filename=filename,
        title=title,
        page_texts=pages,
    )
    if plan is None:
        return atoms

    try:
        from app.core.ids import stable_id
        from app.core.schemas import (
            ArtifactType,
            AtomType,
            AuthorityClass,
            EvidenceAtom,
            ReviewStatus,
            SourceRef,
        )
    except Exception:
        return atoms

    start = plan.transcript_start_page
    # Prose + weakly-typed atoms on transcript pages are rebuilt from speaker
    # turns. Keep strongly structured non-prose (BOM rows, sites, …).
    _REBUILD_TYPES = {
        "scope_item",
        "entity",
        "note",
        "assumption",
        "quantity",
        "risk",
        "open_question",
        "action_item",
        "decision",
        "meeting_commitment",
        "constraint",
        "exclusion",
        "customer_instruction",
    }
    keep: list[Any] = []
    for atom in atoms:
        page = _atom_page(atom)
        at = _atom_type_str(atom)
        # Always keep summary-page atoms.
        if page is None or page < start:
            keep.append(atom)
            continue
        if at not in _REBUILD_TYPES:
            keep.append(atom)
            continue
        # Drop transcript-region prose / weak types — rebuilt below.
        if plan.kind in {"hybrid", "transcript_only"}:
            continue
        text = _atom_text(atom)
        if count_speaker_timestamp_hits(text) >= 1:
            continue
        keep.append(atom)

    # Rebuild turns from structured page text for transcript pages.
    # Collect per-page splits first, then stitch cross-page continuations
    # so a turn cut at the page footer is one atom — not two.
    page_turn_bags: list[tuple[int, list[SpeakerTurn]]] = []
    for page_idx, page_text in enumerate(pages):
        if page_idx < start and plan.kind == "hybrid":
            # On the summary page that also hosts a Full Transcript marker,
            # only rewrite the transcript portion after the marker.
            if not (_FULL_TRANSCRIPT_MARKER_RE.search(page_text or "") and page_idx == start):
                continue
        turns_text = page_text or ""
        if plan.kind == "hybrid" and page_idx == start:
            m = _FULL_TRANSCRIPT_MARKER_RE.search(turns_text)
            if m:
                # Advance past the marker itself — do not keep
                # ``Full Transcript`` glued onto the first speaker stamp.
                turns_text = turns_text[m.end() :]
        turns_text = strip_transcript_section_chrome(turns_text)
        turns = split_speaker_timestamp_turns(turns_text)
        if not turns and count_speaker_timestamp_hits(turns_text) == 0:
            continue
        page_turn_bags.append((page_idx, turns))

    stitched = stitch_cross_page_speaker_turns(page_turn_bags)
    new_atoms: list[EvidenceAtom] = []
    for t_idx, st in enumerate(stitched):
        turn = st.turn
        page_idx = st.page_idx
        body = (turn.text or "").strip()
        # Section chrome ("Full Transcript: …") with no body.
        label = ""
        if turn.speaker and turn.timestamp:
            # Reject speakers that absorbed section chrome / newlines.
            if "\n" in (turn.speaker or "") or _MEETING_SECTION_CHROME_RE.match(
                (turn.speaker or "").strip()
            ):
                continue
            label = f"{turn.speaker} [{turn.timestamp}]"
        display = f"{label} {body}".strip() if body else label
        if not display:
            continue
        # Drop pure section headers / sticky chrome leftovers.
        if re.match(
            r"^(?:full\s+transcripts?(?:\s+and\s+summary)?|executive\s+summary|"
            r"meeting\s+summary(?:\s+and\s+full\s+transcripts?)?|"
            r"key\s+decisions?|action\s+items?|open\s+questions?|"
            r"attendees?|participants?|next\s+steps?|agenda|discussion|"
            r"notes|follow[\s-]?ups?)\s*:?\s*$",
            display,
            re.I,
        ):
            continue
        # Speaker-less lead that is only leftover chrome crumbs ("Key").
        if not turn.speaker and len(re.findall(r"[a-z0-9]+", display.lower())) <= 2:
            if not _SPEAKER_TS_RE.search(display):
                continue
        role = classify_transcript_turn_role(body or display)
        locator: dict[str, Any] = {
            "page": page_idx,
            "block_kind": "transcript_turn",
            "speaker": turn.speaker,
            "timestamp_start": turn.timestamp,
            "utterance_index": t_idx,
            "hybrid_plan": plan.kind,
        }
        if st.page_end is not None and st.page_end != page_idx:
            locator["page_end"] = st.page_end
            locator["cross_page_stitched"] = True
        source = SourceRef(
            id=stable_id(
                "src",
                artifact_id,
                page_idx,
                t_idx,
                turn.speaker or "",
                turn.timestamp or "",
                body[:80],
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator=locator,
            extraction_method="hybrid_summary_transcript_v1",
            parser_version=parser_version,
        )
        if role != "deal":
            new_atoms.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm",
                        project_id,
                        artifact_id,
                        "conversation_meta",
                        page_idx,
                        t_idx,
                        display[:120],
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.deal_metadata,
                    raw_text=display,
                    normalized_text=display.strip(),
                    value={
                        "kind": CONVERSATION_META_KIND,
                        "role": role,
                        "text": display,
                        "non_deal": True,
                        "head_exclude": True,
                        "speaker": turn.speaker,
                        "timestamp": turn.timestamp,
                    },
                    entity_keys=[],
                    source_refs=[source],
                    authority_class=AuthorityClass.meeting_note,
                    confidence=0.7,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[
                        "conversation_meta",
                        "head_exclude",
                        f"turn_role:{role}",
                    ],
                    parser_version=parser_version,
                )
            )
            continue
        # Deal turn — emit as meeting_note scope with transcript provenance;
        # the transcript rule engine (when available) further types it.
        typed_atoms = _emit_deal_turn_atoms(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            display=display,
            body=body or display,
            speaker=turn.speaker,
            timestamp=turn.timestamp,
            page_idx=page_idx,
            utterance_index=t_idx,
            source=source,
            AtomType=AtomType,
            EvidenceAtom=EvidenceAtom,
            AuthorityClass=AuthorityClass,
            ReviewStatus=ReviewStatus,
            stable_id=stable_id,
        )
        new_atoms.extend(typed_atoms)

    # Reply-to adjacency: "Good." must carry what it answers.
    # Keep individual meta atoms (audit-friendly); collapse is available
    # via collapse_greeting_clusters for callers that want denser chrome.
    combined = keep + new_atoms
    stamp_conversation_reply_adjacency(combined)
    return combined


def _emit_deal_turn_atoms(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    display: str,
    body: str,
    speaker: str | None,
    timestamp: str | None,
    page_idx: int,
    utterance_index: int,
    source: Any,
    AtomType: Any,
    EvidenceAtom: Any,
    AuthorityClass: Any,
    ReviewStatus: Any,
    stable_id: Any,
) -> list[Any]:
    """Type a substantive transcript turn via TranscriptParser rules when possible."""
    try:
        from app.parsers.transcript_parser import TranscriptParser

        parser = TranscriptParser()
        segment = {
            "utterance_index": utterance_index,
            "line_start": utterance_index + 1,
            "line_end": utterance_index + 1,
            "speaker": speaker,
            "timestamp_start": timestamp,
            "timestamp_end": None,
            "section": None,
            "text": body,
        }
        atoms = parser._atoms_from_segment(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            segment=segment,
        )
        # Stamp PDF page onto source locators so substance gates / UI keep page.
        for atom in atoms:
            for ref in getattr(atom, "source_refs", None) or []:
                loc = getattr(ref, "locator", None)
                if isinstance(loc, dict):
                    loc["page"] = page_idx
                    loc["block_kind"] = "transcript_turn"
                    loc["hybrid_plan"] = "rewritten"
            # Prefer PDF artifact type for hybrid PDF source.
            try:
                from app.core.schemas import ArtifactType

                for ref in getattr(atom, "source_refs", None) or []:
                    ref.artifact_type = ArtifactType.pdf
                    ref.extraction_method = "hybrid_summary_transcript_v1"
            except Exception:
                pass
        if atoms:
            return _collapse_hybrid_turn_atoms(list(atoms))
    except Exception:
        pass

    return [
        EvidenceAtom(
            id=stable_id(
                "atm",
                project_id,
                artifact_id,
                "transcript_deal_turn",
                page_idx,
                utterance_index,
                body[:120],
            ),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.scope_item,
            raw_text=display,
            normalized_text=display.strip(),
            value={
                "kind": "transcript_turn",
                "text": body,
                "speaker": speaker,
                "timestamp": timestamp,
            },
            entity_keys=[],
            source_refs=[source],
            authority_class=AuthorityClass.meeting_note,
            confidence=0.76,
            review_status=ReviewStatus.needs_review,
            review_flags=["hybrid_transcript_turn"],
            parser_version=parser_version,
        )
    ]


# One primary typed atom per utterance (plus site entities). TranscriptParser
# often fires action_item+open_question (or decision+meeting_commitment) on the
# same short turn — that duplicates audit rows and pollutes heads.
_HYBRID_TYPE_RANK: dict[str, int] = {
    "physical_site": 0,  # always kept separately
    "decision": 10,
    "action_item": 20,
    "customer_instruction": 30,
    "constraint": 40,
    "exclusion": 50,
    "quantity": 60,
    "risk": 70,
    "scope_item": 80,
    "meeting_commitment": 90,
    "open_question": 100,
}


def _collapse_hybrid_turn_atoms(atoms: list[Any]) -> list[Any]:
    """Keep physical_site atoms + a single best typed atom per raw_text."""
    if len(atoms) <= 1:
        return atoms
    sites: list[Any] = []
    by_text: dict[str, list[Any]] = {}
    for atom in atoms:
        at = _atom_type_str(atom)
        if at == "physical_site":
            sites.append(atom)
            continue
        key = (getattr(atom, "normalized_text", None) or getattr(atom, "raw_text", None) or "").strip().lower()
        by_text.setdefault(key, []).append(atom)
    primary: list[Any] = []
    for group in by_text.values():
        if len(group) == 1:
            primary.append(group[0])
            continue
        group_sorted = sorted(
            group,
            key=lambda a: (
                _HYBRID_TYPE_RANK.get(_atom_type_str(a), 500),
                -float(getattr(a, "confidence", 0) or 0),
            ),
        )
        primary.append(group_sorted[0])
    return sites + primary


__all__ = [
    "CONVERSATION_META_KIND",
    "CONVERSATION_META_ROLES",
    "NON_DEAL_META_KINDS",
    "HybridPlan",
    "SpeakerTurn",
    "classify_transcript_turn_role",
    "collapse_greeting_clusters",
    "count_speaker_timestamp_hits",
    "detect_hybrid_summary_transcript",
    "is_conversation_meta_atom",
    "is_head_excluded_atom",
    "is_non_deal_meta_atom",
    "looks_like_summary_filename",
    "looks_like_transcript_filename",
    "page_texts_from_structured_doc",
    "retag_conversational_to_meta",
    "rewrite_hybrid_pdf_atoms",
    "split_speaker_timestamp_turns",
    "stamp_conversation_reply_adjacency",
    "stitch_cross_page_speaker_turns",
    "strip_transcript_section_chrome",
]
