from __future__ import annotations

from datetime import datetime, timezone

from app.core.textio import decode_html_entities, read_text

import json
import re
from pathlib import Path
from typing import Any

from app.domain import get_active_domain_pack
from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.ids import stable_id
from app.core.normalizers import (
    detect_speaker,
    detect_section,
    extract_meeting_entities,
    fold_standalone_speaker_lines,
    normalize_text,
    normalize_transcript_text,
    parse_timestamp,
    split_transcript_segments,
)
from app.core.segments import ArtifactSegment
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserOutput,
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.segmenters import segment_transcript
from app.parsers.structured_projection import (
    derived_files_for,
    make_bullet_list,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    stamp_section_and_block_ids,
)
from app.domain.schemas import DomainPack

STRUCTURED_SCHEMA_TRANSCRIPT = "orbitbrief.transcript.structured.v1"

#: W3C voice span, e.g. ``<v.loud Cliff Creech>text</v>``. Only the
#: dependency-free fallback needs this -- webvtt-py handles it natively.
_VTT_VOICE_SPAN_RE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>(.*?)(?:</v>|$)", re.I | re.S)

DECISION_RE = re.compile(
    r"\b(decision:|decided|agreed|confirmed|approved|we will|the plan is|final decision)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(action item|ai:|todo|owner:|customer to|purtera to|customer will|purtera will|follow up|send|confirm|provide)\b",
    re.I,
)
QUESTION_RE = re.compile(r"\?|open question|tbd|need to confirm|confirm whether|unknown|pending", re.I)
CONSTRAINT_RE = re.compile(
    r"\b(access window|escort required|escort access|badge required|loading dock|parking|after hours|weekdays|weekends|site access|security requirement|staging|approval gate)\b",
    re.I,
)
EXCLUSION_RE = re.compile(
    r"\b(exclude|excluded|removed from scope|out of scope|not in scope|do not include|customer will not proceed with)\b",
    re.I,
)
SCOPE_RE = re.compile(
    r"\b(install|deploy|replace|remove|survey|rack|configure|camera|ap|switch|reader|device|rollout)\b",
    re.I,
)
CUSTOMER_DIRECTIVE_RE = re.compile(
    r"\b(please remove|please add|we approve|do not proceed|hold off|go ahead)\b",
    re.I,
)
QUANTITY_RE = re.compile(
    r"\b(add|remove|reduce to|set to|additionally add|may add)?\s*(\d+)\s*(more\s+)?(ip cameras?|cameras?|aps?|access points?|devices?)\b",
    re.I,
)

SCOPE_IMPACTING_TYPES = {
    AtomType.scope_item,
    AtomType.exclusion,
    AtomType.customer_instruction,
    AtomType.decision,
    AtomType.meeting_commitment,
    AtomType.quantity,
}



# ---------------------------------------------------------------------------
# Re-joining sentences the diariser cut in two.
#
# Fireflies stores a turn as several cues and punctuates each one, so a single
# spoken sentence can arrive as "I need to see the." followed, half a second
# later and from the same speaker, by "I guess the locations, because it should
# be doable." One atom per cue then gives the brief a dangling half sentence
# ("Do you do per.", "And we'll do.") and a second atom that has lost its
# subject. Whether two consecutive cues of the SAME speaker are one sentence is
# read from how complete each side is:
#   - the earlier cue stops on a word no sentence ends on ("the", "per", "my",
#     "gonna"), or a short cue's last content word opens the next one;
#   - the later cue continues the earlier one: it opens on a preposition ("...
#     until next week." / "On what's actually at the sites.") or is a phrase
#     with no verb that picks up a word of the earlier one ("... a lot of those
#     for start of the year." / "A lot of site directors.");
#   - a verbless head runs into another fragment ("Like texts." / "Do you do per.");
#   - a short cut cue that nothing of the speaker's continues ("Is it per
#     school?" / "How do you kind of." and the other speaker answers) closes
#     the sentence before it.
# Another speaker's back-channel ("Okay.") between the two halves is skipped:
# it stays its own atom, in order. Two complete sentences ("Is that doable?" /
# "Is that too tight?"), a restart ("How are we gonna." / "How are we
# thinking?") and two different speakers are never joined.
# ---------------------------------------------------------------------------

#: Words a sentence does not end on. A cue that stops on one was cut.
_CUT_ARTICLES = frozenset({"a", "an", "the"})
_CUT_DETERMINERS = frozenset({"my", "our", "your", "their", "his", "her", "its", "every", "each"})
_CUT_PREPOSITIONS = frozenset({"of", "per", "into", "onto", "between", "among", "toward", "towards", "via", "versus", "than"})
_CUT_CONJUNCTIONS = frozenset({"and", "or", "but", "because", "if", "nor"})
_CUT_PRONOUNS = frozenset({"i", "we", "they", "he", "she"})
#: A word stub the diariser cut before its stem ("I'm gonna re." / "Quote.",
#: "before 1 o'." / "Clock.").
_CUT_PREFIXES = frozenset({"re", "pre", "un", "non", "co"})
#: An auxiliary right after its subject ends an elliptical clause ("We do.",
#: "she will.", "where it is.") unless a subordinator opens it ("so I can.").
_ELLIPSIS_SUBJECTS = frozenset({"i", "we", "you", "they", "he", "she", "it"})
#: Auxiliaries end a cut ("And we'll do.") but also a short answer ("Yes, we
#: do."), so they count only when the cue does not open as an answer.
_CUT_AUXILIARIES = frozenset({
    "do", "does", "did", "can", "could", "will", "would", "should", "shall",
    "must", "might", "may", "be", "am", "is", "are", "was", "were",
    "have", "has", "had", "gonna", "wanna", "gotta",
})
_ANSWER_OPENERS = frozenset({
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "sure", "ok", "okay",
    "right", "absolutely", "definitely", "correct", "exactly", "of",
})
#: Function words a speaker repeats when restarting ("So that's." / "That's
#: kind of what." / "What I figured ..."); echoing one is a restart, not a cut.
_ECHO_STOP = frozenset({
    "that", "this", "what", "which", "who", "so", "well", "yeah", "yes", "no",
    "okay", "ok", "right", "it", "now", "then", "there", "here", "just", "like",
    "you", "me", "us", "them", "um", "uh",
}) | _CUT_ARTICLES | _CUT_DETERMINERS | _CUT_PREPOSITIONS | _CUT_CONJUNCTIONS | _CUT_PRONOUNS | _CUT_AUXILIARIES
_ECHO_MAX_WORDS = 8
_JOIN_MAX_CUES = 6
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

#: Prepositions that open a phrase continuing the sentence before it.
_CONT_PREPOSITIONS = frozenset({
    "on", "at", "with", "for", "from", "about", "to", "in", "into", "of", "by",
    "around", "until", "including", "regarding", "without", "within", "across",
    "through", "during", "per",
})
#: Set phrases that stand alone although they open on a preposition.
_CONT_IDIOMS = frozenset({
    ("of", "course"), ("for", "sure"), ("for", "example"), ("in", "fact"),
    ("by", "the"), ("at", "all"), ("in", "general"),
})
_CONT_MAX_WORDS = 7
#: A phrase holding a subject pronoun is a clause of its own.
_SUBJECTS = frozenset({"i", "we", "you", "they", "he", "she"})
#: Lexical verbs common in calls. With the auxiliaries, a subject pronoun, a
#: past form or a verb contraction they mark a cue as a clause; a cue with
#: none of them ("The other sites as well.") is a verbless phrase.
_VERBS = frozenset({
    "get", "gets", "got", "go", "goes", "went", "make", "makes", "made", "know",
    "knows", "knew", "think", "thinks", "thought", "need", "needs", "want",
    "wants", "see", "sees", "saw", "look", "looks", "sound", "sounds", "work",
    "works", "take", "takes", "took", "send", "sends", "sent", "say", "says",
    "said", "come", "comes", "came", "mean", "means", "meant", "give", "gives",
    "gave", "put", "puts", "let", "lets", "start", "starts", "guess", "believe",
    "figure", "feel", "feels", "seem", "seems", "tell", "tells", "told", "ask",
    "asks", "call", "try", "use", "keep", "change", "add", "schedule", "plan",
    "show", "shows", "find", "run", "runs", "pick", "set", "bring", "check",
    "understand", "agree", "thank", "thanks", "hope", "please",
})
_CONTRACTED_SUBJECTS = frozenset({"it", "that", "there", "what", "here", "he", "she", "who", "where", "how", "let"})
#: A cut cue ending on an auxiliary is an elliptical answer ("I believe we
#: do.") unless it is a subordinate clause with no main one ("Just so I can.").
_SUBORDINATORS = frozenset({"so", "because", "if", "when", "unless", "until", "since"})
_TRAILING_OPENERS = frozenset({"just", "like", "as", "which", "that", "where", "and", "or"})
_TAIL_MAX_WORDS = 6
_HEAD_MAX_WORDS = 4
#: Back-channel words. A cue of only these from another speaker is an
#: interjection the speaker talks over ("Next Friday ..." / "Okay." / "At the latest.").
_BACKCHANNEL = frozenset({
    "okay", "ok", "yeah", "yes", "yep", "yup", "right", "mhm", "mm", "hmm", "uh",
    "huh", "um", "sure", "cool", "great", "got", "it", "gotcha", "alright", "nice",
    "perfect", "oh", "i", "see", "true", "exactly", "correct", "good", "thanks",
    "thank", "you", "awesome", "wow", "all", "definitely", "absolutely", "hey",
    "hi", "hello", "bye",
})
_INTERJECTION_MAX_WORDS = 3
#: The earlier cue's length in words stands in for its duration (cues carry
#: only a start). Evidence on one side allows a short pause after it, on both
#: sides a long one; nothing joins past the ceiling.
_PAUSE_ONE_SIDE = 4.0
_PAUSE_BOTH_SIDES = 10.0
_PAUSE_CEILING = 15.0


def _cue_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _lower_words(text: str) -> list[str]:
    return [w.lower() for w in _cue_words(text)]


def _cue_ends_cut(text: str) -> bool:
    """True when ``text`` stops on a word no sentence ends on."""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in "?!":
        return False
    words = _cue_words(stripped)
    if not words:
        return False
    last = words[-1].lower()
    if last in _CUT_PREFIXES or (len(last) <= 2 and last.endswith("'")):
        return True
    if last in _CUT_ARTICLES or last in _CUT_DETERMINERS or last in _CUT_PREPOSITIONS:
        return True
    if last in _CUT_CONJUNCTIONS or last in _CUT_PRONOUNS:
        return True
    if last in _CUT_AUXILIARIES:
        lower = [w.lower() for w in words]
        if (
            last not in ("gonna", "wanna", "gotta")
            and len(lower) >= 2
            and lower[-2] in _ELLIPSIS_SUBJECTS
            and not (len(lower) >= 3 and lower[-3] in _SUBORDINATORS)
        ):
            return False
        return lower[0] not in _ANSWER_OPENERS
    return False


def _cue_echoes(text: str, nxt: str) -> bool:
    """A short cue whose last content word opens the next cue: the diariser
    cut on that word and repeated it ("... about three." / "Three to four")."""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in "?!":
        return False
    words, nwords = _cue_words(stripped), _cue_words(nxt)
    if not words or not nwords or len(words) > _ECHO_MAX_WORDS:
        return False
    last, first = words[-1], nwords[0]
    if "'" in last or last.lower() in _ECHO_STOP:
        return False
    # A capitalised word mid-cue is a name; a sentence may well open on it.
    if len(words) > 1 and last[:1].isupper():
        return False
    return last.lower() == first.lower()


def _is_backchannel(text: str) -> bool:
    words = _lower_words(text)
    return 0 < len(words) <= _INTERJECTION_MAX_WORDS and all(w in _BACKCHANNEL for w in words)


def _has_clause(words: list[str]) -> bool:
    for w in words:
        if w in _SUBJECTS or w in _CUT_AUXILIARIES or w in _VERBS or w.endswith("n't"):
            return True
        if len(w) > 4 and w.endswith("ed"):  # "changed", "finalized"
            return True
        if "'" in w:
            stem, _, suffix = w.partition("'")
            if suffix in ("re", "ll", "ve", "d", "m") or (suffix == "s" and stem in _CONTRACTED_SUBJECTS):
                return True
    return False


def _is_verbless(text: str) -> bool:
    """A phrase with no verb ("The other sites as well.", "Like texts.")."""
    words = _lower_words(text)
    return bool(words) and not _has_clause(words) and not _is_backchannel(text)


def _continues(text: str) -> bool:
    """The cue carries on the sentence before it rather than starting one."""
    words = _lower_words(text)
    if not words or _is_backchannel(text):
        return False
    if (
        words[0] in _CONT_PREPOSITIONS
        and len(words) <= _CONT_MAX_WORDS
        and tuple(words[:2]) not in _CONT_IDIOMS
        and not any(w in _SUBJECTS for w in words)
    ):
        # "In the meantime, review it." fronts a new clause; "With numbers,
        # just so they know." trails the sentence before.
        _, comma, after = text.partition(",")
        rest = _lower_words(after)
        return not comma or not rest or rest[0] in _SUBORDINATORS or rest[0] in _TRAILING_OPENERS
    return False


def _content_stems(text: str) -> set[str]:
    return {w.rstrip("s") for w in _lower_words(text) if len(w) >= 3 and w not in _ECHO_STOP and "'" not in w}


def _restates(text: str, nxt: str) -> bool:
    """A verbless cue that picks up a content word of the sentence before it
    elaborates on it ("... a lot of those for the year." / "A lot of site
    directors."); one that shares none ("Talk soon.", "Next item.") is its own."""
    return bool(_content_stems(text) & _content_stems(nxt))


def _is_fragment(text: str) -> bool:
    """The cue is not a sentence on its own."""
    return _cue_ends_cut(text) or _is_verbless(text) or _continues(text)


def _is_restart(text: str, nxt: str) -> bool:
    """The next cue starts the same sentence over ("How are we gonna." / "How are we thinking?")."""
    a, b = _lower_words(text), _lower_words(nxt)
    return len(a) >= 3 and len(b) >= 2 and a[:2] == b[:2]


def _finishes_restart(text: str, nxt: str) -> bool:
    """The next cue restates the cut restart's 2-3 word head and finishes it
    ("How are we gonna." / "How are we thinking?"): the restart stopped on an
    auxiliary with no verb after it, so the two are one sentence."""
    a, b = _lower_words(text), _lower_words(nxt)
    if not a or a[-1] not in _CUT_AUXILIARIES or not _cue_ends_cut(text):
        return False
    head = len(a) - 1
    return 2 <= head <= 3 and len(b) > head and a[:head] == b[:head] and not _cue_ends_cut(nxt)


#: A hedge verb whose subject the speaker dropped ("Think there's a cap.")
#: and the clause openers it hedges.
_HEDGE_VERBS = frozenset({"think", "guess"})
_HEDGE_CLAUSES = frozenset({"there's", "it's", "that's", "there", "it", "we", "they"})


def _hedge_restarts(text: str, nxt: str) -> bool:
    """A short hedge with a dropped subject ("Think there's a cap.") whose
    clause the next cue opens again and carries on ("There's a few sites
    that ..."): the hedge heads that sentence."""
    a, b = _lower_words(text), _lower_words(nxt)
    return (
        3 <= len(a) <= _HEAD_MAX_WORDS
        and a[0] in _HEDGE_VERBS
        and a[1] in _HEDGE_CLAUSES
        and text[:1].isupper()
        and len(b) > len(a)
        and a[1] == b[0]
        and not nxt.rstrip().endswith("?")
    )


def _is_open_tail(text: str) -> bool:
    """A short cut cue that may close the sentence before it."""
    words = _lower_words(text)
    if not _cue_ends_cut(text) or len(words) > _TAIL_MAX_WORDS:
        return False
    if words[-1] in _CUT_AUXILIARIES and words[-1] not in ("gonna", "wanna", "gotta"):
        return any(w in _SUBORDINATORS for w in words[:2])
    return True


def _as_seconds(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _within_pause(prev: dict[str, Any], nxt: dict[str, Any], *, both_sides: bool) -> bool:
    """The next cue starts soon enough after the earlier one to continue it.
    Missing times leave the words to decide."""
    a, b = _as_seconds(prev.get("timestamp_start")), _as_seconds(nxt.get("timestamp_start"))
    if a is None or b is None:
        return True
    spoken = 1.5 + 0.6 * len(_cue_words(str(prev.get("text", ""))))
    pause = _PAUSE_BOTH_SIDES if both_sides else _PAUSE_ONE_SIDE
    return 0 <= b - a <= min(spoken + pause, _PAUSE_CEILING)


def _resumes(text: str, nxt: str) -> bool:
    """The next cue picks up the auxiliary the cue stopped on ("I can." /
    "We can ramp up ..."): the clause was cut, not elliptical."""
    a, b = _lower_words(text), _lower_words(nxt)
    return bool(a) and a[-1] in _CUT_AUXILIARIES and a[-1] in b[:2]


_ASIDE_OPENERS = frozenset({"yeah", "yes", "yep", "yup", "okay", "ok", "oh", "right", "sure", "um", "uh", "hmm", "mhm"})


def _opens_aside(text: str) -> bool:
    """The cue opens on a response word ("Yeah, just do it."): a new turn."""
    words = _lower_words(text)
    return bool(words) and words[0] in _ASIDE_OPENERS


def _continues_cue(
    prev: dict[str, Any], seg: dict[str, Any], after: dict[str, Any] | None, *, prev_alone: bool = True
) -> bool:
    """``seg`` belongs to the sentence ``prev`` (same speaker) is part of.
    ``after`` is the speaker's next cue reachable from ``seg``, if any;
    ``prev_alone`` is False when ``prev`` already closes a joined sentence."""
    p, s = str(prev.get("text", "")), str(seg.get("text", ""))
    if _is_backchannel(p) or _is_backchannel(s):
        return False
    s_question = s.rstrip().endswith("?")
    prev_open = (_cue_ends_cut(p) or _resumes(p, s)) and not _is_restart(p, s)
    # "..., too, but." / "Any questions?": the speaker trailed off and asks anew.
    if prev_open and _lower_words(p)[-1] in _CUT_CONJUNCTIONS and s_question:
        prev_open = False
    if prev_open or _cue_echoes(p, s):
        return _within_pause(prev, seg, both_sides=_is_fragment(s))
    if _finishes_restart(p, s) or (prev_alone and _hedge_restarts(p, s)):
        return _within_pause(prev, seg, both_sides=False)
    if _continues(s) or (
        _is_verbless(s) and not p.rstrip().endswith("?") and not s_question
        and not _opens_aside(s) and _restates(p, s)
    ):
        return _within_pause(prev, seg, both_sides=_is_verbless(p))
    # A verbless head ("Like texts.") runs only into a short cut cue ("Do you
    # do per."), and only when it opens the sentence.
    if (
        prev_alone
        and _is_verbless(p)
        and not p.rstrip().endswith("?")
        and len(_cue_words(p)) <= _HEAD_MAX_WORDS
        and _cue_ends_cut(s)
        and len(_cue_words(s)) <= _TAIL_MAX_WORDS
    ):
        return _within_pause(prev, seg, both_sides=True)
    # A short cut cue nothing of the speaker's picks up closes the sentence
    # before it. One the speaker restarts ("What was the." / "What was the
    # last day ...?") heads the next sentence instead.
    if _is_open_tail(s):
        if after is None or not _within_pause(seg, after, both_sides=True):
            return _within_pause(prev, seg, both_sides=False)
    return False


def join_cut_fragments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive same-speaker cues that are one cut sentence.

    The joined segment keeps the first cue's ``utterance_index``, start time
    and ``line_start``; ``line_end`` reaches the last cue, and
    ``joined_utterance_indexes`` lists every cue it holds. Each cue's text is
    kept verbatim, separated by a space. An interjection another speaker made
    between two joined cues stays its own segment, after the joined one, and
    its index is not in ``joined_utterance_indexes``.
    """

    def speaker_next(i: int) -> dict[str, Any] | None:
        """The speaker's next cue after ``segments[i]``, past one interjection."""
        speaker = segments[i].get("speaker")
        for j in (i + 1, i + 2):
            if j >= len(segments):
                return None
            if segments[j].get("speaker") == speaker:
                return segments[j]
            if j == i + 2 or not _is_backchannel(str(segments[j].get("text", ""))):
                return None
        return None

    out: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []  # the last cue folded into each out[k]
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker")
        target = None
        if speaker and out:
            if out[-1].get("speaker") == speaker:
                target = len(out) - 1
            elif (
                len(out) >= 2
                and out[-2].get("speaker") == speaker
                and "joined_utterance_indexes" not in out[-1]
                and _is_backchannel(str(out[-1].get("text", "")))
                # after a question the other speaker's word is an answer
                and not str(tails[-2].get("text", "")).rstrip().endswith("?")
            ):
                target = len(out) - 2
        if target is not None:
            cur = out[target]
            if (
                cur.get("section") == seg.get("section")
                and len(cur.get("joined_utterance_indexes") or [cur["utterance_index"]]) < _JOIN_MAX_CUES
                and _continues_cue(
                    tails[target], seg, speaker_next(i),
                    prev_alone="joined_utterance_indexes" not in cur,
                )
            ):
                joined = list(cur.get("joined_utterance_indexes") or [cur["utterance_index"]])
                joined.append(seg["utterance_index"])
                cur["joined_utterance_indexes"] = joined
                cur["text"] = f"{cur['text']} {seg['text']}"
                cur["line_end"] = seg["line_end"]
                if seg.get("timestamp_end") is not None:
                    cur["timestamp_end"] = seg["timestamp_end"]
                tails[target] = seg
                continue
        out.append(dict(seg))
        tails.append(seg)
    return out


_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _ulid_iso(ulid: str) -> str | None:
    """The creation time a ULID encodes, as ISO-8601 UTC, when the id is one."""
    u = (ulid or "").strip().upper()
    if not _ULID_RE.match(u):
        return None
    ms = 0
    for ch in u[:10]:
        ms = ms * 32 + _ULID_ALPHABET.index(ch)
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if not (2000 <= dt.year <= 2100):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_date(value: Any) -> str | None:
    """A JSON date field (ISO string or epoch ms) as ISO-8601 UTC, or None."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            ms = float(value)
            dt = datetime.fromtimestamp(ms / 1000 if ms > 1e11 else ms, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        text = str(value).strip()
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

_EXPORT_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z -]{0,30}):\s(.*)$")


def _looks_like_value(text: str) -> bool:
    """An id, a timestamp, a URL, a one-word tag: a field, not a sentence."""
    t = text.strip()
    return bool(t) and (" " not in t or t.startswith(("http://", "https://")))


def peel_export_header(raw_text: str) -> tuple[dict[str, str] | None, str]:
    """Split an exporter's header block off the top of a plain-text transcript.

    A CRM meeting export opens with a run of ``Label: value`` lines before its
    first blank line -- the meeting's title, its id, when it ran, where the
    recording is. Read as speech, every one of those is a "speaker" saying
    something, and the first became a task on live 010095 ("HubSpot Meeting:
    010095 Lantronix device installation and setup(with Orcle)" -- the deal's
    name, offered to the Deal Kit as work nobody asked for).

    The block is recognised by its shape, not its labels: at least three
    consecutive ``Label: value`` lines from the top, terminated by a blank
    line, with every label distinct and at least half of the values being
    fields rather than sentences (an id, a timestamp, a URL, a single word).
    Speech never looks like that: speakers repeat and utterances are prose.

    Returns ``(fields, body, offset)``: ``fields`` keeps insertion order with
    the first entry the title line; ``body`` is the text from its first spoken
    line on; ``offset`` is how many lines were peeled before it, so the caller
    can give every segment its number in the original file. (The header is not
    blanked in place because the transcript normaliser strips leading blank
    lines, which would renumber the body anyway.)
    """
    lines = raw_text.splitlines()
    fields: dict[str, str] = {}
    end = 0
    for line in lines:
        if not line.strip():
            break
        m = _EXPORT_FIELD_RE.match(line.strip())
        if not m or m.group(1).strip() in fields:
            return None, raw_text, 0
        fields[m.group(1).strip()] = m.group(2).strip()
        end += 1
    if end < 3 or sum(_looks_like_value(v) for v in fields.values()) * 2 < len(fields):
        return None, raw_text, 0
    first_body = end
    while first_body < len(lines) and not lines[first_body].strip():
        first_body += 1
    if first_body >= len(lines):
        return None, raw_text, 0
    return fields, "\n".join(lines[first_body:]), first_body


class TranscriptParser(BaseParser):
    parser_name = "transcript"
    parser_version = "transcript_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".txt", ".md", ".vtt", ".srt", ".json"],
        supported_artifact_types=[ArtifactType.transcript, ArtifactType.txt],
        emitted_atom_types=[
            AtomType.decision,
            AtomType.meeting_commitment,
            AtomType.action_item,
            AtomType.open_question,
            AtomType.constraint,
            AtomType.exclusion,
            AtomType.scope_item,
            AtomType.customer_instruction,
            AtomType.quantity,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del domain_pack
        suffix = path.suffix.lower()
        text = sample_text or ""
        lowered = normalize_text(text)
        reasons: list[str] = []
        confidence = 0.0
        artifact_type = ArtifactType.transcript if suffix in {".vtt", ".srt", ".json"} else ArtifactType.txt
        if suffix in {".vtt", ".srt"}:
            confidence = 0.95
            reasons.append(f"caption_extension:{suffix}")
        elif suffix == ".json" and text:
            # A transcript is recognised by SHAPE, not by parsing whole.
            #
            # This required json.loads() to succeed on `sample_text` -- which is
            # a truncated head of the file. A Fireflies transcript runs ~77 KB,
            # so the sample is cut mid-array, json.loads raises, and this scored
            # 0.0. JsonParser's deliberate 0.55 deferral then won by default and
            # flattened the call into key/value atoms: 1,315 of them on one deal,
            # typed `scope_item`, including `utterances[25].speaker: Trent
            # Torrence`. A speaker's name became a scope item, and 147,132 atoms
            # corpus-wide -- 35% of all evidence -- were conversational
            # fragments wearing the authority of extracted scope.
            #
            # Both parsers already agree on the signature; only this side
            # insisted on proof it cannot have from a truncated sample.
            head = text[:8000]
            # "segments" alone is NOT a transcript signal. It is an ordinary
            # business word -- network segments, cable segments, customer
            # segments -- and claiming every file containing it was how an
            # intake manifest ended up here. Diarised speech is `utterances`,
            # or per-item speaker AND text.
            shaped = '"utterances"' in head or ('"speaker"' in head and '"text"' in head)
            parsed_whole = False
            try:
                parsed_whole = isinstance(json.loads(text), (dict, list))
            except Exception:
                parsed_whole = False
            # Shape is required either way. This branch used to fire on ANY
            # .json that parsed, so a `case_manifest.json` mentioning
            # "segments" -- network segments, cable segments, an ordinary
            # business word -- was claimed as a transcript at 0.8 and taken
            # from the parser that could actually read it.
            if parsed_whole and shaped:
                confidence = 0.8
                reasons.append("json_transcript_candidate")
            elif shaped:
                # Above JsonParser's 0.55 deferral, below a clean parse, because
                # a shape read off a truncated head is the weaker claim.
                confidence = 0.7
                reasons.append("json_transcript_shape_truncated_sample")
        elif suffix in {".txt", ".md"}:
            if "open questions:" in lowered or "decisions:" in lowered:
                confidence = 0.9
                reasons.append("meeting_sections_detected")
            else:
                # Per-line check: ``detect_speaker`` / ``parse_timestamp``
                # were anchored on a leading-letter for safety, and the
                # bare ``.+$`` end-of-string anchor inside detect_speaker
                # fails when the full document has more than one line.
                # Walk the first ~40 lines and accept on any single hit.
                # A TIMESTAMP is unambiguous transcript evidence. A bare
                # "Name:" line is not -- ``detect_speaker`` matches any
                # "Label: value", which is what business documents are made
                # of. Accepting one such line in the first forty and stopping
                # meant "School District Contact:" on page one of a Request
                # for Proposals claimed the whole document at 0.82.
                #
                # Measured across 19 real .txt files: eight RFPs, SOWs, specs
                # and addenda were taken this way, while the two files that
                # actually ARE meeting notes have a speaker-line density of
                # 0.0% and qualify through ``meeting_sections_detected``
                # above. The highest densities in the corpus -- 37% to 43% --
                # are customer emails. The signal was never measuring
                # transcript-ness.
                #
                # So: timestamps qualify on sight; speaker labels qualify only
                # when they are how the document is BUILT, which is what a
                # transcript is. The threshold sits above every business
                # document measured (RFP 3.2-7.6%, SOW 1.7%, specs 1.3%,
                # addendum 10.2%, Q&A 15.2%) and above email headers, which
                # are short files where a few header lines dominate -- those
                # are claimed by EmailParser on its own evidence anyway.
                speaker_or_ts = False
                scan = [ln for ln in text.splitlines()[:400] if ln.strip()]
                if any(parse_timestamp(ln) is not None for ln in scan):
                    speaker_or_ts = True
                elif len(scan) >= 8:
                    turns = sum(1 for ln in scan if detect_speaker(ln) is not None)
                    if turns / len(scan) >= 0.50:
                        speaker_or_ts = True
                if not speaker_or_ts:
                    # Otter, Rev and Zoom put the speaker on its OWN line, so
                    # the colon density above reads 0% and a real transcript
                    # landed on the prose floor -- read, but with every
                    # utterance unattributed. Ask the folder, which is the
                    # same function the parser uses to canonicalise the file,
                    # so this decision and that rewrite can never disagree.
                    _folded, fold_stats = fold_standalone_speaker_lines(text)
                    if fold_stats["qualifies"]:
                        speaker_or_ts = True
                        reasons.append(
                            "standalone_speaker_lines("
                            f"{fold_stats['folds']} turns, "
                            f"{fold_stats['distinct_speakers']} speakers)"
                        )
                if not speaker_or_ts:
                    # PDF/meeting exports often use ``Name [mm:ss]`` mid-paragraph.
                    if re.search(
                        r"[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}\s*"
                        r"\[\d{1,2}:\d{2}(?::\d{2})?\]",
                        text[:4000],
                    ):
                        speaker_or_ts = True
                if speaker_or_ts:
                    confidence = 0.82
                    reasons.append("speaker_or_timestamp_markers")
        # Filename/title cue. This was 0.78 -- above MATCH_THRESHOLD -- and
        # applied to ANY suffix, so the name alone both created a claim and
        # created it for file types this parser does not support.
        #
        # Swept across 2500 real artifacts, renaming each without touching a
        # byte of content:
        #
        #   .json  NONE(0.00)        -> named "meeting_transcript" -> 0.78   2062
        #   .xlsx  XlsxParser(0.58)  -> named "meeting_transcript" -> 0.78     82
        #
        # The second is the bad one: a spreadsheet handed to this parser, which
        # reads it as text -- and .xlsx is not in supported_extensions at all.
        # "Q3_transcript_summary.xlsx" is an ordinary filename, so this needs
        # no adversary to happen.
        #
        # The same sweep answered whether the prior is load-bearing: of 2500
        # real artifacts, ZERO change parser when their name is neutralised.
        # Nothing relies on it. Timestamps, speaker density and the
        # own-line-speaker fold claim real transcripts on their own evidence.
        #
        # So: restricted to the extensions this parser actually supports, and
        # scored below MATCH_THRESHOLD, where a prior belongs. Kept in the
        # reasons so routing stays explainable.
        if suffix in {".txt", ".md", ".vtt", ".srt", ".json"} and (
            "transcript" in path.name.lower().replace("_", " ").replace("-", " ")
        ):
            confidence = max(confidence, 0.45)
            reasons.append("filename_transcript")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=artifact_type,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact(project_id="unknown_project", artifact_id=artifact_id, path=artifact_path)

    def segment_artifact(self, project_id: str, artifact_id: str, path: Path) -> list[ArtifactSegment]:
        return segment_transcript(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            parser_version=self.parser_version,
        )

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
        segments, export_header = self._segments_and_export_header(path)
        atoms: list[EvidenceAtom] = []
        header = self._call_header_atom(project_id=project_id, artifact_id=artifact_id, path=path)
        if header is None and export_header:
            header = self._export_header_atom(project_id=project_id, artifact_id=artifact_id, path=path, fields=export_header)
        if header is not None:
            atoms.append(header)
        for segment in segments:
            atoms.extend(
                self._atoms_from_segment(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    segment=segment,
                )
            )
        structured_doc = self._build_structured_doc(filename=path.name, segments=segments)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _export_header_atom(
        self, *, project_id: str, artifact_id: str, path: Path, fields: dict[str, str]
    ) -> EvidenceAtom:
        """The exporter's header block as one provenance record: the meeting's
        title (the first field's value), and its date from the first field
        value that parses as one. Same shape as the JSON call header so the
        envelope dates the document the same way."""
        items = list(fields.items())
        title = " ".join(items[0][1].split()) if items else ""
        date_iso = next((d for d in (_iso_date(v) for _, v in items[1:]) if d), None)
        parts = [f"Meeting: {title}"] if title else []
        if date_iso:
            parts.append(f"Date: {date_iso}")
        text = " | ".join(parts) or "Meeting export header"
        value: dict[str, Any] = {"kind": "meeting_header", "text": text, "title": title or None,
                                 "fields": {k: v for k, v in items[1:]}}
        if date_iso:
            value["document_date"] = date_iso
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "meeting_header", text),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.deal_metadata,
            raw_text=text,
            normalized_text=normalize_text(text),
            value=value,
            authority_class=AuthorityClass.meeting_note,
            confidence=0.95,
            review_status=ReviewStatus.auto_accepted,
            entity_keys=[],
            parser_version=self.parser_version,
            source_refs=[
                SourceRef(
                    id=stable_id("src", artifact_id, "meeting_header"),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.transcript,
                    filename=path.name,
                    locator={"line_start": 1, "line_end": len(items)},
                    extraction_method="transcript_export_header",
                    parser_version=self.parser_version,
                )
            ],
        )

    def _call_header_atom(self, *, project_id: str, artifact_id: str, path: Path) -> EvidenceAtom | None:
        """One provenance record for the call itself: title, date, participants.

        A JSON transcript states its own date when the exporter wrote one;
        a Fireflies id is a ULID, whose first ten characters are the
        creation time, so a call is dated even when the exporter did not
        say (live 010300: the Carl Painter call sorted after every document
        because nothing dated it). ``document_date`` is what the envelope
        reads as the document's own date.
        """
        if path.suffix.lower() != ".json":
            return None
        try:
            payload = json.loads(read_text(path))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        title = " ".join(str(payload.get("title") or "").split())
        date_iso = _iso_date(payload.get("date")) or _ulid_iso(str(payload.get("id") or ""))
        participants = [str(x).strip() for x in (payload.get("participants") or []) if str(x).strip()]
        if not (title or date_iso or participants):
            return None
        parts = []
        if title:
            parts.append(f"Call: {title}")
        if date_iso:
            parts.append(f"Date: {date_iso}")
        if participants:
            parts.append("Participants: " + ", ".join(participants))
        text = " | ".join(parts)
        value: dict[str, Any] = {"kind": "transcript_header", "text": text, "title": title or None}
        if date_iso:
            value["document_date"] = date_iso
        if participants:
            value["participants"] = participants
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "transcript_header", text),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.deal_metadata,
            raw_text=text,
            normalized_text=normalize_text(text),
            value=value,
            authority_class=AuthorityClass.meeting_note,
            confidence=0.95,
            review_status=ReviewStatus.auto_accepted,
            entity_keys=[],
            parser_version=self.parser_version,
            source_refs=[
                SourceRef(
                    id=stable_id("src", artifact_id, "transcript_header"),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.transcript,
                    filename=path.name,
                    locator={"line_start": 0, "line_end": 0, "kind": "transcript_header"},
                    extraction_method="transcript_metadata",
                    parser_version=self.parser_version,
                )
            ],
        )

    def _build_structured_doc(
        self,
        *,
        filename: str,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Render a transcript as a single page with one section per topic
        (Decisions, Action Items, Open Questions, Constraints, Discussion).
        Each utterance becomes a bullet item carrying its speaker and
        timestamp in plain text so an LLM can quote it directly.
        """
        bucket_order = [
            "Decisions",
            "Action Items",
            "Open Questions",
            "Constraints",
            "Discussion",
        ]
        buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in bucket_order}

        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = segment.get("speaker") or "Unknown"
            timestamp = segment.get("timestamp_start")
            stamp = f" [{timestamp}]" if timestamp else ""
            bullet_text = f"**{speaker}**{stamp}: {text}"
            section = (segment.get("section") or "").strip().lower()
            target = "Discussion"
            if "decision" in section:
                target = "Decisions"
            elif "action" in section:
                target = "Action Items"
            elif "question" in section:
                target = "Open Questions"
            elif "constraint" in section:
                target = "Constraints"
            else:
                lowered = text.lower()
                if DECISION_RE.search(text):
                    target = "Decisions"
                elif ACTION_RE.search(text):
                    target = "Action Items"
                elif QUESTION_RE.search(text):
                    target = "Open Questions"
                elif CONSTRAINT_RE.search(text):
                    target = "Constraints"
                del lowered
            buckets[target].append({"text": bullet_text, "children": []})

        sections: list[dict[str, Any]] = []
        for label in bucket_order:
            items = buckets[label]
            if not items:
                continue
            sections.append(
                make_section(
                    heading=label,
                    level=2,
                    blocks=[make_bullet_list(items=items)],
                )
            )
        if not sections:
            sections.append(
                make_section(
                    heading="Transcript",
                    level=2,
                    blocks=[make_paragraph("(empty transcript)")],
                )
            )
        page = make_page(page=0, title=filename, sections=sections)
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_TRANSCRIPT,
            filename=filename,
            artifact_type=ArtifactType.transcript.value,
            title=filename,
            metadata=[f"utterance_count: {len(segments)}"],
            pages=[page],
        )

    def _segments_from_path(self, path: Path) -> list[dict[str, Any]]:
        return self._segments_and_export_header(path)[0]

    def _segments_and_export_header(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
        """The file's speech segments, and the exporter's header block if a
        plain-text file opened with one (see :func:`peel_export_header`)."""
        suffix = path.suffix.lower()
        raw = read_text(path)
        if suffix == ".json":
            return self._segments_from_json(raw), None
        if suffix == ".vtt":
            return self._segments_from_text(self._clean_vtt(raw)), None
        if suffix == ".srt":
            return self._segments_from_text(self._clean_srt(raw)), None
        # A CRM meeting export (HubSpot's recap) carries its title and body
        # HTML-escaped, the title twice over: 010087's recap read "Summit 360
        # &amp;amp; PurTera IT". Two decode rounds undo the two encoding
        # layers (HubSpot's HTML storage, then the export); stopping there
        # keeps an "&amp;" the author actually typed (stored "&amp;amp;amp;")
        # as written instead of collapsing it to "&". Line breaks stay as
        # stored, so every segment keeps its line number.
        raw = decode_html_entities(raw, max_rounds=2)
        header, body, offset = peel_export_header(raw)
        segments = self._segments_from_text(body)
        for seg in segments:
            for key in ("line_start", "line_end"):
                if isinstance(seg.get(key), int):
                    seg[key] += offset
        return segments, header

    def _segments_from_json(self, raw_text: str) -> list[dict[str, Any]]:
        payload = json.loads(raw_text)
        items: list[dict[str, Any]]
        if isinstance(payload, dict):
            items = payload.get("utterances") or payload.get("segments") or []
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        segments: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            speaker = item.get("speaker")
            timestamp_start = item.get("start") or item.get("timestamp")
            segments.append(
                {
                    "utterance_index": idx,
                    "line_start": idx + 1,
                    "line_end": idx + 1,
                    "speaker": speaker,
                    "timestamp_start": str(timestamp_start) if timestamp_start is not None else None,
                    "timestamp_end": str(item.get("end")) if item.get("end") is not None else None,
                    "section": item.get("section"),
                    "text": text,
                }
            )
        return join_cut_fragments(segments)

    def _segments_from_text(self, raw_text: str) -> list[dict[str, Any]]:
        text = normalize_transcript_text(raw_text)
        return split_transcript_segments(text)

    def _clean_vtt(self, raw_text: str) -> str:
        """Reduce a WebVTT file to ``Speaker: text`` lines.

        The previous version dropped the ``WEBVTT`` banner, the timing lines
        and blank lines, and kept everything else verbatim. Three things went
        wrong with that, all visible on a file Teams would produce:

        * ``<v Cliff Creech>`` is the W3C voice span -- the standard way every
          major platform names a speaker -- and it survived into the atom as
          literal markup, with the speaker recorded as ``None``.
        * a cue *identifier* is an optional line before the timing line, and
          it is usually just ``1``, ``2``, ``3``. Those became atoms whose
          entire text was a digit. (``_clean_srt`` directly below already
          skipped these; VTT never got the same treatment.)
        * ``NOTE`` comments, ``STYLE`` and ``REGION`` blocks and the other cue
          payload tags (``<c>``, ``<i>``, inline timestamps) were all kept as
          if they were speech.

        ``webvtt-py`` implements the spec, so the parsing is delegated rather
        than re-derived: it separates identifier, timing and payload, strips
        cue tags, and exposes the voice span as ``caption.voice``. The old
        line filter remains as the fallback for a file the library refuses, so
        a malformed transcript degrades instead of raising.
        """
        try:
            import webvtt
        except Exception:  # pragma: no cover - dependency-free fallback
            return self._clean_vtt_fallback(raw_text)
        try:
            captions = list(webvtt.from_string(raw_text))
        except Exception:
            # Malformed, or not actually VTT -- keep the old behaviour.
            return self._clean_vtt_fallback(raw_text)
        if not captions:
            return self._clean_vtt_fallback(raw_text)

        lines: list[str] = []
        for caption in captions:
            text = " ".join((caption.text or "").split())
            if not text:
                continue
            speaker = " ".join((caption.voice or "").split())
            lines.append(f"{speaker}: {text}" if speaker else text)
        return "\n".join(lines)

    def _clean_vtt_fallback(self, raw_text: str) -> str:
        """The pre-library line filter, plus the cue-identifier skip it lacked."""
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.upper() == "WEBVTT":
                continue
            if stripped.upper().startswith(("NOTE", "STYLE", "REGION")):
                continue
            if "-->" in line:
                continue
            if not stripped:
                continue
            if stripped.isdigit():  # a cue identifier, not speech
                continue
            match = _VTT_VOICE_SPAN_RE.search(line)
            if match:
                speaker = " ".join(match.group(1).split())
                said = " ".join(match.group(2).split())
                line = f"{speaker}: {said}" if said else speaker
            lines.append(line)
        return "\n".join(lines)

    def _clean_srt(self, raw_text: str) -> str:
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                continue
            if "-->" in stripped:
                continue
            if not stripped:
                continue
            lines.append(line)
        return "\n".join(lines)

    def _speaker_role(self, speaker: str | None, text: str) -> str:
        source = normalize_text(f"{speaker or ''} {text}")
        if any(token in source for token in ("customer", "client")):
            return "customer"
        if any(token in source for token in ("purtera", "pm", "project manager", "coordinator")):
            return "internal"
        if speaker and "@" in speaker:
            email = speaker.split("<")[-1].strip("> ").lower()
            domain = email.split("@")[-1] if "@" in email else ""
            if domain and "purtera" not in domain:
                return "customer"
            if "purtera" in domain:
                return "internal"
        return "unknown"

    def _base_source_ref(self, artifact_id: str, filename: str, segment: dict[str, Any], speaker_role: str) -> SourceRef:
        section = segment.get("section")
        locator: dict[str, Any] = {
            "line_start": segment["line_start"],
            "line_end": segment["line_end"],
            "speaker": segment.get("speaker"),
            "speaker_role": speaker_role,
            "timestamp_start": segment.get("timestamp_start"),
            "timestamp_end": segment.get("timestamp_end"),
            "section": section,
            "utterance_index": segment["utterance_index"],
        }
        if segment.get("joined_utterance_indexes"):
            locator["joined_utterance_indexes"] = list(segment["joined_utterance_indexes"])
        if section:
            locator["section_path"] = [str(section)]
        return SourceRef(
            id=stable_id("src", artifact_id, segment["utterance_index"], segment["line_start"], segment["text"]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.transcript,
            filename=filename,
            locator=locator,
            extraction_method="transcript_rule_engine",
            parser_version=self.parser_version,
        )

    def _atoms_from_segment(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        segment: dict[str, Any],
    ) -> list[EvidenceAtom]:
        text = str(segment.get("text", "")).strip()
        if not text:
            return []
        lowered = normalize_text(text)
        pack = get_active_domain_pack()
        speaker_role = self._speaker_role(segment.get("speaker"), text)
        source_ref = self._base_source_ref(artifact_id, filename, segment, speaker_role)
        entity_keys = extract_meeting_entities(text)

        # A question is an ask, never a commitment. Live 010300: "Is the
        # children dentistry done after hours?" was emitted twice, once as an
        # open_question and once as a CONSTRAINT, so the brief carried a
        # question mark as a rule the crew had to work to. Asking whether
        # something is so does not make it so, whatever words the sentence
        # shares with a real constraint.
        _is_question = text.rstrip().endswith("?")

        atom_types: list[AtomType] = []
        if DECISION_RE.search(text):
            atom_types.append(AtomType.decision)
            if "we will" in lowered:
                atom_types.append(AtomType.meeting_commitment)
        if ACTION_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(alias))}\b", lowered)
            for aliases in pack.action_aliases.values()
            for alias in aliases
        ):
            atom_types.append(AtomType.action_item)
        if QUESTION_RE.search(text):
            atom_types.append(AtomType.open_question)
        if CONSTRAINT_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
            for patterns in pack.constraint_patterns.values()
            for pattern in patterns
        ):
            if not _is_question:
                atom_types.append(AtomType.constraint)
        if EXCLUSION_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
            for pattern in pack.exclusion_patterns
        ):
            atom_types.append(AtomType.exclusion)
        if SCOPE_RE.search(text):
            atom_types.append(AtomType.scope_item)
        if speaker_role == "customer" and (
            CUSTOMER_DIRECTIVE_RE.search(text)
            or any(
                re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
                for pattern in pack.customer_instruction_patterns
            )
        ):
            atom_types.append(AtomType.customer_instruction)
        if QUANTITY_RE.search(text):
            atom_types.append(AtomType.quantity)

        # section-driven typing for note bullets
        section = (segment.get("section") or "").lower()
        if section == "decisions" and AtomType.decision not in atom_types:
            atom_types.append(AtomType.decision)
        if section == "action Items".lower() and AtomType.action_item not in atom_types:
            atom_types.append(AtomType.action_item)
        if section == "open Questions".lower() and AtomType.open_question not in atom_types:
            atom_types.append(AtomType.open_question)

        atoms: list[EvidenceAtom] = []
        try:
            from app.core.vendor_site_ban import is_purtera_vendor_address

            banned_vendor_addr = is_purtera_vendor_address(text=text)
        except Exception:
            banned_vendor_addr = False
        if not banned_vendor_addr:
            for parsed in find_us_addresses_in_text(text):
                if (
                    not parsed.city
                    or not parsed.state
                    or parsed.state not in US_STATES
                    or not parsed.street_address
                ):
                    continue
                slug = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    f"{parsed.city}_{parsed.state}_{parsed.zip or parsed.street_address}".lower(),
                ).strip("_")
                display = f"{parsed.street_address}, {parsed.city}, {parsed.state} {parsed.zip or ''}".strip()
                site_keys = list(dict.fromkeys([*entity_keys, f"site:{slug}"]))
                aliases = list(dict.fromkeys(parsed.aliases))
                names = list(dict.fromkeys([display, parsed.city, *aliases]))
                atoms.append(
                    EvidenceAtom(
                        id=stable_id("atm", project_id, artifact_id, "transcript_note_physical_site", slug),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.physical_site,
                        raw_text=display,
                        normalized_text=normalize_text(display),
                        value={
                            "kind": "physical_site",
                            "id": slug,
                            "site_id": slug,
                            "name": display,
                            "names": names,
                            "aliases": aliases,
                            "street_address": parsed.street_address,
                            "address": parsed.street_address,
                            "city": parsed.city,
                            "state": parsed.state,
                            "zip": parsed.zip,
                            "inferred": True,
                            "source_context": text[:600],
                        },
                        entity_keys=site_keys,
                        source_refs=[source_ref],
                        authority_class=AuthorityClass.meeting_note,
                        confidence=0.72,
                        review_status=ReviewStatus.needs_review,
                        review_flags=["transcript_note_physical_site"],
                        parser_version=self.parser_version,
                    )
                )
        deduped_types: list[AtomType] = []
        for atom_type in atom_types:
            if atom_type not in deduped_types:
                deduped_types.append(atom_type)

        # Coverage floor. Every branch above is a *pattern* -- a keyword, an
        # alias, a question mark. When none fires the loop below runs zero
        # times and the utterance is gone: no atom, no receipt, nothing that
        # records it was ever said. Measured on a ten-turn call written in
        # ordinary language, five turns vanished, and they were the wrong
        # five: the scope ("forty sites before end of Q3"), the access
        # constraint ("dock is only open until two"), the exclusion ("not
        # paying for the mid-turn jumpers") and the part number all went,
        # while "Understood", "Noted" and "Good" survived on their keywords.
        #
        # So the utterance is kept untyped instead. Typing it is a judgement
        # that belongs downstream where it can be learned and corrected;
        # deciding it was never spoken is not a judgement this layer is
        # entitled to make.
        #
        # Kept -- and typed the way an email sentence no pattern recognised is
        # typed (``app.core.utterance_typing``): a pleasantry is an admission
        # chatter atom, anything else gets the shared coarse prose type,
        # flagged as a fallback guess. ``raw_utterance`` is not a type the
        # labeler has, so a 582-turn call reached the labeling page untyped.
        fallback_chatter: str | None = None
        fallback_typed = False
        if not deduped_types:
            from app.core.utterance_typing import fallback_utterance_type

            _fb_type, fallback_chatter = fallback_utterance_type(text)
            deduped_types.append(_fb_type)
            fallback_typed = True

        for atom_type in deduped_types:
            value: dict[str, Any] = {"text": text}
            review_status = ReviewStatus.auto_accepted
            review_flags: list[str] = []
            confidence = 0.78

            if atom_type == AtomType.raw_utterance or fallback_typed:
                # Deliberately the lowest confidence any transcript atom
                # carries, so it never outranks a typed one covering the same
                # words and never reads as an assertion about the deal.
                from app.core.utterance_typing import FALLBACK_TYPED_FLAG

                confidence = 0.40
                review_flags.append(FALLBACK_TYPED_FLAG)
                value["typed_by"] = "utterance_fallback"

            if atom_type == AtomType.quantity:
                match = QUANTITY_RE.search(text)
                if match:
                    op = (match.group(1) or "").strip().lower() or None
                    quantity = int(match.group(2))
                    item = (match.group(4) or "").strip()
                    value.update(
                        {
                            "quantity": quantity,
                            "unit": "count",
                            "item": item,
                            "operation": op,
                        }
                    )
            if atom_type == AtomType.action_item:
                owner = "customer" if "customer to" in lowered else ("purtera" if "purtera to" in lowered else speaker_role)
                value.update({"owner": owner, "action": text})
                if any(token in lowered for token in ("scope", "add", "remove", "price", "cost", "commercial")):
                    review_status = ReviewStatus.needs_review
            if atom_type == AtomType.constraint:
                value.update({"constraint_type": "access", "raw_constraint": text})
            if atom_type == AtomType.open_question:
                review_status = ReviewStatus.needs_review
                review_flags.append("missing_information_candidate")
                confidence = 0.74
            if atom_type == AtomType.exclusion:
                review_status = ReviewStatus.needs_review
                review_flags.extend(["verbal_commitment_requires_confirmation", "exclusion_present"])
            if atom_type in {AtomType.scope_item, AtomType.decision, AtomType.meeting_commitment, AtomType.quantity} and not fallback_typed:
                review_status = ReviewStatus.needs_review
                if "verbal_commitment_requires_confirmation" not in review_flags:
                    review_flags.append("verbal_commitment_requires_confirmation")
            if atom_type == AtomType.customer_instruction:
                review_status = ReviewStatus.needs_review
                review_flags.extend(["customer_spoken_instruction", "verbal_commitment_requires_confirmation"])

            atom = EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    segment["utterance_index"],
                    # A fallback-typed turn keeps the id it had as an untyped
                    # one, so labels already given to it still attach.
                    "raw_utterance" if fallback_typed else atom_type.value,
                    text,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=text,
                normalized_text=text.strip(),
                value=value,
                entity_keys=entity_keys,
                source_refs=[source_ref],
                authority_class=AuthorityClass.meeting_note,
                confidence=confidence,
                review_status=review_status,
                review_flags=sorted(set(review_flags)),
                parser_version=self.parser_version,
            )
            if fallback_chatter:
                from app.core.admission_chatter import mark_admission_chatter

                mark_admission_chatter(atom, fallback_chatter)
            atoms.append(atom)
        return atoms
