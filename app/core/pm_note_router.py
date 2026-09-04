"""One note from a PM, routed into every head it actually corrects.

The correction loop until now needed the PM to already know the answer to two
questions the system should answer itself: *which head got this wrong* and
*how far does my judgment reach*. The chip UI made them pick a head; the
question loop only ever spoke to the `gap` head. So a PM who writes what they
actually mean —

    "I prefer SLO instead of SLA because we guarantee objectives, not
     agreements. Also Chase quoted this one, and when Chase is on it we bill
     the blended rate, so stop asking about the PM fee."

— was teaching nothing. That note carries three separate lessons for three
different heads, one of them conditional on who is assigned, and every one of
them has a reason attached that is more valuable than the correction itself.

This module turns a note into :class:`Lesson` rows and hands them to the store.

**Routing** follows the house precedence, STORE → LLM → PATTERN, and never
guesses silently:

* **STORE** — routing is itself a relation (``note_head``), so a correction
  written against it makes the next note like this one route the same way with
  no model call. Nothing writes those corrections yet; the hook is here so
  mis-routing becomes teachable rather than a code change.
* **LLM** — proposes head, verdict and scope against the real head registry
  and the real atom taxonomy, as strict JSON. It only proposes.
* **PATTERN** — shape rules that need no model at all: "prefer X instead of
  Y", "should be a constraint", "stop asking about Z". These also run first
  where they are unambiguous, so the common notes cost nothing.

**Conditions** are the other half of what the PM said. "when Chase is
assigned" is not scope, it is a predicate on the deal, and a lesson carrying
one must not fire on deals where it does not hold. The condition rides in the
correction's ``relations["when"]`` and is enforced in
:meth:`FeedbackStore.resolve`, which refuses to fire a conditioned correction
unless the caller supplies facts that satisfy it.

**The why is kept verbatim.** It becomes the correction's instruction, so when
the lesson fires anywhere the brief can say whose judgment it was and why.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.pm_feedback import HEAD_REGISTRY

# ── the shapes a note takes, read structurally rather than by keyword list ──

#: "prefer SLO instead of SLA" / "use SLO over SLA" / "say SLO not SLA"
#: A term ends where the sentence turns: a connective, punctuation, or the end.
_TERM = r"[A-Za-z][\w./&+-]{0,40}(?:\s+(?!because|since|so\b|and\b|but\b|when\b|for\b)[\w./&+-]{1,20}){0,3}"
_PREFERENCE_RE = re.compile(
    r"\b(?:prefer|use|say|write|call\s+it)\s+"
    rf"[\"'“”]?(?P<new>{_TERM})[\"'“”]?\s+"
    r"(?:instead\s+of|rather\s+than|not|over|in\s+place\s+of)\s+"
    rf"[\"'“”]?(?P<old>{_TERM})[\"'“”]?",
    re.IGNORECASE,
)

#: "should be a constraint", "is a milestone not a task", "call this an exclusion"
_TYPE_RE = re.compile(
    r"\b(?:should\s+be|is|are|make\s+it|mark\s+it\s+as|call\s+(?:it|this))\s+"
    r"(?:an?\s+)?(?P<type>[a-z][a-z_]{3,30})\b",
    re.IGNORECASE,
)

#: "stop asking", "never ask", "don't ask about", "quit asking"
_STOP_ASKING_RE = re.compile(
    r"\b(?:stop|quit|never|don'?t|do\s+not)\s+(?:ever\s+)?(?:asking|ask)\b", re.IGNORECASE
)

#: "always ask", "make sure to ask", "we need to ask"
_ALWAYS_ASK_RE = re.compile(
    r"\b(?:always\s+ask|make\s+sure\s+(?:to\s+)?ask|we\s+(?:need|have)\s+to\s+ask|remember\s+to\s+ask)\b",
    re.IGNORECASE,
)

#: "never a site", "is not a site", "drop it", "that is noise"
_DROP_RE = re.compile(
    r"\b(?:never\s+(?:an?\s+)?(?:site|location|stakeholder|person)|is\s+not\s+(?:an?\s+)?\w+|"
    r"drop\s+(?:it|that|those)|that'?s\s+noise|ignore\s+(?:it|that|those))\b",
    re.IGNORECASE,
)

#: "when Chase is assigned", "if Chase owns it", "whenever Chase quotes"
_CONDITION_RE = re.compile(
    r"\b(?:when|whenever|if|for)\s+(?P<who>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?)"
    r"\s+(?:is\s+assigned|is\s+the\s+(?:owner|rep|pm|ae)|owns|quoted|quotes|is\s+on\s+it|runs\s+it|leads)\b",
    re.IGNORECASE,
)

#: "Chase quoted this" — the same fact stated the other way round.
_ACTOR_DID_RE = re.compile(
    r"\b(?P<who>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?)\s+"
    r"(?:quoted|priced|scoped|built|owns|owned|runs|ran)\s+(?:this|it|the\s+deal)\b"
)

#: The reason half of a note. Kept verbatim; never parsed for meaning.
_BECAUSE_RE = re.compile(r"\b(?:because|since|as|so\s+that|reason\s+being)\b\s*(?P<why>.+)", re.IGNORECASE)

#: Sentence-ish split that keeps decimals and abbreviations together.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(])|\n+|\s+(?:and\s+)?also\s+", re.IGNORECASE)

_TERMINOLOGY_HEAD = "terminology"


def _target_id(head: str, exemplar: str) -> str:
    """A stable identity for the thing a lesson judges."""
    probe = " ".join(str(exemplar or "").split()).lower()[:200]
    return f"note:{head}:{hashlib.sha1(probe.encode()).hexdigest()[:12]}" if probe else ""


@dataclass(frozen=True)
class Lesson:
    """One teachable judgment extracted from a note."""

    head: str
    exemplar: str
    new_value: str
    old_value: str = ""
    scope: str = "deal"
    condition: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = "pattern"
    confidence: float = 0.0

    def as_payload(self, *, deal_id: str, pm: str = "") -> dict[str, Any]:
        """The universal PM-correction payload :func:`apply_pm_correction` takes."""
        return {
            "head": self.head,
            "dealId": deal_id,
            # Two lessons for the same head on the same deal are two
            # judgments, not one restated: without a target of their own they
            # hash to the same correction id and the second silently replaces
            # the first, taking its reason with it.
            "targetId": _target_id(self.head, self.exemplar),
            "text": self.exemplar,
            "oldValue": self.old_value,
            "newValue": self.new_value,
            "scope": self.scope,
            # The reason is recorded, never embedded: it describes the
            # judgment, not the text the judgment must match.
            "rationale": self.rationale,
            "pm": pm,
            "relations": {"when": dict(self.condition)} if self.condition else {},
        }


@dataclass
class NoteRouting:
    lessons: list[Lesson] = field(default_factory=list)
    unrouted: list[str] = field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "lessons": [
                {
                    "head": l.head,
                    "head_label": (HEAD_REGISTRY[l.head].label if l.head in HEAD_REGISTRY else l.head),
                    "exemplar": l.exemplar,
                    "old_value": l.old_value,
                    "new_value": l.new_value,
                    "scope": l.scope,
                    "condition": l.condition,
                    "rationale": l.rationale,
                    "source": l.source,
                    "confidence": l.confidence,
                }
                for l in self.lessons
            ],
            "unrouted": list(self.unrouted),
            "rationale": self.rationale,
        }


# ── the pieces of a note ───────────────────────────────────────────────


def split_clauses(note: str) -> list[str]:
    """A note is usually several lessons wearing one paragraph."""
    text = " ".join(str(note or "").split())
    if not text:
        return []
    parts = [p.strip(" ,;") for p in _CLAUSE_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= 8]


def extract_rationale(clause: str) -> str:
    """The PM's own why, verbatim. Never interpreted, only carried."""
    m = _BECAUSE_RE.search(clause or "")
    return " ".join(m.group("why").split()).rstrip(".") if m else ""


def _normalize_person(name: str) -> str:
    return re.sub(r"[^a-z]+", "_", str(name or "").strip().lower()).strip("_")


def extract_condition(clause: str, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """"when Chase is assigned" / "Chase quoted this" → a predicate on the deal.

    Grounded when the deal's own people are supplied: a name that matches a
    known owner or stakeholder is recorded as that person, so the condition
    survives a nickname. An ungrounded name is still recorded — the PM knows
    who they mean — but only ever fires where the caller supplies facts.
    """
    m = _CONDITION_RE.search(clause or "") or _ACTOR_DID_RE.search(clause or "")
    if not m:
        return {}
    who = _normalize_person(m.group("who"))
    if not who:
        return {}
    known = {}
    for key in ("owner", "owners", "stakeholders", "people"):
        val = (facts or {}).get(key)
        if isinstance(val, str):
            known[_normalize_person(val)] = val
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                known[_normalize_person(v)] = str(v)
    # First-name match against a known full name ("chase" → "Chase Whitfield").
    if who not in known:
        for k, original in known.items():
            if k.split("_")[0] == who:
                who = k
                break
    return {"field": "owner", "equals": who}


def _atom_type_names() -> set[str]:
    try:
        from app.core.schemas import AtomType

        return {t.value for t in AtomType}
    except Exception:
        return set()


# ── routing ────────────────────────────────────────────────────────────


#: The instruction half of a note, stripped so the exemplar is the SUBJECT.
#: "Stop asking about the PM fee because …" teaches nothing if the prototype
#: is the sentence "stop asking …": what has to match a real question is "the
#: PM fee". The prototype is the thing judged, never the judging.
_ASK_LEAD_RE = re.compile(
    r"^\s*(?:(?:please|just)\s+)?(?:stop|quit|never|don'?t|do\s+not|always|make\s+sure\s+to|"
    r"remember\s+to|we\s+(?:need|have)\s+to)\s+(?:ever\s+)?(?:asking|ask)\s*"
    r"(?:about|for|whether|if|that)?\s*",
    re.IGNORECASE,
)
_TRAILING_WHY_RE = re.compile(r"\s*\b(?:because|since|so\s+that|reason\s+being)\b.*$", re.IGNORECASE)
_SUBJECT_BEFORE_COPULA_RE = re.compile(
    r"^(?P<subject>[^,;]{2,60}?)\s*(?:,\s*(?:it|they|that|this)\s+)?\b(?:is|are|was|were)\b",
    re.IGNORECASE,
)


#: Openers that make a phrase an ask. A gap prototype is matched against real
#: questions, so it has to be shaped like one: the same words as a statement
#: sit 0.06 further away than as a question, which is the difference between
#: firing and not.
_INTERROGATIVE_RE = re.compile(
    r"^(?:who|what|when|where|which|why|how|whose|whom|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|has|have|had)\b",
    re.IGNORECASE,
)


def _as_question(phrase: str) -> str:
    text = " ".join(str(phrase or "").split()).strip(" .,:;")
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    if _INTERROGATIVE_RE.match(text) and not text.endswith("?"):
        text += "?"
    return text


#: "stop asking" anywhere in the sentence, not only at its head: a PM writes
#: "Who owns pathway is not a customer question, stop asking it" as readily as
#: "Stop asking who owns pathway".
_ASK_ANYWHERE_RE = re.compile(
    r"\b(?:stop|quit|never|don'?t|do\s+not|always|make\s+sure\s+to|remember\s+to|"
    r"we\s+(?:need|have)\s+to)\s+(?:ever\s+)?(?:asking|ask)\b\s*"
    r"(?:about|for|whether|if|that)?\s*",
    re.IGNORECASE,
)
#: Pronouns that point back at the subject rather than naming it.
_PRONOUN_ONLY_RE = re.compile(r"^(?:it|that|this|those|them|these)\b[\s.,]*$", re.IGNORECASE)


def _ask_subject(clause: str) -> str:
    """What the ask is ABOUT, with the instruction and the reason removed,
    written the way the question it must match is written.

    The prototype has to be the QUESTION, not the sentence telling us to drop
    it: "stop asking who provides the bridge" embedded whole sits 0.66 from
    the real question and never fires; "Who provides the bridge?" sits 0.75
    and does.
    """
    core = _TRAILING_WHY_RE.sub("", str(clause or "")).strip()
    m = _ASK_ANYWHERE_RE.search(core)
    if m:
        tail = core[m.end():].strip(" .,:;")
        head = core[: m.start()].strip(" .,:;")
        if len(tail.split()) >= 2 and not _PRONOUN_ONLY_RE.match(tail):
            core = tail
        elif head:
            core = _judged_subject(head)
        else:
            core = tail or head
    else:
        core = _judged_subject(core)
    if len(core.split()) < 2:
        return " ".join(str(clause or "").split())
    return _as_question(core)


def _judged_subject(clause: str) -> str:
    """What a keep/drop judgment is about: the noun phrase before the verb."""
    core = _TRAILING_WHY_RE.sub("", str(clause or "")).strip()
    m = _SUBJECT_BEFORE_COPULA_RE.match(core)
    if m:
        subject = m.group("subject").strip(" .,:;")
        if 0 < len(subject.split()) <= 6:
            return subject
    return " ".join(str(clause or "").split())


#: Words that carry no subject matter, so they cannot vote on which question a
#: note is about.
_STOPWORDS = frozenset(
    """a an the this that these those and or but for to of in on at by with from is are was
    were be been being do does did we you they it he she our your their my i us them who what
    when where which why how confirm please stop asking ask about any all each per not no""".split()
)


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def ground_in_questions(subject: str, questions: list[str], *, floor: float = 0.6) -> str:
    """The question on the PM's screen that this note is about, if any.

    A note is written while looking at a card. "Stop asking who owns pathway"
    means the card that reads "Who owns pathway (conduit, sleeves, fish,
    raceway, drywall patch) on this project?" — and that card, not the
    paraphrase, is what the lesson must be learned from: the paraphrase sits
    0.67 from it, under any threshold worth having, while the card itself is
    an exact match forever after.

    Containment rather than similarity: the note is deliberately shorter than
    the question, so what matters is that its words are IN the question.
    """
    want = _content_words(subject)
    if not want or not questions:
        return ""
    best, best_score = "", 0.0
    for q in questions:
        have = _content_words(q)
        if not have:
            continue
        score = len(want & have) / max(1, min(len(want), len(have)))
        if score > best_score:
            best, best_score = str(q), score
    return best if best_score >= floor else ""


def _pattern_lessons(
    clause: str,
    condition: dict,
    rationale: str,
    scope: str,
    questions: list[str] | None = None,
) -> list[Lesson]:
    """The shapes that need no model. Each is a claim about STRUCTURE."""
    out: list[Lesson] = []

    m = _PREFERENCE_RE.search(clause)
    if m:
        new = " ".join(m.group("new").split())
        old = " ".join(m.group("old").split())
        if new.lower() != old.lower():
            out.append(
                Lesson(
                    head=_TERMINOLOGY_HEAD,
                    exemplar=old,
                    new_value=new,
                    old_value=old,
                    scope=scope,
                    condition=condition,
                    rationale=rationale,
                    source="pattern",
                    confidence=0.95,
                )
            )

    if _STOP_ASKING_RE.search(clause):
        subject = _ask_subject(clause)
        out.append(
            Lesson(
                head="gap",
                exemplar=ground_in_questions(subject, questions or []) or subject,
                new_value="invalid",
                old_value="valid",
                scope=scope,
                condition=condition,
                rationale=rationale,
                source="pattern",
                confidence=0.9,
            )
        )
    elif _ALWAYS_ASK_RE.search(clause):
        subject = _ask_subject(clause)
        out.append(
            Lesson(
                head="gap",
                exemplar=ground_in_questions(subject, questions or []) or subject,
                new_value="valid",
                old_value="invalid",
                scope=scope,
                condition=condition,
                rationale=rationale,
                source="pattern",
                confidence=0.9,
            )
        )

    types = _atom_type_names()
    for m in _TYPE_RE.finditer(clause):
        candidate = m.group("type").lower()
        if candidate in types:
            out.append(
                Lesson(
                    head="type",
                    exemplar=clause,
                    new_value=candidate,
                    scope=scope,
                    condition=condition,
                    rationale=rationale,
                    source="pattern",
                    confidence=0.85,
                )
            )
            break

    if _DROP_RE.search(clause) and not any(l.head == "gap" for l in out):
        out.append(
            Lesson(
                head="admission",
                exemplar=_judged_subject(clause),
                new_value="drop",
                old_value="keep",
                scope=scope,
                condition=condition,
                rationale=rationale,
                source="pattern",
                confidence=0.8,
            )
        )
    return out


def _store_head(clause: str, *, deal_id: str, store: Any) -> str:
    """Learned routing: which head does a note like this correct?"""
    if store is None:
        return ""
    try:
        from app.core.decide import DecisionScope

        hit = store.resolve(
            relation="note_head",
            text=clause,
            candidates=sorted(set(HEAD_REGISTRY) | {_TERMINOLOGY_HEAD}),
            context="",
            scope=DecisionScope(deal_id=deal_id),
            instruction="",
            relations=None,
        )
    except Exception:
        return ""
    return str(getattr(hit, "verdict", "") or "") if hit is not None else ""


_LLM_PROMPT = """You convert a project manager's note about a bid-document \
parser into STRUCTURED lessons for a learned correction store.

Each lesson names the HEAD it corrects and the VERDICT that head should return.

Heads:
{heads}

Atom types (verdicts for the "type" head):
{types}

Rules:
- One lesson per distinct judgment. A note may contain several.
- "exemplar" is the text the lesson should fire on: quote the PM's own words for
  the thing being judged, not the whole note.
- "scope" is "deal" when the judgment is about this deal's circumstances and
  "global" when it is about how we work in general.
- Never invent a head or a verdict outside the lists above.

Note:
{note}

Output ONLY a JSON object: {{"lessons": [{{"head": ..., "exemplar": ..., \
"new_value": ..., "old_value": "", "scope": "deal"}}]}}

JSON:"""


def _default_synthesize(note: str) -> dict:
    from app.core.plain_rule_compiler import _call_ollama, _extract_json_object

    heads = "\n".join(
        f'- "{k}": {v.label} (relation {v.relation})' for k, v in sorted(HEAD_REGISTRY.items())
    )
    heads += f'\n- "{_TERMINOLOGY_HEAD}": Preferred wording (relation preferred_term)'
    types = ", ".join(sorted(_atom_type_names())) or "(unavailable)"
    raw = _call_ollama(_LLM_PROMPT.format(heads=heads, types=types, note=note), max_tokens=768)
    return _extract_json_object(raw)


def _llm_lessons(
    note: str,
    condition: dict,
    rationale: str,
    scope: str,
    synthesize: Callable[[str], dict] | None,
) -> list[Lesson]:
    fn = synthesize or _default_synthesize
    try:
        proposal = fn(note) or {}
    except Exception:
        return []
    rows = proposal.get("lessons")
    if not isinstance(rows, list):
        return []
    valid_heads = set(HEAD_REGISTRY) | {_TERMINOLOGY_HEAD}
    types = _atom_type_names()
    out: list[Lesson] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        head = str(row.get("head") or "").strip().lower()
        new_value = str(row.get("new_value") or "").strip()
        exemplar = " ".join(str(row.get("exemplar") or "").split())
        if head not in valid_heads or not new_value or not exemplar:
            continue
        if head == "type" and types and new_value not in types:
            continue
        row_scope = str(row.get("scope") or scope).strip().lower()
        out.append(
            Lesson(
                head=head,
                exemplar=exemplar,
                new_value=new_value,
                old_value=str(row.get("old_value") or "").strip(),
                scope=row_scope if row_scope in ("deal", "pack", "global") else scope,
                condition=condition,
                rationale=rationale,
                source="llm",
                confidence=0.7,
            )
        )
    return out


def _dedupe(lessons: list[Lesson]) -> list[Lesson]:
    seen: set[tuple] = set()
    out: list[Lesson] = []
    for l in lessons:
        key = (l.head, l.new_value.lower(), l.exemplar.lower()[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out


def route_note(
    note: str,
    *,
    deal_id: str = "",
    facts: dict[str, Any] | None = None,
    store: Any = None,
    synthesize: Callable[[str], dict] | None = None,
    default_scope: str = "deal",
    questions: list[str] | None = None,
) -> NoteRouting:
    """A PM's note → the lessons it contains, each aimed at one head.

    The condition and the reason are read from the WHOLE note and carried onto
    every lesson in it: "when Chase is on it ... because we bill blended" is
    one circumstance and one reason governing everything the PM said.
    """
    text = " ".join(str(note or "").split())
    if not text:
        return NoteRouting()

    clauses = split_clauses(text) or [text]

    # A sentence that only says who did the work sets the context for the note
    # ("Chase quoted this one."). A circumstance written INSIDE a sentence
    # conditions that sentence alone: in "when Chase is assigned we bill the
    # blended rate, so stop asking about the PM fee. Also prefer SLO instead
    # of SLA", the wording preference is not about Chase.
    note_condition: dict[str, Any] = {}
    body: list[str] = []
    for clause in clauses:
        cond = extract_condition(clause, facts)
        bare_actor = bool(_ACTOR_DID_RE.search(clause)) and len(_content_words(clause)) <= 6
        if cond and bare_actor and not note_condition:
            note_condition = cond
            continue
        body.append(clause)

    routing = NoteRouting(rationale=extract_rationale(text))
    unrouted: list[str] = []
    for clause in body or clauses:
        # The reason belongs to the sentence that gives it, never to the ones
        # after it: a greedy read attached "because we commit to objectives"
        # to two later lessons it had nothing to do with.
        rationale = extract_rationale(clause)
        condition = extract_condition(clause, facts) or note_condition
        found = _pattern_lessons(clause, condition, rationale, default_scope, questions)
        if not found:
            head = _store_head(clause, deal_id=deal_id, store=store)
            if head:
                found = [
                    Lesson(
                        head=head,
                        exemplar=clause,
                        new_value="invalid" if head == "gap" else "drop" if head == "admission" else "",
                        scope=default_scope,
                        condition=condition,
                        rationale=rationale,
                        source="store",
                        confidence=0.9,
                    )
                ]
                found = [l for l in found if l.new_value]
        if not found:
            unrouted.append(clause)
        routing.lessons.extend(found)

    if unrouted:
        proposed = _llm_lessons(
            " ".join(unrouted), note_condition, routing.rationale, default_scope, synthesize
        )
        if proposed:
            routing.lessons.extend(proposed)
            unrouted = []

    routing.lessons = _dedupe(routing.lessons)
    routing.unrouted = unrouted
    return routing


def apply_note(
    note: str,
    *,
    store: Any,
    deal_id: str = "",
    pm: str = "",
    facts: dict[str, Any] | None = None,
    synthesize: Callable[[str], dict] | None = None,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    """Route a note and commit every lesson in it. Returns what was learned."""
    from app.core.pm_feedback import apply_pm_correction

    routing = route_note(
        note, deal_id=deal_id, facts=facts, store=store, synthesize=synthesize, questions=questions
    )
    committed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for lesson in routing.lessons:
        try:
            cid = apply_pm_correction(store, lesson.as_payload(deal_id=deal_id, pm=pm))
            committed.append({"correction_id": cid, **routing.as_dict()["lessons"][len(committed)]})
        except Exception as exc:  # one bad lesson must not lose the others
            failed.append({"head": lesson.head, "error": str(exc)[:200]})
    return {
        "note": " ".join(str(note or "").split()),
        "condition": routing.lessons[0].condition if routing.lessons else {},
        "rationale": routing.rationale,
        "committed": committed,
        "failed": failed,
        "unrouted": routing.unrouted,
    }


__all__ = [
    "Lesson",
    "NoteRouting",
    "apply_note",
    "extract_condition",
    "ground_in_questions",
    "extract_rationale",
    "route_note",
    "split_clauses",
]
