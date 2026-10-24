"""The labeling opportunities, read from the registries the labeling page uses.

A *labeling opportunity* is one question the card asks about a line: its type,
one reading (``sow_coverage``, ``crew_size``...), one relation (``answers``),
or the company's verdict. Every opportunity belongs to exactly one head of
``app/core/label_heads.json`` and carries text written for the labeler:

* the head's ``label`` and ``question`` ("What will this line turn out to mean
  for the work...?"),
* the reading's ``label`` and ``desc`` from ``app/core/atom_types.json``,
* one description per answer it allows (a type's ``desc``, an ``about`` or
  ``wants`` option's ``desc``, a closed-list reading's ``value_desc``).

This module turns those two JSON files into a ``Schema``: the list of
opportunities, each with its description text and its answer descriptions.
The model never sees a hand-kept list of classes. Its output layer for an
opportunity is *built from that text* (see ``heads.py``), so editing a
description in the registry is editing the model.

It reads the JSON files by path and never imports ``app``: the parser service
does not depend on this package, and this package does not depend on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .values import (BILLING_TYPES, COMMERCIAL_NUMBERS, DEAL_SERVICE_DESC, DEAL_SERVICES, HOURS_PER,
                     TERM_ENUMS, TERM_KINDS, TERM_NUMBERS)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADS_PATH = REPO_ROOT / "app" / "core" / "label_heads.json"
TYPES_PATH = REPO_ROOT / "app" / "core" / "atom_types.json"
JUDGMENTS_PATH = Path(__file__).resolve().parent / "judgment_heads.json"
REASONS_PATH = REPO_ROOT / "app" / "core" / "judgment_reasons.json"

#: Words a universal description must not carry (project instructions, note
#: grammar). A sentence holding one is dropped from a universal opportunity's
#: description, so the base never learns company names from its own schema.
POLICY_WORDS = (
    "purtera", "deal kit", "atlas", "portal", "gantt", "hubspot",
    "55\"", "55-inch", "55 inch", "our rule", "we send", "pricing workbook",
)

#: Kinds of answer an opportunity takes.
CLASS = "class"          # one of a fixed set
BINARY = "binary"        # true / false
NUMBER = "number"        # a number the text states (crew_size, labor_hours)
PRESENCE = "presence"    # a phrase is present or not (commitment, urgency)
RELATION = "relation"    # an edge to another line
CAUSAL_RELATIONS = {"answers": 1, "derived_from": 1, "contradicts": 1}

#: What one answer is about. Line questions are asked of every line; the
#: judgment tabs also ask about two lines, a group of lines (a document, a
#: table, a sheet) or the whole deal.
LINE, PAIR, GROUP, DEAL = "line", "pair", "group", "deal"
SIZES = (LINE, PAIR, GROUP, DEAL)

#: Registry key of a column's question text where it is not "<col>_doc".
_DOC_KEYS = {"supplier": "suppliers_doc"}

#: Who said a line and to whom (the label's ``said_by`` / ``said_to``): the
#: party's role and side, as column heads. "unknown" is the page's blank and
#: teaches nothing. The line's sender *inputs* (speaker_role / side / org) are
#: the parser's read of the same people (c3_record.sender), in a coarser
#: vocabulary; these heads are the person's answer.
SAID_ROLES = {
    "customer": "The end customer the work is for, or someone on its staff.",
    "provider": "The company doing the field work, writing as the provider of the job.",
    "seller": "The provider's sales side: the account manager or seller who quotes the job.",
    "internal": "Someone on the provider's own side talking among the provider's own people.",
    "reseller": "A company that buys the work to resell it to the end customer.",
    "partner": "Another company working with the provider or the customer on this job: a partner, subcontractor or distributor.",
    "system": "A system, not a person: a ticketing tool, a form or a scheduler on the provider's side.",
    "automated": "An automated message from outside: a notification, an auto-reply or a mailing system.",
}
SAID_SIDES = {
    "ours": "The provider's own side: the company doing the work.",
    "theirs": "The other side: the customer, a reseller, a partner or anyone outside the provider.",
}
#: Column heads on a dotted label field: (column, key, field, question, vocabulary, several answers).
SAID_COLUMNS = {
    "said_by": (("col:said_by_role", "said_by.role", "Who said this line: the role of the person or system that wrote or spoke it.", SAID_ROLES, False),
                ("col:said_by_side", "said_by.side", "Which side said this line: the provider's own side or the other side.", SAID_SIDES, False)),
    "said_to": (("col:said_to_role", "said_to.role", "Who this line was said to: the role of each person it was addressed to.", SAID_ROLES, True),
                ("col:said_to_side", "said_to.side", "Which sides this line was said to: the provider's own side, the other side, or both.", SAID_SIDES, True)),
}

#: Absent is always a class: a reading the line does not have.
ABSENT = "_absent"


@dataclass(frozen=True)
class Answer:
    value: str
    description: str


@dataclass(frozen=True)
class Opportunity:
    key: str                 # "read:sow_coverage", "col:label_type", "rel:answers"
    field: str               # the label field it reads ("sow_coverage")
    source: str              # "column" | "read" | "relation" | "judgment"
    head: str                # "consequence.outcome"
    space: str               # "consequence"
    layer: str               # "universal" | "company"
    kind: str                # CLASS | BINARY | NUMBER | PRESENCE | RELATION
    description: str         # the question the labeler is asked, in full
    answers: tuple[Answer, ...] = ()
    size: str = LINE         # LINE | PAIR | GROUP | DEAL
    multiple: bool = False   # independent selected answers, not a first-member class

    @property
    def universal(self) -> bool:
        return self.layer == "universal"

    def index(self, value: Any) -> int | None:
        """Class index of a stored value, or None when it is outside the set."""
        if value is None or value == "":
            value = ABSENT
        if self.kind == BINARY:
            value = {"true": "true", "false": "false", True: "true", False: "false"}.get(
                value if isinstance(value, bool) else str(value).lower(), ABSENT)
        elif self.kind in (PRESENCE, NUMBER) and value != ABSENT:
            value = "present"
        for i, a in enumerate(self.answers):
            if a.value == value:
                return i
        return None


@dataclass
class Schema:
    opportunities: list[Opportunity]
    spaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: str = ""
    #: Sentences dropped from universal guidance because they name company
    #: policy: (opportunity key, sentence). Shown by ``python -m ml.c3.card``.
    scrubbed: list[tuple[str, str]] = field(default_factory=list)
    #: Free-text readings ("the reasoning behind <field>"): not answers, but
    #: per-field WHYs the teacher reads (data.field_note_targets).
    note_fields: list[str] = field(default_factory=list)
    #: Judgment tabs that ask an existing line question in other words:
    #: judgment head -> (opportunity key, verdict -> that question's answer).
    judgment_aliases: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)
    #: Closed reason codes per labeling surface. A sentence is not a code.
    reason_codes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Judgment surfaces whose verdict is a string, not a closed class.
    literal_judgments: tuple[str, ...] = ("norm", "terminology")

    def by_key(self) -> dict[str, Opportunity]:
        return {o.key: o for o in self.opportunities}

    def of_field(self, source: str, name: str) -> Opportunity | None:
        return next((o for o in self.opportunities
                     if o.source == source and o.field == name), None)

    def select(self, *, layer: str | None = None, kind: str | None = None,
               space: str | None = None, size: str | None = LINE) -> list[Opportunity]:
        """Opportunities by layer, kind and space. Line questions only unless
        ``size`` says otherwise (None: every size)."""
        return [o for o in self.opportunities
                if (layer is None or o.layer == layer)
                and (kind is None or o.kind == kind)
                and (space is None or o.space == space)
                and (size is None or o.size == size)]

    def texts(self) -> list[str]:
        """Every description the model encodes, opportunity first, then answers."""
        out: list[str] = []
        for o in self.opportunities:
            out.append(o.description)
            out.extend(a.description for a in o.answers)
        return out


def shape_of(schema: "Schema") -> list[tuple[str, list[str]]]:
    """What the weights depend on: each opportunity's key, in order, with its
    answer values in order. The texts are encoded at run time, so a schema that
    changes only text keeps this shape (and every tensor's size)."""
    return [(o.key, [a.value for a in o.answers]) for o in schema.opportunities]


def shape_digest(schema: "Schema") -> str:
    """sha256 of ``shape_of`` (canonical JSON, as ``readiness.digest``)."""
    return hashlib.sha256(json.dumps(shape_of(schema), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def scrub(text: str, universal: bool) -> str:
    """Drop the sentences of a universal description that name company policy."""
    text = " ".join(str(text or "").split())
    if not universal:
        return text
    keep = [s for s in re.split(r"(?<=[.!?])\s+", text)
            if not any(w in s.lower() for w in POLICY_WORDS)]
    return " ".join(keep).strip()


#: How a stored value is written in the text the model reads, where its name
#: would mislead: the noise type is stored as ``_keep`` but "keep" is the
#: opposite answer in admission and co_action.
_SPELLING = {"_keep": "noise"}


def _spell(value: str) -> str:
    return _SPELLING.get(value) or value.replace("_", " ").replace("-", " ").strip()


def _named(name: str, doc: str) -> str:
    """"<name>: <doc>", without saying the name twice when the doc opens with it."""
    if doc.lower().startswith(name.lower() + ":"):
        doc = doc[len(name) + 1:].lstrip()
    return f"{name}: {doc}"


#: Pilot A root cause (M2): the "Unlike a site attribute ..., a site
#: implementation note ... or a stakeholder" clauses b395fbb wrote into the
#: line-type texts name the look-alikes inside the text that gets EMBEDDED, so
#: each prototype is pulled toward its neighbours' (the live weights under the
#: new texts read 010358 at 37 sites, not 89). The contrast is taught by the
#: WHYs and the near-miss pairs; the registry (atom_types.json "desc", read by
#: labelers) keeps it, and ``type_contrast`` returns it, but the model's answer
#: text does not carry it. C3_EMBED_CONTRAST=1 embeds it again (the b395fbb
#: texts); the schema digest records which, so a checkpoint never reads under
#: the other.
EMBED_CONTRAST = os.environ.get("C3_EMBED_CONTRAST", "0") == "1"
_UNLIKE = re.compile(r"^\s*unlike\b", re.I)


def _contrast_pieces(doc: str, others: Iterable[str]) -> list[tuple[str, bool]]:
    """(piece, is_contrast) for each clause of ``doc`` (split after ". " / "; ").
    A contrast clause opens with "Unlike", or redirects to another answer of
    the same head ("invoice timing is a payment term", "it would be a task")."""
    names = sorted({o.lower() for o in others if o}, key=len, reverse=True)
    alts = "|".join(r"[\s-]+".join(map(re.escape, re.split(r"[\s-]+", n))) for n in names)
    redirect = (re.compile(r"\b(?:is|are|would be)\s+(?:a\s+|an\s+)?(?:" + alts + r")\s*[.;]?\s*$", re.I)
                if names else None)
    out = []
    for piece in re.split(r"(?<=[.;])\s+", doc.strip()):
        if not piece:
            continue
        core = piece.strip()
        out.append((piece, bool(_UNLIKE.match(core) or (redirect is not None and redirect.search(core)))))
    return out


#: A lead-in that only pointed back at a dropped contrast clause ("Greetings
#: and closers are small talk; this is the rest: ...").
_BACK_REF = re.compile(r"^this is the rest:\s*", re.I)


def embedded_doc(doc: str, others: Iterable[str]) -> str:
    """``doc`` without its contrast clauses (the text the model embeds)."""
    kept, dropped = [], False
    for p, c in _contrast_pieces(doc, others):
        if c:
            dropped = True
            continue
        if dropped:
            p = _BACK_REF.sub("", p)
            p = p[:1].upper() + p[1:]
        kept.append(p)
        dropped = False
    text = " ".join(kept).strip()
    if text.endswith(";"):
        text = text[:-1].rstrip() + "."
    return text


#: Text pins (pilot A M2, the description gate): heads whose rewritten texts cost
#: the start checkpoint more than the gate allows read INIT's texts instead
#: (ml/c3/runE/init_texts.json, run C at 0b6efce), until a refit or a run has
#: learned the new ones. ``heads``: the question and every answer; ``answers``:
#: single answers of a head ({key: [value, ...]}). C3_TEXT_PINS=0 turns them off
#: (every head reads this code's texts); a path reads another pin file. The
#: schema digest records the texts, so a checkpoint never reads under the other.
TEXT_PINS_PATH = REPO_ROOT / "ml" / "c3" / "runE" / "text_pins.json"
INIT_TEXTS_PATH = REPO_ROOT / "ml" / "c3" / "runE" / "init_texts.json"


def text_pins_path() -> Path | None:
    v = os.environ.get("C3_TEXT_PINS", "")
    if v == "0":
        return None
    p = Path(v) if v else TEXT_PINS_PATH
    return p if p.is_file() else None


def apply_text_pins(opps: list["Opportunity"], pins: dict[str, Any], init: list) -> list["Opportunity"]:
    """``opps`` with the pinned heads / answers read in INIT's words. A head
    whose answer values differ from INIT's is left as it is (no INIT text to pin)."""
    from dataclasses import replace as _replace  # noqa: PLC0415

    old = {k: (q, dict((v, t) for v, t in ans), [v for v, _ in ans]) for k, q, ans in init}
    heads = set(pins.get("heads") or [])
    answers = {k: set(v) for k, v in (pins.get("answers") or {}).items()}
    out = []
    for o in opps:
        got = old.get(o.key)
        if got is None or got[2] != [a.value for a in o.answers] or (o.key not in heads and o.key not in answers):
            out.append(o)
            continue
        q, texts, _ = got
        if o.key in heads:
            o = _replace(o, description=q, answers=tuple(Answer(a.value, texts[a.value]) for a in o.answers))
        else:
            o = _replace(o, answers=tuple(Answer(a.value, texts[a.value]) if a.value in answers[o.key] else a
                                          for a in o.answers))
        out.append(o)
    return out


def type_contrast(types_path: Path | None = None) -> dict[str, list[str]]:
    """The line types' contrast clauses (guidance for labelers and WHY writers;
    never embedded): type name -> clauses, from atom_types.json."""
    types = json.loads(Path(types_path or TYPES_PATH).read_text(encoding="utf-8"))
    items = [(t["name"], t.get("desc", "")) for t in types["types"]] + [(types["keep"]["name"], types["keep"]["desc"])]
    spelled = {n: _spell(n) for n, _ in items}
    out = {}
    for n, d in items:
        got = [p.strip() for p, c in _contrast_pieces(d, [v for k, v in spelled.items() if k != n]) if c]
        if got:
            out[n] = got
    return out


def _value_kind(values: str) -> tuple[str, list[str]]:
    v = str(values or "").strip()
    if v == "true/false":
        return BINARY, ["true", "false"]
    if "|" in v:
        return CLASS, [p.strip() for p in v.split("|") if p.strip()]
    if v in ("a number",) or v.startswith("a size in") or v == "the hours":
        return NUMBER, []
    return PRESENCE, []


def _answers(kind: str, values: Iterable[str], subject: str,
             docs: dict[str, str] | None = None,
             names: dict[str, str] | None = None) -> tuple[Answer, ...]:
    """Answer texts. A closed-list answer reads "<value>. <its description>":
    the subject stays in the question, so the answers of one head do not all
    open with the same words (a shared prefix pulls their prototypes together).
    ``names`` writes a value by its label instead of its spelled name."""
    docs, names = docs or {}, names or {}
    if kind == BINARY:
        return (Answer(ABSENT, f"{subject}: not marked"),
                Answer("true", f"{subject}: yes"),
                Answer("false", f"{subject}: no"))
    if kind in (PRESENCE, NUMBER):
        return (Answer(ABSENT, f"{subject}: the line does not say"),
                Answer("present", f"{subject}: the line says it"))
    out = [Answer(ABSENT, f"{subject}: not this kind of line")]
    for v in values:
        out.append(Answer(v, f"{names.get(v) or _spell(v)}. {docs.get(v, '')}".strip()))
    return tuple(out)


def dropped(text: str) -> list[str]:
    """The sentences ``scrub`` would drop from a universal description."""
    text = " ".join(str(text or "").split())
    return [s for s in re.split(r"(?<=[.!?])\s+", text)
            if any(w in s.lower() for w in POLICY_WORDS)]


def apply_guidance(schema: Schema, guidance: dict[str, Any]) -> Schema:
    """Add written decision guidance to opportunities and their answers.

    ``guidance`` maps an opportunity key to
    ``{"how_to_decide": "...", "answers": {"<value>": "when to pick it, and why"}}``.
    The text is appended to the descriptions the heads are built from, so it
    changes the heads directly. Universal guidance loses any sentence naming
    company policy (recorded in ``schema.scrubbed``); write that in the
    company opportunities or in a company rule card instead.
    """
    by_key = schema.by_key()
    unknown = sorted(set(guidance) - set(by_key))
    if unknown:
        raise ValueError(f"guidance for unknown opportunities: {unknown}")
    out = []
    for o in schema.opportunities:
        g = guidance.get(o.key)
        if not g:
            out.append(o)
            continue
        how = str(g.get("how_to_decide", ""))
        notes = {str(k): str(v) for k, v in (g.get("answers") or {}).items()}
        bad = sorted(set(notes) - {a.value for a in o.answers})
        if bad:
            raise ValueError(f"guidance for {o.key}: not answers of it: {bad}")
        if o.universal:
            schema.scrubbed += [(o.key, s) for t in [how, *notes.values()] for s in dropped(t)]
        desc = f"{o.description} How to decide: {scrub(how, o.universal)}" if how else o.description
        answers = tuple(Answer(a.value, f"{a.description} {scrub(notes[a.value], o.universal)}".strip())
                        if a.value in notes else a for a in o.answers)
        out.append(Opportunity(**{**o.__dict__, "description": desc, "answers": answers}))
    schema.opportunities = out
    return schema


def guidance_template(schema: Schema) -> dict[str, Any]:
    """An empty guidance file covering every class-like opportunity."""
    return {o.key: {"how_to_decide": "", "answers": {a.value: "" for a in o.answers}}
            for o in schema.opportunities if o.kind != RELATION}


#: The term kinds are line types; their answer texts say what each one is.
_TERM_KIND_DESC = {
    "payment_term": "When and how invoices are raised and paid.",
    "contract_term": "A legal term of the agreement: length, renewal, warranty, liability, termination.",
    "change_order_rule": "How changes to the work are requested, priced and charged, including cancellation fees.",
    "pricing_assumption": "A term that sets how the price is worked out: billing basis, increments, minimums, rates.",
    "commercial_total": "A total or subtotal of the price.",
    "not_a_term": "The line states no commercial term.",
}


def _add_extract_and_reason_opportunities(opps: list[Opportunity], homes: dict[str, dict[str, Any]],
                                         aliases: dict[str, tuple[str, dict[str, str]]],
                                         reason_heads: dict[str, list[dict[str, Any]]]) -> None:
    """Closed extract slots and reason codes. Free-text verdicts stay literals."""

    def home(name: str) -> dict[str, Any]:
        return dict(homes.get(name) or {"head": name, "space": "content", "layer": "universal"})

    deal = {"head": "deal.answers", "space": "content", "layer": "universal"}
    # OWNER-CHECK: the alm, camera_vms_operations and msp texts in
    # values.DEAL_SERVICE_DESC were drafted from the value names alone.
    service_docs = {v: DEAL_SERVICE_DESC.get(v, "") for v in DEAL_SERVICES}
    opps.append(Opportunity(
        key="deal:primary_service", field="primary_service", source="judgment", kind=CLASS, size=DEAL,
        description=("The deal as a whole. What kind of field work is this job mainly? Decide from the signed or "
                     "latest scope of work: the work most of the labour goes to, not the equipment brand, "
                     "the customer's industry or a side task."),
        answers=_answers(CLASS, service_docs, "primary service", service_docs), **deal))

    # How many sites the job has, as the customer or partner states it (the
    # deal answers' declared_site_count). Presence trains as a whole-deal
    # card; the count trains the deal-number regressor (losses.deal_number_loss).
    opps.append(Opportunity(
        key="deal:site_count", field="declared_site_count", source="judgment", kind=NUMBER, size=DEAL,
        description=("The deal as a whole. How many sites does the job have, as the customer or partner states it before "
                     "the quote: \"145 locations\", a list of 101 sites, \"about 160 stores\". Count places the crew goes, "
                     "not devices, visits or floors; a count the provider worked out itself is not a stated count."),
        answers=_answers(NUMBER, [], "site count", {}), **deal))

    term = home("term")
    kind_docs = {k: _TERM_KIND_DESC.get(k, "") for k in TERM_KINDS}
    opps.append(Opportunity(
        key="jdg:term", field="term_kind", source="judgment", kind=CLASS, size=LINE,
        description="What kind of commercial term does this line state?",
        answers=_answers(CLASS, kind_docs, "term kind", kind_docs), **term))
    for key, allowed in TERM_ENUMS.items():
        docs = {v: f"The {_spell(key)} is {_spell(v)}." for v in allowed}
        opps.append(Opportunity(
            key=f"jdg:term:{key}", field=key, source="judgment", kind=CLASS, size=LINE,
            description=f"The {_spell(key)} this term states.",
            answers=_answers(CLASS, docs, key, docs), **term))
    for key in TERM_NUMBERS:
        opps.append(Opportunity(
            key=f"jdg:term:{key}", field=key, source="judgment", kind=NUMBER, size=LINE,
            description=f"The {_spell(key)} this term states.",
            answers=_answers(NUMBER, [], key, {}), **term))

    commercial = home("commercial")
    billing_docs = {v: f"The finished kit bills it {_spell(v)}." for v in BILLING_TYPES}
    opps.append(Opportunity(
        key="jdg:commercial", field="commercial_billing", source="judgment", kind=CLASS, size=LINE,
        description="What billing shape did the finished kit give this request?",
        answers=_answers(CLASS, billing_docs, "billing", billing_docs), **commercial))
    for key in COMMERCIAL_NUMBERS:
        opps.append(Opportunity(
            key=f"jdg:commercial:{key}", field=key, source="judgment", kind=NUMBER, size=LINE,
            description=f"The kit's {_spell(key)}.",
            answers=_answers(NUMBER, [], key, {}), **commercial))

    # The hours verdict's unit and count (hours=..;per=..;qty=..), on the main model.
    hours = home("hours")
    per_docs = {"site": "The hours are counted per site or location.",
                "visit": "The hours are counted per visit or trip.",
                "device": "The hours are counted per device or unit installed.",
                "job": "The hours are for the whole job."}
    opps.append(Opportunity(
        key="jdg:hours:per", field="hours_per", source="judgment", kind=CLASS, size=LINE,
        description="What one unit of the hours estimate for this request is: a site, a visit, a device or the whole job.",
        answers=_answers(CLASS, {v: per_docs[v] for v in HOURS_PER}, "hours per", per_docs), **hours))
    opps.append(Opportunity(
        key="jdg:hours:qty", field="hours_qty", source="judgment", kind=NUMBER, size=LINE,
        description="How many of those units the hours estimate for this request multiplies by.",
        answers=_answers(NUMBER, [], "hours qty", {}), **hours))

    by_key = {o.key: o for o in opps}
    for name, items in reason_heads.items():
        if not items:
            continue
        parent = by_key.get(f"jdg:{name}")
        if parent is None and name in aliases:
            parent = by_key.get(aliases[name][0])
        base = home(name)
        if parent is not None:
            base = {"head": parent.head, "space": parent.space, "layer": parent.layer}
        universal = base["layer"] == "universal"
        docs = {str(item["value"]): scrub(str(item.get("desc") or ""), universal) for item in items}
        opps.append(Opportunity(
            key=f"jdg:{name}:reason", field=f"{name}_reason", source="judgment", kind=CLASS,
            size=parent.size if parent is not None else LINE,
            description=f"Why was the { _spell(name) } verdict chosen?",
            answers=_answers(CLASS, docs, f"{name} reason", docs), **base))


def load_schema(heads_path: Path = HEADS_PATH, types_path: Path = TYPES_PATH,
                guidance: str | Path | dict[str, Any] | None = None,
                judgments_path: Path = JUDGMENTS_PATH) -> Schema:
    heads = json.loads(Path(heads_path).read_text(encoding="utf-8"))
    judgments = json.loads(Path(judgments_path).read_text(encoding="utf-8"))["heads"]
    aliases: dict[str, tuple[str, dict[str, str]]] = {}
    types = json.loads(Path(types_path).read_text(encoding="utf-8"))
    reads = {r["key"]: r for r in types["reads"]}
    rels = {r["key"]: r for r in types["relations"]}
    spaces = {s["key"]: s for s in heads["spaces"]}

    def lead(h: dict[str, Any], universal: bool) -> str:
        return scrub(f"{h.get('label', '')}. {h.get('question', '')}", universal)

    opps: list[Opportunity] = []
    note_fields: list[str] = []
    homes: dict[str, dict[str, Any]] = {}
    for h in heads["heads"]:
        if h.get("layer") not in ("universal", "company"):
            continue                      # meta: bookkeeping, never trained
        universal = h["layer"] == "universal"
        base = dict(head=h["key"], space=h.get("space") or "", layer=h["layer"])

        for col in h.get("columns", []):
            if col == "label_type":
                docs = {t["name"]: scrub(t.get("desc", ""), universal) for t in types["types"]}
                docs[types["keep"]["name"]] = scrub(types["keep"]["desc"], universal)
                values = list(docs)
                if not EMBED_CONTRAST:
                    docs = {v: embedded_doc(d, [_spell(u) for u in values if u != v]) for v, d in docs.items()}
                opps.append(Opportunity(
                    key="col:label_type", field="label_type", source="column", kind=CLASS,
                    description=lead(h, universal) + " What kind of fact the line states.",
                    answers=tuple(Answer(v, _named(_spell(v), docs[v])) for v in values), **base))
            elif col in ("about", "wants", "supplier"):
                options = types[col if col != "supplier" else "suppliers"]
                docs = {o["key"]: scrub(o.get("desc", ""), universal) for o in options}
                names = {o["key"]: scrub(o.get("label", ""), universal) for o in options}
                opps.append(Opportunity(
                    key=f"col:{col}", field=col, source="column", kind=CLASS,
                    # The supplier options and their question live under the
                    # plural key ("suppliers", "suppliers_doc").
                    description=f"{lead(h, universal)} {scrub(types.get(_DOC_KEYS.get(col, col + '_doc'), ''), universal)}".strip(),
                    answers=_answers(CLASS, docs, col, docs, names), **base))
            elif col in SAID_COLUMNS:
                for key, fld, question, vocab, several in SAID_COLUMNS[col]:
                    docs = {v: scrub(d, universal) for v, d in vocab.items()}
                    opps.append(Opportunity(
                        key=key, field=fld, source="column", kind=CLASS, multiple=several,
                        description=f"{lead(h, universal)} {scrub(question, universal)}",
                        answers=_answers(CLASS, docs, key.split(":", 1)[1].replace("_", " "), docs), **base))
            elif col == "internal_only":
                opps.append(Opportunity(
                    key="col:internal_only", field="internal_only", source="column", kind=BINARY,
                    description=f"{lead(h, universal)} Was this line said only among the provider's own people, "
                                "with nobody from the customer or another company in the conversation?",
                    answers=_answers(BINARY, [], "internal only"), **base))
            elif col == "hints":
                # "What told you?": which part of the context decided the label
                # The context-hint chips permit several simultaneous answers.
                docs = {c["key"]: scrub(c.get("desc", ""), universal) for c in types.get("context_hints", [])}
                names = {c["key"]: scrub(c.get("label", ""), universal) for c in types.get("context_hints", [])}
                opps.append(Opportunity(
                    key="col:hints", field="hints", source="column", kind=CLASS, multiple=True,
                    description=f"{lead(h, universal)} Which part of the context decided it.",
                    answers=_answers(CLASS, docs, "what told you", docs, names), **base))
            # entity_keys, weight_tier, hint_refs, note, rejected: spans, weights
            # and text, trained by the claim, rationale and conduct paths.

        if "admission" in h.get("tasks", []):
            # Should this text be an atom at all? keep, or drop as not a fact
            # (wreckage, boilerplate, small talk, a fragment). Derived per row
            # in data.featurize the way app.learning.human_labels does.
            # A drop is noise or background (data._derived drops copies,
            # repeats, pointers, summaries and reasons, which are real facts).
            docs = {"keep": "The line states a fact of its own about the work and is the learned line for it, "
                            "even when the company sets that fact aside on purpose.",
                    "drop": "The line teaches no fact of its own. Either it is noise (a greeting, sign-off, "
                            "message header, footer, page chrome, empty cell, filler speech) or it is background, "
                            "not an input: a copy or repeat of an earlier line, a heading or lead-in that only points "
                            "at the lines under it, a summary that more specific lines already say, a reason that only "
                            "explains another line, or an introduction that only says who someone is."}
            opps.append(Opportunity(
                key="col:admission", field="admission", source="column", kind=CLASS,
                description="Which lines are inputs. Should this line be learned as a fact, or is it noise or background?",
                answers=_answers(CLASS, docs, "admission", docs),
                **base))

        for rk in h.get("reads", []):
            r = reads.get(rk)
            if r is None or r.get("layer") not in ("universal", "company"):
                continue
            if r.get("values") == "free text":
                note_fields.append(rk)    # a per-field WHY, not an answer
                continue
            kind, values = _value_kind(r.get("values", ""))
            subject = scrub(r.get("label", rk), universal) or _spell(rk)
            # Each closed-list answer's own written description (value_desc).
            value_docs = {str(v): scrub(d, universal) for v, d in (r.get("value_desc") or {}).items()}
            opps.append(Opportunity(
                key=f"read:{rk}", field=rk, source="read", kind=kind, multiple=bool(r.get("multi")) or rk in ("train_for", "needed_by"),
                description=" ".join(x for x in (
                    lead(h, universal), f"{subject}.", scrub(r.get("desc", ""), universal)) if x),
                answers=_answers(kind, values, subject, value_docs), **base))
            if rk == "train_for":
                # Which parser a Deal Kit line trains is our own routing rule
                # (the Deal Kit is our pricing workbook), so it lives in the
                # company layer; data._derived moves it there per line.
                opps.append(Opportunity(
                    key="read:co_deal_kit_route", field="co_deal_kit_route", source="read", kind=kind, multiple=True,
                    description=f"{lead(h, False)} Which parser one of our own Deal Kit lines trains.",
                    answers=_answers(kind, values, subject, value_docs),
                    **{**base, "layer": "company"}))

        for rel in h.get("relations", []):
            r = rels.get(rel, {"label": rel})
            opps.append(Opportunity(
                key=f"rel:{rel}", field=rel, source="relation", kind=RELATION,
                description=" ".join(x for x in (
                    lead(h, universal), f"{r.get('label', rel)}.", scrub(r.get("desc", ""), universal)) if x),
                **base))

        for name in h.get("judgments", []):
            # A judgment tab's question, at the size it is asked (one line, two
            # lines, a document or the whole deal), in the layer of the head
            # that lists it. Its text comes from judgment_heads.json.
            homes[name] = base
            j = judgments.get(name)
            if j is None:
                continue                  # extract / literal surfaces added below
            if "alias" in j:
                aliases[name] = (j["alias"]["target"], dict(j["alias"]["map"]))
                continue
            docs = {str(v): scrub(d, universal) for v, d in j["answers"].items()}
            subject = scrub(j.get("subject", ""), universal) or _spell(name)
            opps.append(Opportunity(
                key=f"jdg:{name}", field=name, source="judgment", kind=CLASS, size=j["size"],
                multiple=bool(j.get("multi")),
                description=" ".join(x for x in (
                    lead(h, universal), scrub(j.get("question", ""), universal)) if x),
                answers=_answers(CLASS, docs, subject, docs), **base))

    reasons = json.loads(REASONS_PATH.read_text(encoding="utf-8")).get("heads") or {}
    reason_codes = {str(name): tuple(str(item["value"]) for item in items)
                    for name, items in reasons.items()}
    _add_extract_and_reason_opportunities(opps, homes, aliases, reasons)
    pins_path = text_pins_path()
    if pins_path is not None and INIT_TEXTS_PATH.is_file():
        opps = apply_text_pins(opps, json.loads(pins_path.read_text(encoding="utf-8")),
                               json.loads(INIT_TEXTS_PATH.read_text(encoding="utf-8"))["texts"])
    schema = Schema(opportunities=opps, spaces=spaces, version=str(heads.get("version", "")),
                    note_fields=note_fields, judgment_aliases=aliases, reason_codes=reason_codes)
    if guidance is not None:
        if not isinstance(guidance, dict):
            guidance = json.loads(Path(guidance).read_text(encoding="utf-8"))
        guidance = {k: v for k, v in guidance.items() if not k.startswith("_")}
        apply_guidance(schema, guidance)
    return schema
