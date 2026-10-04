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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADS_PATH = REPO_ROOT / "app" / "core" / "label_heads.json"
TYPES_PATH = REPO_ROOT / "app" / "core" / "atom_types.json"
JUDGMENTS_PATH = Path(__file__).resolve().parent / "judgment_heads.json"

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

#: What one answer is about. Line questions are asked of every line; the
#: judgment tabs also ask about two lines, a group of lines (a document, a
#: table, a sheet) or the whole deal.
LINE, PAIR, GROUP, DEAL = "line", "pair", "group", "deal"
SIZES = (LINE, PAIR, GROUP, DEAL)

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


def scrub(text: str, universal: bool) -> str:
    """Drop the sentences of a universal description that name company policy."""
    text = " ".join(str(text or "").split())
    if not universal:
        return text
    keep = [s for s in re.split(r"(?<=[.!?])\s+", text)
            if not any(w in s.lower() for w in POLICY_WORDS)]
    return " ".join(keep).strip()


def _spell(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


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
             docs: dict[str, str] | None = None) -> tuple[Answer, ...]:
    docs = docs or {}
    if kind == BINARY:
        return (Answer(ABSENT, f"{subject}: not marked"),
                Answer("true", f"{subject}: yes"),
                Answer("false", f"{subject}: no"))
    if kind in (PRESENCE, NUMBER):
        return (Answer(ABSENT, f"{subject}: the line does not say"),
                Answer("present", f"{subject}: the line says it"))
    out = [Answer(ABSENT, f"{subject}: not this kind of line")]
    for v in values:
        out.append(Answer(v, f"{subject}: {_spell(v)}. {docs.get(v, '')}".strip()))
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
                opps.append(Opportunity(
                    key="col:label_type", field="label_type", source="column", kind=CLASS,
                    description=lead(h, universal) + " What kind of fact the line states.",
                    answers=tuple(Answer(v, f"{_spell(v)}: {docs[v]}") for v in values), **base))
            elif col in ("about", "wants", "supplier"):
                options = types[col if col != "supplier" else "suppliers"]
                docs = {o["key"]: scrub(f"{o.get('label', '')}. {o.get('desc', '')}", universal)
                        for o in options}
                opps.append(Opportunity(
                    key=f"col:{col}", field=col, source="column", kind=CLASS,
                    description=f"{lead(h, universal)} {scrub(types.get(col + '_doc', ''), universal)}".strip(),
                    answers=_answers(CLASS, docs, col, docs), **base))
            elif col == "hints":
                # "What told you?": which part of the context decided the label
                # (the card's context-hint chips; a list, its first member taught).
                docs = {c["key"]: scrub(f"{c.get('label', '')}. {c.get('desc', '')}", universal)
                        for c in types.get("context_hints", [])}
                opps.append(Opportunity(
                    key="col:hints", field="hints", source="column", kind=CLASS,
                    description=f"{lead(h, universal)} Which part of the context decided it.",
                    answers=_answers(CLASS, docs, "what told you", docs), **base))
            # entity_keys, weight_tier, hint_refs, note, rejected: spans, weights
            # and text, trained by the claim, rationale and conduct paths.

        if "admission" in h.get("tasks", []):
            # Should this text be an atom at all? keep, or drop as not a fact
            # (wreckage, boilerplate, small talk, a fragment). Derived per row
            # in data.featurize the way app.learning.human_labels does.
            docs = {"keep": "keep: a real fact about the work, worth an atom",
                    "drop": "drop: not a fact; wreckage, boilerplate, small talk or a torn fragment"}
            opps.append(Opportunity(
                key="col:admission", field="admission", source="column", kind=CLASS,
                description=lead(h, universal), answers=_answers(CLASS, docs, "admission", docs),
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
                key=f"read:{rk}", field=rk, source="read", kind=kind,
                description=" ".join(x for x in (
                    lead(h, universal), f"{subject}.", scrub(r.get("desc", ""), universal)) if x),
                answers=_answers(kind, values, subject, value_docs), **base))
            if rk == "train_for":
                # Which parser a Deal Kit line trains is our own routing rule
                # (the Deal Kit is our pricing workbook), so it lives in the
                # company layer; data._derived moves it there per line.
                opps.append(Opportunity(
                    key="read:co_deal_kit_route", field="co_deal_kit_route", source="read", kind=kind,
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
            j = judgments.get(name)
            if j is None:
                continue                  # not trained (judgment_heads.json says why)
            if "alias" in j:
                aliases[name] = (j["alias"]["target"], dict(j["alias"]["map"]))
                continue
            docs = {str(v): scrub(d, universal) for v, d in j["answers"].items()}
            subject = scrub(j.get("subject", ""), universal) or _spell(name)
            opps.append(Opportunity(
                key=f"jdg:{name}", field=name, source="judgment", kind=CLASS, size=j["size"],
                description=" ".join(x for x in (
                    lead(h, universal), scrub(j.get("question", ""), universal)) if x),
                answers=_answers(CLASS, docs, subject, docs), **base))

    schema = Schema(opportunities=opps, spaces=spaces, version=str(heads.get("version", "")),
                    note_fields=note_fields, judgment_aliases=aliases)
    if guidance is not None:
        if not isinstance(guidance, dict):
            guidance = json.loads(Path(guidance).read_text(encoding="utf-8"))
        guidance = {k: v for k, v in guidance.items() if not k.startswith("_")}
        apply_guidance(schema, guidance)
    return schema
