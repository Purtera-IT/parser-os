"""Every labeled signal trains: an audit by ablation and gradients.

For each signal the training files carry (a label field, a reading, a note
kind, a link relation and its note, a card head and its note, a card field,
a deal answer), the audit

1. counts it in the files (rows the loader would train on: a person's row,
   or an approved draft's, not set aside),
2. removes it from one or more carrier files and featurizes them again, the
   way the loop does (``loop.training_batches``): the Batch fields that move
   are the signal's trace through ``data.featurize``,
3. runs ``losses.c3_loss`` on the part with and without the signal, same
   seed, eval mode, every term computed: the terms whose value moves are the
   signal's consumers (``losses.py:<line>`` etc.),
4. backpropagates the difference of each consuming term, at the run's own
   weight for that term (the loss table, cumulative over the stages the run
   trains, as ``loop.gold_weights`` opens them), into the model: the
   gradient share is ||grad(with) - grad(without)|| / ||grad(with)|| of the
   step's total loss.

Status: ``OK`` (the step's gradient moves with the signal), ``UNUSED`` (the
Batch is the same without it: featurize drops it), ``ZERO-GRAD`` (the Batch
moves but no weighted term's gradient does: a term at weight 0 in every
trained stage, a head the view masks, a field no loss reads), ``ERROR``
(featurize refuses the file without it: the signal is required, so it is
consumed; reported, not gated).

``--min-count N --allowlist FILE`` turns it into a gate: exit 1 when a
signal with at least N rows is UNUSED or ZERO-GRAD in every set it appears
in, unless the allowlist names it with a reason. ``pod/runE.sh`` runs it in
the SMOKE path with ``SIGNAL_AUDIT=1``.

    python -m ml.c3.signal_audit --tiny --weights-from ml/c3/pod/runE.sh \\
        --train ~/c3-local/runE/train_presow/*.json --out audit.md
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from .data import Batch, _is_a_person, excluded
from .losses import LossWeights, c3_loss
from .notes import _CHAIN, PARSER_TAG

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- the signals

#: Label row fields that are the line's identity, provenance or the page's
#: context, not a labeler's answer.
LABEL_META = frozenset({
    "id", "deal_id", "label_key", "labeler", "atom_id", "compile_id", "text", "filename",
    "doc_type", "page", "section", "lead_in", "neighbors_above", "neighbors_below", "table_ref",
    "decide_text", "correction_id", "purpose", "labeled_at", "origin", "entered_at", "source_date",
    "doc_id", "doc_kind", "order", "speaker_role", "speaker_side", "speaker_org", "doc_direction",
    "stage_at", "created_at", "updated_at", "saved_at",
    "note", "reads_set",            # audited by note kind and by reading
})
CARD_META = frozenset({"id", "deal_id", "labeler", "compile_id", "purpose", "judged_at",
                       "target_key", "target", "text", "parser_value", "correction_id"})
ANSWER_META = frozenset({"deal_id", "labeler", "purpose", "answered_at"})
NOTE_KINDS = ("universal", "purtera", "parser", "arithmetic")


def _empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _purtera_at(lines: list[str], company: str) -> int | None:
    marker = f"[{company}]"
    for k, ln in enumerate(lines):
        if ln.lstrip().lower().startswith(marker):
            return k
    return None


def note_kinds(note: str, company: str = "purtera") -> set[str]:
    """The parts of a note, by the note grammar (notes.split_note)."""
    lines = str(note or "").splitlines()
    out = set()
    if any(ln.lstrip().lower().startswith(PARSER_TAG) for ln in lines):
        out.add("parser")
    lines = [ln for ln in lines if not ln.lstrip().lower().startswith(PARSER_TAG)]
    k = _purtera_at(lines, company)
    universal = "\n".join(lines if k is None else lines[:k]).strip()
    if universal:
        out.add("universal")
        if _CHAIN.search(universal):
            out.add("arithmetic")
    if k is not None and "\n".join(lines[k:]).strip()[len(company) + 2:].strip():
        out.add("purtera")
    return out


def without_kind(note: str, kind: str, company: str = "purtera") -> str:
    """``note`` with one of its parts removed."""
    lines = str(note or "").splitlines()
    parser = [ln for ln in lines if ln.lstrip().lower().startswith(PARSER_TAG)]
    body = [ln for ln in lines if not ln.lstrip().lower().startswith(PARSER_TAG)]
    k = _purtera_at(body, company)
    universal, company_part = (body, []) if k is None else (body[:k], body[k:])
    if kind == "parser":
        parser = []
    elif kind == "purtera":
        company_part = []
    elif kind == "universal":
        # Keep an EXCLUDE marker: removing the WHY must not un-exclude a row.
        universal = [ln for ln in universal[:1] if "EXCLUDE_FROM_TRAINING" in ln.upper()]
    elif kind == "arithmetic":
        universal = [_CHAIN.sub("", ln) for ln in universal]
    return "\n".join(universal + company_part + parser)


def _rows(payload: dict, part: str) -> list[dict]:
    return [r for r in payload.get(part) or [] if isinstance(r, dict)]


def _trains(row: dict) -> bool:
    return _is_a_person(row.get("labeler")) and not excluded(row)


def _parts(note: str, company: str) -> tuple[str, str]:
    """(universal, company) text of a note, the [parser] line dropped."""
    lines = [ln for ln in str(note or "").splitlines() if not ln.lstrip().lower().startswith(PARSER_TAG)]
    k = _purtera_at(lines, company)
    if k is None:
        return "\n".join(lines).strip(), ""
    rest = "\n".join(lines[k:]).strip()
    return "\n".join(lines[:k]).strip(), rest[len(company) + 2:].strip()


def _words(t: Any) -> int:
    return len(str(t or "").split())


def _is_sentence(reason: Any) -> bool:
    """A reason the labeler wrote as a sentence, not a picker code (notes.judgment_note)."""
    from .notes import _REASON_CODE_RE  # noqa: PLC0415

    t = str(reason or "").strip()
    return bool(t) and not _REASON_CODE_RE.fullmatch(t)


def census(payload: dict) -> tuple[Counter, Counter]:
    """(signal -> rows that carry it and would train, signal -> reasoning words
    it carries). Words are counted on the text signals only: label WHYs and
    [purtera] lines, link notes, card notes and sentence reasons, field notes
    (``<read>_note``, ``rejected_reads``) and deal-answer notes."""
    c: Counter = Counter()
    w: Counter = Counter()
    company = str(payload.get("company") or "purtera")
    for r in _rows(payload, "labels"):
        if not _trains(r):
            continue
        for k, v in r.items():
            if k not in LABEL_META and not _empty(v) and v is not False:
                c[f"label.{k}"] += 1
        if isinstance(r.get("rejected_reads"), dict):
            w["label.rejected_reads"] += sum(_words(v) for v in r["rejected_reads"].values())
        for k, v in (r.get("reads_set") or {}).items() if isinstance(r.get("reads_set"), dict) else ():
            if not _empty(v):
                c[f"read.{k}"] += 1
                if k.endswith("_note"):
                    w[f"read.{k}"] += _words(v)
        u, pol = _parts(r.get("note") or "", company)
        for kind in note_kinds(r.get("note") or "", company):
            c[f"note.{kind}"] += 1
        w["note.universal"] += _words(u)
        w["note.purtera"] += _words(pol)
    for r in _rows(payload, "links"):
        if not _is_a_person(r.get("labeler")):
            continue
        rel = str(r.get("relation") or "")
        c[f"link.{rel}"] += 1
        if str(r.get("note") or "").strip():
            c[f"link_note.{rel}"] += 1
            u, pol = _parts(r["note"], company)
            w[f"link_note.{rel}"] += _words(u)
            if pol:
                c["link_note.[purtera]"] += 1
                w["link_note.[purtera]"] += _words(pol)
    for j in _rows(payload, "judgments"):
        if not _trains(j):
            continue
        head = str(j.get("head") or "")
        c[f"card.{head}"] += 1
        if str(j.get("note") or "").strip():
            c[f"card_note.{head}"] += 1
            u, pol = _parts(j["note"], company)
            w[f"card_note.{head}"] += _words(u)
            if pol:
                c["card_note.[purtera]"] += 1
                w["card_note.[purtera]"] += _words(pol)
        if str(j.get("reason") or "").strip():
            c[f"card_reason.{head}"] += 1
            if _is_sentence(j["reason"]):
                c[f"card_reason_text.{head}"] += 1
                w[f"card_reason_text.{head}"] += _words(j["reason"])
        for k, v in (j.get("fields") or {}).items() if isinstance(j.get("fields"), dict) else ():
            if not _empty(v) and v is not False:
                c[f"card_field.{k}"] += 1
        for k, v in j.items():
            if k not in CARD_META and k not in ("head", "verdict", "note", "reason", "fields") \
                    and not _empty(v):
                c[f"card_meta.{k}"] += 1
    for a in _rows(payload, "deal_answers"):
        if not _is_a_person(a.get("labeler")):
            continue
        for k, v in a.items():
            if k not in ANSWER_META and not _empty(v):
                c[f"deal_answer.{k}"] += 1
        u, pol = _parts(a.get("note") or "", company)
        w["deal_answer.note"] += _words(u)
        if pol:
            c["deal_answer.note[purtera]"] += 1
            w["deal_answer.note[purtera]"] += _words(pol)
    for part in ("missed_history", "missed_rows"):
        n = sum(1 for r in _rows(payload, part) if _is_a_person(r.get("labeler")))
        if n:
            c[part] += n
    return c, +w


def _trained_rows(payload: dict, part: str) -> list[dict]:
    """The rows census counted: a row set aside stays as it is, so removing a
    field never un-excludes it (weight_tier exclude, consumer ignore, skip)."""
    return [r for r in _rows(payload, part) if _trains(r)]


def ablate(payload: dict, signal: str) -> None:
    """Remove ``signal`` from ``payload`` in place, on the rows that train."""
    company = str(payload.get("company") or "purtera")
    family, _, name = signal.partition(".")
    if family == "label":
        for r in _trained_rows(payload, "labels"):
            r.pop(name, None)
    elif family == "read":
        for r in _trained_rows(payload, "labels"):
            if isinstance(r.get("reads_set"), dict):
                r["reads_set"].pop(name, None)
    elif family == "note":
        for r in _trained_rows(payload, "labels"):
            if r.get("note"):
                r["note"] = without_kind(r["note"], name, company)
    elif family == "link":
        payload["links"] = [r for r in payload.get("links") or [] if str(r.get("relation") or "") != name]
    elif family == "link_note":
        for r in _rows(payload, "links"):
            if name == "[purtera]":
                if r.get("note"):
                    r["note"] = without_kind(r["note"], "purtera", company)
            elif str(r.get("relation") or "") == name:
                r["note"] = ""
    elif family == "card":
        # The verdict goes; the card stays (links anchored on its line still resolve).
        for j in _trained_rows(payload, "judgments"):
            if str(j.get("head") or "") == name:
                j["verdict"] = ""
    elif family == "card_note":
        for j in _trained_rows(payload, "judgments"):
            if name == "[purtera]":
                if j.get("note"):
                    j["note"] = without_kind(j["note"], "purtera", company)
            elif str(j.get("head") or "") == name:
                j["note"] = ""
    elif family == "card_reason":
        for j in _trained_rows(payload, "judgments"):
            if str(j.get("head") or "") == name:
                j.pop("reason", None)
    elif family == "card_reason_text":
        for j in _trained_rows(payload, "judgments"):
            if str(j.get("head") or "") == name and _is_sentence(j.get("reason")):
                j.pop("reason", None)
    elif family == "card_field":
        for j in _trained_rows(payload, "judgments"):
            if isinstance(j.get("fields"), dict):
                j["fields"].pop(name, None)
    elif family == "card_meta":
        for j in _trained_rows(payload, "judgments"):
            j.pop(name, None)
    elif family == "deal_answer":
        for a in _rows(payload, "deal_answers"):
            if name == "note[purtera]":
                if a.get("note"):
                    a["note"] = without_kind(a["note"], "purtera", company)
            else:
                a.pop(name, None)
    elif signal in ("missed_history", "missed_rows"):
        payload[signal] = []
    else:
        raise ValueError(f"unknown signal {signal}")


# ---------------------------------------------------------------- loading

def approve(payload: dict) -> None:
    """The file's drafts as approved (loop.accept_drafts). A draft that edits a
    saved row (``supersedes``: the saved row's id) replaces it, as runE-data's
    accept_drafts does (8c3ba97); a no-op where the loop already did it."""
    from .loop import accept_drafts  # noqa: PLC0415

    accept_drafts(payload)
    for part in ("labels", "judgments", "deal_answers", "links", "missed_history"):
        rows = payload.get(part)
        if not isinstance(rows, list):
            continue
        gone = {str(r["supersedes"]) for r in rows if isinstance(r, dict) and r.get("supersedes")}
        if gone:
            payload[part] = [r for r in rows if not (isinstance(r, dict) and str(r.get("id")) in gone
                                                     and not r.get("supersedes"))]


def read_payload(path: str, approved: bool) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if approved:
        approve(payload)
    return payload


def payload_parts(payload: dict, schema, max_atoms: int) -> list[Batch]:
    """The loop's parts of one training file (loop.load_deal + training_batches)."""
    from .data import DealExample  # noqa: PLC0415
    from .loop import training_batches  # noqa: PLC0415

    deal = DealExample.from_training_blob(payload, payload["atoms"], deal_id=str(payload["deal_id"]),
                                          company=str(payload.get("company") or "purtera"))
    if payload.get("company_policy"):
        deal.company_policy = str(payload["company_policy"])
    return training_batches(deal, schema, max_atoms=max_atoms)[0]


_SKIP_DIFF = {"deal_id", "notes", "keys"}


def _same(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and bool(torch.equal(a, b))
    return a == b


def batch_diff(a: Batch, b: Batch) -> list[str]:
    """The Batch fields (dict fields by key) that differ."""
    out = []
    for f in dataclasses.fields(Batch):
        if f.name in _SKIP_DIFF:
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y or not _same(x[k], y[k]):
                    out.append(f"{f.name}[{k}]")
        elif f.name == "judged":
            kx = Counter(j.key for j in x) - Counter(j.key for j in y)
            ky = Counter(j.key for j in y) - Counter(j.key for j in x)
            moved = sorted(set(kx) | set(ky))
            if not moved and x != y:
                moved = sorted({j.key for j in set(x) ^ set(y)}) or ["?"]
            out += [f"judged[{k}]" for k in moved]
        elif not _same(x, y):
            out.append(f.name)
    return out


# ---------------------------------------------------------------- loss terms

#: LossWeights fields that are settings of a term, not a term's weight.
NOT_A_TERM = frozenset({"why_hard_negative", "why_swap_cap", "keyword_dropout_p",
                        "keyword_dropout_cap", "near_miss_margin", "link_why_hard_negative",
                        "described_alone_p", "card_cap", "free_h_out", "company_r_grad",
                        "space", "head"})
#: Terms that need a bank or a teacher this audit does not build.
NEVER_ON = frozenset({"clause_use"})


def _term_functions() -> dict[str, Callable]:
    from . import consequence, link_why, losses, why2, why_carry  # noqa: PLC0415

    def get(mod, name):
        return getattr(mod, name, None)

    table = {
        "heads": get(losses, "head_losses"), "numbers": get(losses, "head_losses"),
        "conduct": get(losses, "head_losses"), "relations": get(losses, "relation_loss"),
        "governs": get(losses, "governs_loss"), "why_align": get(losses, "why_losses"),
        "why_sufficiency": get(losses, "why_losses"), "hindsight": get(losses, "hindsight_loss"),
        "variance": get(losses, "hindsight_loss"), "absence": get(losses, "absence_loss"),
        "explanation_only": get(losses, "explanation_only_loss"),
        "why_echo": get(losses, "why_echo_loss"), "pointers": get(losses, "pointer_loss"),
        "entities": get(losses, "entity_loss"), "judgments": get(losses, "judgment_loss"),
        "judgment_rationale": get(losses, "judgment_rationale_loss"),
        "literals": get(losses, "literal_span_loss"), "deal_numbers": get(losses, "deal_number_loss"),
        "negatives": get(losses, "negative_loss"), "near_misses": get(losses, "near_miss_loss"),
        "layout_lock": get(losses, "layout_lock_loss"), "clause_use": get(losses, "clause_use_loss"),
        "changes": get(consequence, "changes_loss"), "claims": get(consequence, "claims_loss"),
        "rule_links": get(consequence, "rule_link_loss"),
        "why_contrast": get(why_carry, "why_contrast_loss"), "why_swap": get(why_carry, "why_swap_loss"),
        "why_alone": get(why_carry, "why_alone_loss"),
        "keyword_dropout": get(why_carry, "keyword_dropout_loss"),
        "link_why_contrast": get(link_why, "link_why_contrast_loss"),
        "link_why_alone": get(link_why, "link_why_alone_loss"),
        "described_alone": get(why2, "described_alone_loss"), "card_alone": get(why2, "card_alone_loss"),
        "card_contrast": get(why2, "card_contrast_loss"), "co_why_contrast": get(why2, "co_why_contrast_loss"),
        "co_why_alone": get(why2, "co_why_alone_loss"), "number_why_alone": get(why2, "number_why_alone_loss"),
        "judgment_why_alone": get(why2, "judgment_why_alone_loss"),
    }
    return {k: v for k, v in table.items() if v is not None}


def where(term: str) -> str:
    fn = _term_functions().get(term, c3_loss)
    try:
        path = Path(inspect.getsourcefile(fn)).resolve().relative_to(REPO)
        return f"{path}:{inspect.getsourcelines(fn)[1]}"
    except (TypeError, OSError, ValueError):
        return "?"


STAGE_ORDER = ("lines", "links", "cards", "numbers", "deal", "open_links")


def run_weights(loss_weight: dict[str, float], head_weight: dict[str, float], stage_steps: str
                ) -> tuple[LossWeights, dict[str, str]]:
    """The weight each term trains at in the run: the largest over the stages
    with steps, each stage as the loop builds it (gold_weights, then the
    table's overrides only where the stage has the term on). Also the first
    stage each term is on."""
    from .loop import gold_weights  # noqa: PLC0415

    steps = {k: float(v) for k, _, v in (s.partition("=") for s in stage_steps.split(",") if s)}
    best: dict[str, float] = {}
    first: dict[str, str] = {}
    last = LossWeights()
    for stage in STAGE_ORDER:
        if not steps.get(stage, 0):
            continue
        w, _ = gold_weights("deal" if stage == "open_links" else stage)
        w = replace(w, **{k: v for k, v in loss_weight.items() if getattr(w, k)})
        last = w
        for f in dataclasses.fields(LossWeights):
            if f.name in NOT_A_TERM:
                continue
            v = float(getattr(w, f.name))
            if v and f.name not in first:
                first[f.name] = stage
            best[f.name] = max(best.get(f.name, 0.0), v)
    union = replace(last, **best, head=dict(head_weight))
    return union, first


def probe_weights(union: LossWeights) -> LossWeights:
    """Every term computed (a term at weight 0 still shows as a consumer)."""
    on = {f.name: (getattr(union, f.name) or 1.0) for f in dataclasses.fields(LossWeights)
          if f.name not in NOT_A_TERM and f.name not in NEVER_ON}
    return replace(union, **on)


def weights_from_script(path: str) -> tuple[dict[str, float], dict[str, float], str]:
    """The loss table, the head weights and the stages of a pod run script
    (runE.sh): ``K=${K:-default}`` defaults, overridden by the environment.
    The table is every ``weights=( ... )`` block and the why2 one (``why2=( ... )``)."""
    text = Path(path).read_text(encoding="utf-8")
    defaults = {m.group(1): m.group(2) for m in re.finditer(r"\b([A-Z_]+)=\$\{\1:-([^}]*)\}", text)}

    def resolve(v: str) -> str:
        def sub(m: re.Match) -> str:
            name = m.group(1) or m.group(3)
            fallback = m.group(2) if m.group(2) is not None else defaults.get(name, "")
            return os.environ.get(name, fallback)
        return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)", sub, v.strip().strip('"'))

    blocks = re.findall(r"^(?:weights|why2)=\((.*?)^\)", text, re.S | re.M)
    if not re.search(r"^weights=\(", text, re.M):
        raise SystemExit(f"{path}: no weights=( ... ) loss table")
    loss = {}
    for line in "\n".join(blocks).splitlines():
        for tok in line.split("#", 1)[0].split():
            k, _, v = tok.partition("=")
            if k and v:
                loss[k] = float(resolve(v))
    heads = {}
    for m in re.finditer(r"--head-weight\s+(\S+?)=(\"?\$[\w{}:-]+\"?|[\d.]+)", text):
        heads[m.group(1)] = float(resolve(m.group(2)))
    sm = re.search(r'^stages="([^"]+)"', text, re.M)
    stages = resolve(sm.group(1)) if sm else "lines=1,links=1,cards=1,numbers=1,deal=0,open_links=1"
    return loss, heads, stages


# ---------------------------------------------------------------- the probe

class Probe:
    """One model, one seeded forward per part, gradients by term."""

    def __init__(self, model, union: LossWeights, device: str = "cpu", seed: int = 0):
        self.model, self.union, self.device, self.seed = model, union, device, seed
        self.probe = probe_weights(union)
        self.params = [p for p in model.parameters() if p.requires_grad]
        model.eval()                      # no dropout: with and without differ by the signal only

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        from .loop import _autocast  # noqa: PLC0415

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        keep: dict = {}
        with _autocast(self.device):
            c3_loss(self.model, batch, self.probe, keep=keep)
        if "parts" not in keep:
            raise SystemExit("losses.c3_loss does not hand back its terms (keep['parts'])")
        return {k: v for k, v in keep["parts"].items() if torch.is_tensor(v) and v.requires_grad}

    def weighted(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(float(getattr(self.union, k, 0.0)) * v.float() for k, v in parts.items())

    def grad_norm(self, t: torch.Tensor) -> float:
        if not torch.is_tensor(t) or not t.requires_grad:
            return 0.0
        gs = torch.autograd.grad(t, self.params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None) or torch.zeros(())))


def build_model(schema, args):
    from .loop import SMALL, release_encoder_top  # noqa: PLC0415
    from .model import C3Model  # noqa: PLC0415
    from .schema import NUMBER, RELATION  # noqa: PLC0415

    torch.manual_seed(args.seed)
    if args.tiny:
        model = C3Model(schema, SMALL)
        if args.init_from:        # a tiny run's own checkpoint (the smoke's)
            from .gates import load_checkpoint  # noqa: PLC0415
            from .readiness import digest  # noqa: PLC0415

            load_checkpoint(args.init_from, model,
                            schema_sha256=digest([dataclasses.asdict(o) for o in schema.opportunities]))
    else:
        from .gates import load_checkpoint  # noqa: PLC0415
        from .loop import live_student, without_heads  # noqa: PLC0415
        from .readiness import digest  # noqa: PLC0415
        from .schema import shape_digest  # noqa: PLC0415

        model = live_student(schema, args.encoder)
        if args.init_from:        # as loop.main: --allow-text-change, the new heads, the schema ref
            known = {}
            if args.init_schema_ref:
                from .gates import schema_digests_at  # noqa: PLC0415

                full, shape = schema_digests_at(args.init_schema_ref)
                known[full] = shape
            load_checkpoint(args.init_from, model,
                            schema_sha256=digest([dataclasses.asdict(o) for o in schema.opportunities]),
                            allow_text_change=True,
                            shape_sha256=shape_digest(without_heads(schema, args.allow_new_heads)),
                            known_shapes=known)
        release_encoder_top(model, args.train_encoder_layers)
    for o in schema.opportunities:      # --free-head all, as runE.sh
        if o.universal and o.kind not in (RELATION, NUMBER) and len(o.answers) > 1:
            model.add_free_head(o.key)
    return model.to(args.device)


# ---------------------------------------------------------------- the audit

@dataclasses.dataclass
class Finding:
    signal: str
    set: str
    status: str = "UNUSED"
    fields: list[str] = dataclasses.field(default_factory=list)
    terms: dict[str, float] = dataclasses.field(default_factory=dict)   # term -> grad share
    share: float = 0.0
    carrier: str = ""
    error: str = ""


def _rank(status: str) -> int:
    return {"OK": 0, "ERROR": 1, "ZERO-GRAD": 2, "UNUSED": 3, "NO-CARRIER": 4}.get(status, 5)


def audit_set(name: str, files: list[tuple[str, bool]], counts_by_file: dict[str, Counter],
              probe: Probe, schema, args, log) -> dict[str, Finding]:
    """Each signal of one set, on its smallest carrier files (up to
    ``args.carriers``), until one shows the signal training."""
    signals = sorted({s for f, _ in files for s in counts_by_file[f]})
    size = {f: os.path.getsize(f) for f, _ in files}
    approved = dict(files)
    plan: dict[str, list[str]] = defaultdict(list)       # carrier -> signals
    carriers_of: dict[str, list[str]] = {}
    for s in signals:
        have = [f for f, _ in files if counts_by_file[f].get(s)]
        have.sort(key=(lambda f: size[f]) if args.carrier_order == "size"
                  else (lambda f: (-counts_by_file[f][s] / size[f], size[f])))
        carriers_of[s] = have[:args.carriers]
    findings = {s: Finding(s, name, status="NO-CARRIER") for s in signals}
    pending = set(signals)
    for round_i in range(args.carriers):
        plan.clear()
        for s in sorted(pending):
            if round_i < len(carriers_of[s]):
                plan[carriers_of[s][round_i]].append(s)
        for path in sorted(plan, key=lambda f: size[f]):
            raw = Path(path).read_text(encoding="utf-8")
            base_payload = json.loads(raw)
            if approved[path]:
                approve(base_payload)
            try:
                base = payload_parts(base_payload, schema, args.max_atoms)
            except Exception as e:  # noqa: BLE001
                log(f"{name}: {Path(path).name} does not featurize: {e}")
                continue
            cache: dict[int, tuple[dict, float]] = {}

            def baseline(k: int):
                if k not in cache:
                    parts = probe.forward(base[k])
                    cache[k] = (parts, probe.grad_norm(probe.weighted(parts)))
                return cache[k]

            started = time.time()
            for s in plan[path]:
                payload = json.loads(raw)
                if approved[path]:
                    approve(payload)
                ablate(payload, s)
                f = Finding(s, name, carrier=Path(path).name)
                try:
                    parts = payload_parts(payload, schema, args.max_atoms)
                except Exception as e:  # noqa: BLE001
                    f.status, f.error = "ERROR", f"featurize refuses the file without it: {e}"[:200]
                    findings[s] = min(findings[s], f, key=lambda x: _rank(x.status))
                    continue
                moved = []
                if len(parts) != len(base):
                    f.fields = ["parts"]
                    moved = list(range(min(len(parts), len(base))))
                else:
                    for k, (a, b) in enumerate(zip(base, parts)):
                        d = batch_diff(a, b)
                        if d:
                            moved.append(k)
                            f.fields += [x for x in d if x not in f.fields]
                if not f.fields:
                    f.status = "UNUSED"
                else:
                    f.status = "ZERO-GRAD"
                    for k in moved[:args.parts_per_signal]:
                        with_parts, total_norm = baseline(k)
                        without = probe.forward(parts[k])
                        diff_total = 0.0
                        changed = []
                        for term, t in with_parts.items():
                            u = without.get(term)
                            dv = float(t.detach().float()) - (float(u.detach().float()) if u is not None else 0.0)
                            if abs(dv) > args.value_tol * max(1.0, abs(float(t.detach().float()))):
                                changed.append(term)
                        for term in sorted(set(without) - set(with_parts)):
                            changed.append(term)
                        weight = {t: float(getattr(probe.union, t, 0.0)) for t in changed}
                        for term in changed:
                            a_t, b_t = with_parts.get(term), without.get(term)
                            delta = (a_t if a_t is not None else 0) - (b_t if b_t is not None else 0)
                            share = (probe.grad_norm(weight[term] * delta) / total_norm
                                     if weight[term] and total_norm else 0.0)
                            f.terms[term] = max(f.terms.get(term, 0.0), share)
                        live = [t for t in changed if weight[t]]
                        if live:
                            d = sum(weight[t] * ((with_parts[t] if t in with_parts else 0)
                                                 - (without[t] if t in without else 0)) for t in live)
                            diff_total = probe.grad_norm(d) / total_norm if total_norm else 0.0
                        f.share = max(f.share, diff_total)
                        del without
                    if f.share > args.grad_tol:
                        f.status = "OK"
                if _rank(f.status) < _rank(findings[s].status) or findings[s].status == "NO-CARRIER":
                    findings[s] = f
                if f.status == "OK":
                    pending.discard(s)
            log(f"{name}: {Path(path).name} {len(plan[path])} signals, {len(base)} parts, "
                f"{time.time() - started:.0f}s")
            cache.clear()
        if not pending:
            break
    return findings


def _fmt_share(x: float) -> str:
    return "0" if not x else f"{100 * x:.3g}%"


def report(sets: list[str], counts: dict[str, Counter], found: dict[str, dict[str, Finding]],
           union: LossWeights, first: dict[str, str], allow: dict[str, str], min_count: int,
           header: str) -> tuple[str, list[str]]:
    signals = sorted({s for c in counts.values() for s in c})
    lines = [header, "",
             "| signal | " + " | ".join(f"n {s}" for s in sets)
             + " | Batch fields (featurize) | consuming term (file:line) | weight in run (from stage) "
             "| grad share | status " + " / ".join(sets) + " |",
             "|---|" + "---:|" * len(sets) + "---|---|---|---:|---|"]
    failing = []
    for s in signals:
        per = [found.get(st, {}).get(s) for st in sets]
        best = min((f for f in per if f), key=lambda f: _rank(f.status), default=None)
        terms: dict[str, float] = {}
        fields: list[str] = []
        for f in per:
            if f:
                for t, v in f.terms.items():
                    terms[t] = max(terms.get(t, 0.0), v)
                fields += [x for x in f.fields if x not in fields]
        term_txt = "<br>".join(f"{t} `{where(t)}`" for t in sorted(terms, key=lambda t: -terms[t])) or "-"
        w_txt = "<br>".join(f"{t}={getattr(union, t, 0):g} ({first.get(t, 'off')})"
                            for t in sorted(terms, key=lambda t: -terms[t])) or "-"
        share = max((f.share for f in per if f), default=0.0)
        status = " / ".join(f.status if f else "-" for f in per)
        if best and best.status == "ERROR":
            status += f" ({best.error[:80]})"
        n = sum(counts[st].get(s, 0) for st in sets)
        bad = best is None or best.status in ("UNUSED", "ZERO-GRAD", "NO-CARRIER")
        mark = ""
        if bad and n >= min_count:
            if s in allow:
                mark = " (allowlisted)"
            else:
                failing.append(s)
                mark = " **FAIL**"
        f_txt = ", ".join(fields[:6]) + (f" +{len(fields) - 6}" if len(fields) > 6 else "") or "-"
        lines.append(f"| `{s}` | " + " | ".join(str(counts[st].get(s, 0)) for st in sets)
                     + f" | {f_txt} | {term_txt} | {w_txt} | {_fmt_share(share)} | {status}{mark} |")
    return "\n".join(lines) + "\n", failing


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def audit(args, union: LossWeights):
    """Counts and findings for every set (the parent folder of each file)."""
    from .schema import load_schema  # noqa: PLC0415

    accepted = set()
    if args.accept_drafts:
        from .loop import read_draft_approvals  # noqa: PLC0415
        accepted = set(read_draft_approvals(args.accept_drafts))
    prefixes = [str(Path(x).expanduser().resolve()) for x in args.approve]
    files: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for f in args.train:
        name = Path(f).name
        if not re.fullmatch(r"\d{6}(@[a-z_]+)?\.json", name):
            continue
        full = str(Path(f).resolve())
        ok = name in accepted or any(full.startswith(x) for x in prefixes)
        files[Path(f).resolve().parent.name].append((full, ok))
    sets = list(files)
    only = re.compile(args.only) if args.only else None
    counts: dict[str, Counter] = {}
    words: dict[str, Counter] = {}
    by_file: dict[str, Counter] = {}
    for st in sets:
        c, wc = Counter(), Counter()
        for f, ok in files[st]:
            payload = read_payload(f, ok)
            if payload.get("skip") or not payload.get("atoms"):
                by_file[f] = Counter()
                continue
            fc, fw = census(payload)
            if only:
                fc = Counter({k: v for k, v in fc.items() if only.search(k)})
                fw = Counter({k: v for k, v in fw.items() if only.search(k)})
            by_file[f] = fc
            c.update(fc)
            wc.update(fw)
        counts[st], words[st] = c, wc
        log(f"{st}: {len(files[st])} files, {len(c)} signals, {sum(c.values())} rows, "
            f"{sum(wc.values())} reasoning words")

    schema = load_schema()
    model = build_model(schema, args)
    probe = Probe(model, union, args.device, args.seed)
    found = {st: audit_set(st, files[st], by_file, probe, schema, args, log) for st in sets}
    desc = {}
    if args.desc_parts:
        pool = []
        for st in sets:
            for f, ok in sorted(files[st], key=lambda x: os.path.getsize(x[0]))[:args.desc_parts]:
                payload = read_payload(f, ok)
                if payload.get("skip") or not payload.get("atoms"):
                    continue
                try:
                    parts = payload_parts(payload, schema, args.max_atoms)
                except Exception:  # noqa: BLE001
                    continue
                pool.append(max(parts, key=lambda b: (sum(b.labeled), len(b.judged))))
        desc = desc_reach(probe, pool)
        log(f"descriptions: {sum(1 for v in desc.values() if v > 0)}/{len(desc)} questions reach a loss "
            f"on {len(pool)} parts")
    src = f"{len(args.train)} files; approved: {', '.join(args.approve) or args.accept_drafts or 'none'}"
    return sets, counts, words, found, desc, src


def desc_reach(probe: Probe, batches: list[Batch]) -> dict[str, float]:
    """Opportunity key -> gradient norm of the step's weighted loss on its
    encoded question and answer descriptions (summed over ``batches``): the
    description text reaches a loss when it is nonzero."""
    model = probe.model
    orig = model.describe
    reach: dict[str, float] = defaultdict(float)
    try:
        for b in batches:
            base = orig()
            leaves = {k: (q.detach().clone().requires_grad_(True), a.detach().clone().requires_grad_(True))
                      for k, (q, a) in base.items()}
            model.describe = lambda leaves=leaves: leaves
            total = probe.weighted(probe.forward(b))
            keys = list(leaves)
            tensors = [t for k in keys for t in leaves[k]]
            gs = torch.autograd.grad(total, tensors, allow_unused=True)
            for n, k in enumerate(keys):
                g = [x for x in gs[2 * n:2 * n + 2] if x is not None]
                reach[k] += float(sum((x.float() ** 2).sum() for x in g) ** 0.5) if g else 0.0
    finally:
        model.describe = orig
    return dict(reach)


def description_words(schema) -> dict[str, int]:
    """Opportunity key -> words of its question and answer descriptions (schema.texts order)."""
    texts, out, i = schema.texts(), {}, 0
    for o in schema.opportunities:
        n = 1 + len(o.answers)
        out[o.key] = sum(_words(t) for t in texts[i:i + n])
        i += n
    return out


def text_kind(signal: str, schema) -> str | None:
    """The reasoning-text kind a word-carrying signal belongs to."""
    family, _, name = signal.partition(".")
    if signal == "note.universal":
        return "label WHY (universal)"
    if signal == "note.purtera":
        return "label [purtera] line"
    if signal == "link_note.[purtera]":
        return "link note [purtera]"
    if family == "link_note":
        return "link note (universal)"
    if signal == "card_note.[purtera]":
        return "card note [purtera]"
    if family == "card_note":
        by_key = schema.by_key()
        opp = by_key.get(f"jdg:{name}")
        if opp is None and name in schema.judgment_aliases:
            opp = by_key.get(schema.judgment_aliases[name][0])
        return f"card note ({opp.size if opp is not None else 'no head'})"
    if family == "card_reason_text":
        return "reason-only note"
    if family == "read" or signal == "label.rejected_reads":
        return "field note"
    if signal == "deal_answer.note":
        return "deal answer note"
    if signal == "deal_answer.note[purtera]":
        return "deal answer [purtera]"
    return None


def coverage(sets, words, found, desc, schema) -> list[tuple[str, int, int]]:
    """(kind, words, words whose signal reaches a weighted loss) per text kind."""
    tot: Counter = Counter()
    cov: Counter = Counter()
    for st in sets:
        for s, n in words.get(st, {}).items():
            kind = text_kind(s, schema)
            if kind is None or not n:
                continue
            tot[kind] += n
            f = found.get(st, {}).get(s)
            if f is not None and f.status == "OK":
                cov[kind] += n
    if desc:
        for k, n in description_words(schema).items():
            tot["description"] += n
            if desc.get(k, 0.0) > 0:
                cov["description"] += n
    return [(k, tot[k], cov[k]) for k in sorted(tot, key=lambda k: -tot[k])]


def coverage_report(rows, min_kind_words: int, min_coverage: float) -> tuple[str, list[str]]:
    total = sum(n for _, n, _ in rows)
    covered = sum(c for _, _, c in rows)
    share = covered / total if total else 1.0
    fails = []
    lines = ["", "## Reasoning text that reaches a teacher-free loss", "",
             "Words by text kind; a kind's words count as covered where their signal is OK in that set "
             "(the step's weighted gradient moves when the text is removed). Descriptions: the question "
             "and answer texts whose encoding gets a nonzero gradient from the step's loss.", "",
             "| kind | words | covered | share | |", "|---|---:|---:|---:|---|"]
    for k, n, c in rows:
        mark = ""
        if min_kind_words and n >= min_kind_words and c == 0:
            fails.append(f"{k}: {n} words reach no loss")
            mark = "**FAIL**"
        lines.append(f"| {k} | {n} | {c} | {100 * c / n:.1f}% | {mark} |" if n else f"| {k} | 0 | 0 | - | |")
    tmark = ""
    if min_coverage and share < min_coverage:
        fails.append(f"total coverage {100 * share:.1f}% < {100 * min_coverage:.0f}%")
        tmark = "**FAIL**"
    lines.append(f"| **total** | {total} | {covered} | {100 * share:.1f}% | {tmark} |")
    return "\n".join(lines) + "\n", fails


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", nargs="+", default=[], help="training files (NNNNNN[@view].json)")
    p.add_argument("--render", nargs="+", default=[],
                   help="no audit: one report from earlier --json outputs (sets run apart)")
    p.add_argument("--approve", nargs="*", default=[],
                   help="path prefixes whose drafts count as approved (as --accept-drafts would)")
    p.add_argument("--accept-drafts", default="", help="the owner's approval manifest (loop.py)")
    p.add_argument("--tiny", action="store_true", help="the CPU stand-in model (plumbing)")
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--init-from", default="")
    p.add_argument("--allow-new-heads", default="")
    p.add_argument("--init-schema-ref", default="")
    p.add_argument("--train-encoder-layers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-atoms", type=int, default=512)
    p.add_argument("--loss-weight", action="append", default=[])
    p.add_argument("--head-weight", action="append", default=[])
    p.add_argument("--stage-steps", default="")
    p.add_argument("--weights-from", default="", help="read the loss table from a pod script (runE.sh)")
    p.add_argument("--carriers", type=int, default=2, help="carrier files tried per signal and set")
    p.add_argument("--carrier-order", choices=("size", "density"), default="density",
                   help="carriers by smallest file, or by most rows of the signal per byte")
    p.add_argument("--parts-per-signal", type=int, default=2)
    p.add_argument("--value-tol", type=float, default=1e-7)
    p.add_argument("--grad-tol", type=float, default=1e-6)
    p.add_argument("--min-count", type=int, default=0, help="gate: fail on a signal with this many rows")
    p.add_argument("--allowlist", default="", help="JSON {signal: reason} exempt from the gate")
    p.add_argument("--only", default="", help="regex: audit only matching signals")
    p.add_argument("--desc-parts", type=int, default=3,
                   help="files per set whose richest part measures the descriptions' gradient (0: skip)")
    p.add_argument("--min-kind-words", type=int, default=0,
                   help="gate: fail when a text kind with this many words reaches no loss")
    p.add_argument("--min-coverage", type=float, default=0.0,
                   help="gate: fail when less than this share of reasoning words reaches a loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    args = p.parse_args(argv)
    if not args.train and not args.render:
        p.error("--train or --render")
    loss_weight, head_weight, stages = {}, {}, args.stage_steps
    if args.weights_from:
        loss_weight, head_weight, script_stages = weights_from_script(args.weights_from)
        stages = stages or script_stages
    for spec in args.loss_weight:
        k, _, v = spec.partition("=")
        loss_weight[k] = float(v)
    for spec in args.head_weight:
        k, _, v = spec.rpartition("=")
        head_weight[k] = float(v)
    stages = stages or "lines=1,links=1,cards=1,numbers=1,deal=0,open_links=1"
    union, first = run_weights(loss_weight, head_weight, stages)

    allow = {}
    if args.allowlist:
        allow = {k: v for k, v in json.loads(Path(args.allowlist).read_text()).items()
                 if not k.startswith("_")}
    if args.render:
        sets, counts, words, found, desc = [], {}, {}, {}, {}
        for path in args.render:
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
            for st in blob["sets"]:
                # A set seen again (a second pass on some signals): the best finding wins.
                if st not in sets:
                    sets.append(st)
                c = counts.setdefault(st, Counter())
                for k, v in blob["counts"][st].items():
                    c[k] = max(c[k], v)
                wc = words.setdefault(st, Counter())
                for k, v in (blob.get("words") or {}).get(st, {}).items():
                    wc[k] = max(wc[k], v)
                fs = found.setdefault(st, {})
                for k, v in blob["findings"][st].items():
                    f = Finding(**v)
                    if k not in fs or _rank(f.status) < _rank(fs[k].status):
                        fs[k] = f
            for k, v in (blob.get("desc") or {}).items():
                desc[k] = max(desc.get(k, 0.0), v)
        sources = ", ".join(json.loads(Path(x).read_text())["sources"] for x in args.render)
    else:
        sets, counts, words, found, desc, sources = audit(args, union)

    head = (f"Signal audit: {'tiny' if args.tiny else 'live'} model, one step at init, "
            f"{args.max_atoms} lines per part, stages {stages}; weights from "
            f"{args.weights_from or 'the command line'} (head weights {head_weight}). "
            f"Files: {sources}. "
            f"Status per set: {' / '.join(sets)}. A gate failure is a signal with >= {args.min_count} "
            "rows that no set trains, and not allowlisted.")
    text, failing = report(sets, counts, found, union, first, allow, args.min_count, head)
    from .schema import load_schema  # noqa: PLC0415

    cov_rows = coverage(sets, words, found, desc, load_schema())
    cov_text, cov_fails = coverage_report(cov_rows, args.min_kind_words, args.min_coverage)
    text = text + cov_text
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps({
            "sets": sets, "sources": sources, "counts": {k: dict(v) for k, v in counts.items()},
            "findings": {st: {s: dataclasses.asdict(f) for s, f in fs.items()} for st, fs in found.items()},
            "weights": {f.name: getattr(union, f.name) for f in dataclasses.fields(LossWeights)
                        if f.name not in ("space", "head")},
            "words": {k: dict(v) for k, v in words.items()}, "desc": desc,
            "coverage": [{"kind": k, "words": n, "covered": c} for k, n, c in cov_rows],
            "failing": failing, "coverage_failing": cov_fails}, indent=1), encoding="utf-8")
    statuses = Counter(min((found[st].get(s) for st in sets if found[st].get(s)), key=lambda f: _rank(f.status)).status
                       for s in {s for c in counts.values() for s in c})
    log("signal audit: " + " ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    total_w = sum(n for _, n, _ in cov_rows)
    log(f"reasoning words reaching a loss: {sum(c for _, _, c in cov_rows)}/{total_w}")
    if cov_fails:
        log("signal audit FAILED (reasoning text): " + "; ".join(cov_fails))
        if not (args.min_count and failing):
            return 1
    if args.min_count and failing:
        log(f"signal audit FAILED: {len(failing)} signals with >= {args.min_count} rows train nothing "
            f"and are not on the allowlist: {' '.join(failing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
