"""Propose a deal's Division from the WORK, not from a word list (PUR-29).

PMs label each deal with a Division. The values they actually type are messy:
"Network" (29), "Networking" (17), "EUC" (16), "IMAC" (16), "Camera Install"
(8), "AV" (6), "Network/LV" (6), "Cable" (6), "Staff Aug" (4), "Cabling" (3),
"palletize" (3). "Network"/"Networking" and "Cable"/"Cabling" are the same
division spelled two ways.

This module does two things, both learned from labelled examples rather than
hand-written rules:

1. **Label clustering.** Spelling variants collapse when they are close as
   STRINGS (same token count, each token a stem-prefix of the other) AND close
   in what WORK they label (cosine of their work-signal profiles). Neither
   alone is enough: "Network/LV" shares a prefix with "Network" but has a
   different token count and is kept apart unless a PM merges it; two labels
   that happen to cover similar work but are spelled differently are distinct
   divisions. No synonym table.

2. **Proposal.** A multinomial naive Bayes over work signals of the work order
   (the object, unit and action words of each work line, plus object bigrams
   and a device-count bucket) scores each division. The model ABSTAINS -- returns
   ``division=None`` with a reason -- when the evidence is thin (no known
   signal), when the best division is not confident, or when two divisions are
   close (the work could belong to either).

Off by default: ``SOWSMITH_DIVISION_PROPOSAL=1`` enables it, and
``SOWSMITH_DIVISION_LABELS`` points at a local labelled JSON file
(``[{"work_order": {...}, "division": "..."}]``). Without labels the proposer
abstains with ``reason="no_model"``.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

FLAG = "SOWSMITH_DIVISION_PROPOSAL"
LABELS_ENV = "SOWSMITH_DIVISION_LABELS"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def enabled() -> bool:
    return os.environ.get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


# ── label normalisation and clustering ──────────────────────────────────────


def normalize_label(label: str) -> str:
    """Case, whitespace and separator folding only -- no vocabulary."""
    s = str(label or "").strip().lower()
    s = re.sub(r"\s*([/&+,-])\s*", r"\1", s)
    return re.sub(r"\s+", " ", s)


def _label_tokens(norm: str) -> list[str]:
    return _TOKEN_RE.findall(norm)


def _stem_close(a: str, b: str, *, max_extra: int = 3, min_ratio: float = 0.8) -> bool:
    """Two tokens are inflections of one word: long shared prefix, short tail."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > max_extra:
        return False
    lcp = 0
    for x, y in zip(a, b):
        if x != y:
            break
        lcp += 1
    return lcp / min(len(a), len(b)) >= min_ratio


def string_close(a: str, b: str) -> bool:
    ta, tb = _label_tokens(normalize_label(a)), _label_tokens(normalize_label(b))
    return bool(ta) and len(ta) == len(tb) and all(_stem_close(x, y) for x, y in zip(ta, tb))


def _cosine(u: Counter, v: Counter) -> float:
    if not u or not v:
        return 0.0
    dot = sum(c * v.get(k, 0) for k, c in u.items())
    nu = math.sqrt(sum(c * c for c in u.values()))
    nv = math.sqrt(sum(c * c for c in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0


def cluster_labels(
    examples: Iterable[tuple[str, list[str]]], *, min_profile_cosine: float = 0.5
) -> dict[str, str]:
    """Map each normalised label to its cluster's canonical (most frequent) spelling.

    ``examples`` are ``(raw_label, work_features)`` pairs.
    """
    freq: Counter = Counter()
    surface: dict[str, Counter] = defaultdict(Counter)
    profile: dict[str, Counter] = defaultdict(Counter)
    for raw, feats in examples:
        n = normalize_label(raw)
        if not n:
            continue
        freq[n] += 1
        surface[n][str(raw).strip()] += 1
        profile[n].update(feats)

    labels = sorted(freq, key=lambda k: (-freq[k], k))
    parent = {k: k for k in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if find(a) == find(b):
                continue
            if string_close(a, b) and _cosine(profile[a], profile[b]) >= min_profile_cosine:
                parent[find(b)] = find(a)

    groups: dict[str, list[str]] = defaultdict(list)
    for k in labels:
        groups[find(k)].append(k)
    out: dict[str, str] = {}
    for members in groups.values():
        top = max(members, key=lambda m: (freq[m], -len(m)))
        canonical = surface[top].most_common(1)[0][0]
        for m in members:
            out[m] = canonical
    return out


# ── work signals ────────────────────────────────────────────────────────────


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _words(text: Any) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(str(text or "").lower()) if len(t) > 1 and not t.isdigit()]


def work_features(work_order: dict) -> list[str]:
    """Signals of the WORK: what is acted on, in what unit, by which action."""
    feats: list[str] = []
    lines = work_order.get("work_lines") if isinstance(work_order, dict) else None
    for line in lines or []:
        if isinstance(line, str):
            line = {"work": line}
        if not isinstance(line, dict):
            continue
        obj = _words(line.get("object"))
        feats += [f"obj:{w}" for w in obj]
        feats += [f"obj2:{a}_{b}" for a, b in zip(obj, obj[1:])]
        feats += [f"unit:{w}" for w in _words(line.get("unit"))]
        feats += [f"work:{w}" for w in _words(line.get("work"))]
        count = line.get("count")
        if isinstance(count, (int, float)) and not isinstance(count, bool) and count > 0:
            feats.append(f"countbucket:{min(int(math.log10(count)), 3)}")
    for key in ("after_hours", "no_onsite_hands", "customer_supplies_equipment"):
        if isinstance(work_order, dict) and work_order.get(key):
            feats.append(f"fact:{key}")
    return feats


# ── model ───────────────────────────────────────────────────────────────────


@dataclass
class Proposal:
    division: str | None
    confidence: float
    reason: str
    candidates: list[tuple[str, float]] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "division": self.division,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "candidates": [(d, round(p, 3)) for d, p in self.candidates],
            "signals": self.signals,
            "abstained": self.division is None,
        }


@dataclass
class DivisionProposer:
    min_confidence: float = 0.8
    min_margin: float = 0.5
    split_line_confidence: float = 0.7
    min_known_signals: int = 2
    alpha: float = 0.5
    label_map: dict[str, str] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    counts: dict[str, Counter] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)

    def canonical(self, raw_label: str) -> str:
        n = normalize_label(raw_label)
        return self.label_map.get(n, str(raw_label).strip())

    def fit(self, examples: list[dict]) -> "DivisionProposer":
        pairs = [
            (str(e["division"]), work_features(e.get("work_order") or {}))
            for e in examples
            if e.get("division")
        ]
        self.label_map = cluster_labels(pairs)
        per: dict[str, Counter] = defaultdict(Counter)
        docs: Counter = Counter()
        for raw, feats in pairs:
            d = self.canonical(raw)
            docs[d] += 1
            per[d].update(feats)
        n = sum(docs.values()) or 1
        self.priors = {d: c / n for d, c in docs.items()}
        self.counts = dict(per)
        self.totals = {d: sum(c.values()) for d, c in per.items()}
        self.vocab = {f for c in per.values() for f in c}
        return self

    def propose(self, work_order: dict) -> Proposal:
        prop = self._propose(work_order)
        lines = work_order.get("work_lines") if isinstance(work_order, dict) else None
        if prop.division is None or not isinstance(lines, list) or len(lines) < 2:
            return prop
        # Work that splits: lines that each confidently belong to different
        # divisions mean the deal could be either -- abstain, name both.
        line_divs = set()
        for line in lines:
            sub = self._propose({"work_lines": [line]})
            if sub.division is None and sub.candidates and sub.candidates[0][1] >= self.split_line_confidence:
                line_divs.add(sub.candidates[0][0])
            elif sub.division is not None:
                line_divs.add(sub.division)
        if len(line_divs) > 1:
            names = " vs ".join(sorted(line_divs))
            return Proposal(None, prop.confidence, f"ambiguous: work splits across {names}",
                            prop.candidates, prop.signals)
        return prop

    def _propose(self, work_order: dict) -> Proposal:
        if not self.priors:
            return Proposal(None, 0.0, "no_model")
        feats = work_features(work_order)
        known = [f for f in feats if f in self.vocab]
        if len(known) < self.min_known_signals:
            return Proposal(None, 0.0, "no_work_signal", signals=known)
        v = len(self.vocab)
        logp: dict[str, float] = {}
        for d, prior in self.priors.items():
            c, t = self.counts.get(d, Counter()), self.totals.get(d, 0)
            logp[d] = math.log(prior) + sum(
                math.log((c.get(f, 0) + self.alpha) / (t + self.alpha * v)) for f in known
            )
        # Temper by signal count so a long work order is not falsely certain.
        scale = 1.0 / max(1.0, math.sqrt(len(known)))
        top = max(logp.values())
        exp = {d: math.exp((lp - top) * scale) for d, lp in logp.items()}
        z = sum(exp.values())
        ranked = sorted(((d, e / z) for d, e in exp.items()), key=lambda kv: -kv[1])
        best, p1 = ranked[0]
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        top_signals = sorted(
            set(known),
            key=lambda f: -(self.counts.get(best, Counter()).get(f, 0) / max(1, self.totals.get(best, 1))),
        )[:8]
        cands = ranked[:3]
        if p1 - p2 < self.min_margin:
            return Proposal(None, p1, f"ambiguous: {best} vs {ranked[1][0]}", cands, top_signals)
        if p1 < self.min_confidence:
            return Proposal(None, p1, "low_confidence", cands, top_signals)
        return Proposal(best, p1, "proposed_from_work_signals", cands, top_signals)


def load_examples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("examples", data) if isinstance(data, dict) else data


_CACHED: tuple[str, DivisionProposer] | None = None


def default_proposer() -> DivisionProposer:
    global _CACHED
    path = os.environ.get(LABELS_ENV, "").strip()
    if not path or not os.path.exists(path):
        return DivisionProposer()
    if _CACHED and _CACHED[0] == path:
        return _CACHED[1]
    model = DivisionProposer().fit(load_examples(path))
    _CACHED = (path, model)
    return model


def propose_division(work_order: dict) -> dict[str, Any] | None:
    """Flag-gated entry point. None when disabled; otherwise a proposal dict."""
    if not enabled():
        return None
    return default_proposer().propose(work_order).as_dict()


__all__ = [
    "DivisionProposer",
    "Proposal",
    "cluster_labels",
    "enabled",
    "load_examples",
    "normalize_label",
    "propose_division",
    "string_close",
    "work_features",
]
