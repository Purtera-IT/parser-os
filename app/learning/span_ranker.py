"""Which words decided this label?

The labeler answers it on every card -- 159 pointers on 010288 alone, each
tagged with the hint that says why it mattered. That is span supervision, and
unlike a rationale it can run at inference: pointing at evidence needs the
document, which is always there, and the note, which is not.

Only for the pointers that are spans. 72 of the 159 are words in the atom or
the context printed beside it; 29 name another surface, and 56 name a
STRUCTURED FIELD -- who sent it, what kind of document it is -- that no reading
of the page will find text for. All 159 were being emitted for a span head to
locate, which is an instruction to produce words that are not there, and that
is how a span head learns to invent one. `pointer_kind` is the triage, and
`human_labels` sends each to the head that can learn it.

This RANKS candidates rather than generating text, and the difference is the
point: a ranker can only return words that exist. It cannot invent a reason,
which is the failure mode that makes a generated explanation worse than none --
fluent, confident and wrong, laundering an error into something that reads
authoritative. A wrong ranked span is visible at a glance, because a person can
see the words it chose and the words it should have.

    "exclusion, because of 'but not the relay to the lock'"

Three deterministic halves and a model that is almost an afterthought:

  pointer_kind    which head can learn a pointer at all.
  candidates_for  every span the answer could be. Clauses of the atom, the
                  list heading above it, the section it sits in.
  features_for    what distinguishes the deciding span from its neighbours.

`fit` is a linear ranker over those features and can be swapped for anything;
the three above decide whether it has anything to learn from. Measured on
010288: **precision@1 95.1%, atoms held out, against 80.3% for always returning
the whole line.** The two things that mattered were not the model. One pointer
must claim one candidate (see `training_pairs`), and the comparison must be
between two candidates of the SAME atom (see `pairwise_dataset`) -- a pointwise
classifier learns that an atom's own words usually win, which is true across the
corpus and false on every bill-of-materials line, where the heading above the
part name is the only thing that says who supplies it.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

#: Clause boundaries a person would read as a break. Kept conservative: an
#: over-split candidate set buries the true span among fragments, and the
#: deciding words are usually a clause, not a word.
_SPLIT = re.compile(r"(?<=[.;:?!])\s+|\s+--\s+|\s+—\s+|,\s+(?=but|and not|not\b)")

#: Items within one clause. Only applied to clauses that already look like a
#: list, so ordinary prose with a comma in it is not shredded.
_LIST_ITEM = re.compile(r",\s+(?:and\s+)?|\s+and\s+(?=[A-Z0-9])")

#: The shortest span worth pointing at. Below this a "span" is a word, and a
#: word is rarely the reason.
MIN_SPAN = 8
#: A cap on GENERATED clause fragments, so one runaway sentence cannot flood the
#: candidate set. It never applies to the atom's own body: the whole line is
#: always a legitimate answer, and clipping it at 220 cost two atoms on 010288
#: every candidate they had.
MAX_SPAN = 220

#: Hints whose pointer is a rendering of a STRUCTURED FIELD, not words in a
#: document. "who_said_it" is the envelope -- sender, affiliation, direction --
#: and "doc_type" is provenance. The labeler is naming the field that decided
#: the label, and no amount of reading the page will find that text on it. On
#: 010288 these are 56 of 159 pointers, and every one of them was being emitted
#: as a span for a head to locate. They are attribution, and they have their own
#: row.
FIELD_HINTS = frozenset({"who_said_it", "doc_type"})


def pointer_kind(ref: dict[str, Any], label: dict[str, Any] | None = None) -> str:
    """Where a labeler's pointer lives, and therefore which head can learn it.

    Three answers, decided by the pointer itself rather than by guessing at its
    text:

      ``span``        words in this atom or the context printed beside it. The
                      ranker below can find them.
      ``other_atom``  an ``atomId``: the deciding words are on another surface.
                      Real evidence, but retrieval, not extraction -- asking a
                      span head to produce text that is not in front of it
                      teaches it to invent.
      ``field``       the envelope or the document type. Structured, already
                      known at inference, and not on the page.
      ``prose``       the labeler describing rather than selecting -- "the two
                      headings in dispute: ..." names two headings and quotes
                      neither. Only distinguishable by looking, so this answer
                      needs `label`; without it a prose pointer reads as a span.

    The hint gives the intent and the words give the truth, and they disagree
    on three of 010288's pointers. Both callers ask the same question here so
    that the corpus and the ranker cannot drift apart on the answer.
    """
    hint = str(ref.get("hint") or "").strip()
    if hint in FIELD_HINTS:
        return "field"
    if ref.get("atomId"):
        return "other_atom"
    if label is not None and not _locatable(_clean(ref.get("text")), label):
        return "prose"
    return "span"


def _locatable(want: str, label: dict[str, Any]) -> bool:
    """Are these words on offer among the atom's candidates at all?"""
    if len(want) < MIN_SPAN:
        return False
    cands = candidates_for(label)
    return bool(cands) and max(_overlap(c.text, want) for c in cands) >= MIN_OVERLAP


def best_locatable(want: str, label: dict[str, Any], prompt: str) -> str | None:
    """The closest candidate that the given prompt actually contains.

    For a pointer the labeler paraphrased. Returns None when nothing on offer
    is close enough, which means it was prose after all.
    """
    norm = " ".join(prompt.split()).lower()
    best, score = None, 0.0
    for c in candidates_for(label):
        if " ".join(c.text.split()).lower() not in norm:
            continue
        s = _overlap(c.text, want)
        if s > score:
            best, score = c.text, s
    return best if score >= MIN_OVERLAP else None


@dataclass(frozen=True)
class Candidate:
    text: str
    #: Where it came from: the atom's own words, the list heading above it, the
    #: section, a neighbour. The hint the labeler would have chosen.
    source: str
    #: 0-based position among the atom's own clauses; -1 for context spans.
    index: int = -1


def _clean(s: str) -> str:
    return " ".join(str(s or "").split())


def candidates_for(label: dict[str, Any]) -> list[Candidate]:
    """Every span the deciding evidence could be, for one labelled atom.

    The answer is nearly always a clause of the atom itself, but not always:
    "Provided by us:" is a heading, and it is what decides `supplier` on nine
    lines that never say who supplies them.
    """
    out: list[Candidate] = []
    seen: set[str] = set()

    def add(text: str, source: str, index: int = -1) -> None:
        t = _clean(text)
        if not (MIN_SPAN <= len(t) <= MAX_SPAN):
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(Candidate(t, source, index))

    body = _clean(label.get("text"))
    # Unconditionally: the whole atom is always an answer the labeler may have
    # given, however long it ran.
    if len(body) >= MIN_SPAN and body.lower() not in seen:
        seen.add(body.lower())
        out.append(Candidate(body, "own_words", 0))
    for i, part in enumerate(_SPLIT.split(body), start=1):
        add(part, "own_words", i)
        # A list is a clause a person reads as several answers. "PC with Access
        # Control Software, USB Cable, RS232 to USB converter and Relay" was
        # one candidate, and the labeler had pointed at the first item.
        if part.count(",") >= 2:
            for item in _LIST_ITEM.split(part):
                add(item.strip(" .;:"), "list_item", i)

    for head in (label.get("lead_in") or []):
        add(head, "lead_in")
    for sec in (label.get("section") or []):
        add(sec, "section")
    for n in (label.get("neighbors_above") or [])[:2]:
        add(n, "neighbor_above")
    for n in (label.get("neighbors_below") or [])[:2]:
        add(n, "neighbor_below")
    if label.get("table_ref"):
        add(str(label["table_ref"]), "doc_type")
    return out


#: Words that make a clause decisive rather than descriptive. Domain-neutral on
#: purpose: these are the shapes of a reason, not a vocabulary of one industry.
_NEGATION = re.compile(r"\b(not|never|except|exclud|without|no longer|but)\b", re.I)
_MODAL = re.compile(r"\b(must|shall|need|require|cannot|can't|should|will)\b", re.I)
_ATTRIB = re.compile(r"\b(provided by|supplied by|per |defer to|note that|says|"
                     r"my understanding|intend)\b", re.I)
_NUMBER = re.compile(r"\d")
_QUOTED = re.compile(r"[\"'“]")

#: A line is a claim when something in it does work: a verb, a modal, a
#: negation, an attribution. A bill-of-materials line has none -- it is a noun
#: phrase, and whatever it was labelled for was decided somewhere else.
_VERBISH = re.compile(r"\b(is|are|was|were|be|has|have|had|do|does|did|can|"
                      r"we|i|you|they|it)\b|\b\w+(?:ed|ing|es|s)\b\s+(?:by|to|for|the|a|an)\b",
                      re.I)


def _says_nothing(body: str) -> bool:
    """True when the atom is a bare label: no verb, no modal, no attribution."""
    if len(body.split()) > 12:
        return False
    return not (_VERBISH.search(body) or _MODAL.search(body)
                or _ATTRIB.search(body) or _NEGATION.search(body))


def features_for(cand: Candidate, label: dict[str, Any]) -> list[float]:
    """What separates the deciding span from the clause beside it.

    Deliberately shallow and readable. A feature a person cannot argue with is
    a feature a person cannot debug, and this head's whole value is that its
    mistakes are obvious.
    """
    t = cand.text
    body = _clean(label.get("text"))
    silent = _says_nothing(body)
    is_context = cand.source in ("lead_in", "section", "doc_type") or cand.source.startswith("neighbor")
    return [
        1.0 if cand.source == "own_words" else 0.0,
        1.0 if cand.source == "lead_in" else 0.0,
        1.0 if cand.source == "section" else 0.0,
        1.0 if cand.source.startswith("neighbor") else 0.0,
        1.0 if cand.source == "doc_type" else 0.0,
        # a clause, not the whole line: the reason is usually a PART
        1.0 if (cand.index > 0 and t != body) else 0.0,
        min(len(t) / 120.0, 2.0),
        len(t.split()) / 20.0,
        1.0 if _NEGATION.search(t) else 0.0,
        1.0 if _MODAL.search(t) else 0.0,
        1.0 if _ATTRIB.search(t) else 0.0,
        1.0 if _NUMBER.search(t) else 0.0,
        1.0 if _QUOTED.search(t) else 0.0,
        1.0 if t.rstrip().endswith(":") else 0.0,
        # how much of the atom this span accounts for
        len(t) / max(len(body), 1),
        # Does the ATOM say anything on its own? "Mag Lock Cable" is a part
        # name. It carries no verb, no attribution and no negation, and it
        # cannot be the reason for a `supplier` reading -- the heading above it
        # is. Every error the first version made was this shape: it chose the
        # atom's own words because own_words usually wins, on lines whose own
        # words answer nothing.
        1.0 if silent else 0.0,
        # ...and the interaction, written out. A linear model cannot find it on
        # its own, and stating it keeps the head arguable: it looked up because
        # the line said nothing and the words above it said who.
        #
        # On 010288 this separates cleanly: of the 61 rankable atoms, 11 are a
        # silent line under an attributive heading and 10 of those 11 are
        # decided by the heading; the other 50 are decided by their own words.
        1.0 if (silent and is_context and _ATTRIB.search(t)) else 0.0,
        1.0 if (silent and cand.source == "own_words") else 0.0,
    ]


FEATURE_NAMES = (
    "is_own_words", "is_lead_in", "is_section", "is_neighbor", "is_doc_type",
    "is_partial_clause", "len_scaled", "words_scaled", "has_negation",
    "has_modal", "has_attribution", "has_number", "has_quote", "ends_colon",
    "share_of_atom", "atom_says_nothing", "silent_under_a_heading", "silent_and_own",
)


def training_pairs(label: dict[str, Any]) -> list[tuple[Candidate, int]]:
    """Candidates for one atom, marked against the spans the labeler pointed at.

    One pointer claims ONE candidate: the closest words on offer. Containment
    was the first rule and it was wrong twice over. The neighbour above the
    exclusion quotes it in parentheses, so it contained the gold span and was
    marked gold itself -- four candidates, four positives, nothing to rank. And
    a threshold strict enough to stop that rejected a genuine 140-character
    selection out of a 248-character line, for being 56% of it.

    An argmax needs no threshold. The neighbour loses to the clause because it
    carries twenty words the pointer does not, and the long prefix wins its
    line because nothing else covers it.

    Only pointers of kind ``span`` are gold. A pointer at the envelope or at
    another document is real supervision for a different head, and scoring this
    one against it would count a correct abstention as a miss.
    """
    cands = candidates_for(label)
    gold: set[int] = set()
    for ref in (label.get("hint_refs") or []):
        if not isinstance(ref, dict) or pointer_kind(ref, label) != "span":
            continue
        want = _clean(ref.get("text"))
        if len(want) < MIN_SPAN or not cands:
            continue
        scores = [_overlap(c.text, want) for c in cands]
        best = max(range(len(cands)), key=lambda i: (scores[i], -len(cands[i].text)))
        if scores[best] >= MIN_OVERLAP:
            gold.add(best)
    return [(c, 1 if i in gold else 0) for i, c in enumerate(cands)]


#: How close the best candidate must come before a pointer is treated as words
#: in this atom at all. Below it the labeler was writing prose -- "the two
#: headings in dispute: ..." names two headings and quotes neither -- and prose
#: is a rationale, not a span.
MIN_OVERLAP = 0.5

_TOKEN = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> float:
    """Token F1 between two spans: symmetric, and no containment special case.

    Containment is what fooled the first version. F1 charges a candidate for
    the words it adds as well as the words it misses, which is exactly the
    difference between the clause and the neighbour that quotes it.
    """
    ta, tb = _TOKEN.findall(a.lower()), _TOKEN.findall(b.lower())
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    n = sum(common.values())
    return 0.0 if not n else 2 * n / (len(ta) + len(tb))


def dataset(labels: Iterable[dict[str, Any]]) -> tuple[list[list[float]], list[int], list[Candidate]]:
    X: list[list[float]] = []
    y: list[int] = []
    cands: list[Candidate] = []
    for lb in labels:
        for cand, tag in training_pairs(lb):
            X.append(features_for(cand, lb))
            y.append(tag)
            cands.append(cand)
    return X, y, cands


#: How much a load-bearing atom outweighs a slight one, matching the tiers the
#: labeler assigns and the weights `human_labels` already emits.
TIER_WEIGHT = {"load_bearing": 3.0, "ordinary": 1.0, "slight": 0.3}


def pairwise_dataset(labels: Iterable[dict[str, Any]]
                     ) -> tuple[list[list[float]], list[int], list[float]]:
    """Differences between a right candidate and a wrong one, within one atom.

    The first model here was a pointwise classifier and it scored 85%, losing
    every bill-of-materials line: "Mag Lock Cable" is decided by the heading
    above it, and a pointwise model had learned a global prior that the atom's
    own words win, because across the corpus they usually do.

    That prior is the bug. The question is never "are these words decisive in
    general", it is "are these words more decisive than the other words on
    offer for THIS atom". Subtracting two candidates of the same atom cancels
    the prior, and the same features then reach 95%.

    Each pair is emitted twice, once in each direction, so the model cannot
    learn a bias term through the back door.
    """
    X: list[list[float]] = []
    y: list[int] = []
    w: list[float] = []
    for lb in labels:
        pairs = training_pairs(lb)
        feats = [features_for(c, lb) for c, _ in pairs]
        good = [i for i, (_, t) in enumerate(pairs) if t]
        bad = [i for i, (_, t) in enumerate(pairs) if not t]
        weight = TIER_WEIGHT.get(str(lb.get("weight_tier") or ""), 1.0)
        for i in good:
            for j in bad:
                diff = [a - b for a, b in zip(feats[i], feats[j])]
                X.append(diff)
                y.append(1)
                w.append(weight)
                X.append([-v for v in diff])
                y.append(0)
                w.append(weight)
    return X, y, w


def fit(labels: Iterable[dict[str, Any]]):
    """A linear ranker over `FEATURE_NAMES`. Returns the weight vector.

    Deliberately the smallest model that works. The candidate generator and the
    features are what took the work; anything stronger can be dropped in behind
    the same `rank()` without touching them.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    X, y, w = pairwise_dataset(labels)
    if not X or len(set(y)) < 2:
        raise ValueError("no rankable atoms: every candidate is right, or none is")
    model = LogisticRegression(max_iter=5000, fit_intercept=False)
    model.fit(X, y, sample_weight=w)
    return list(model.coef_[0])


def rank(label: dict[str, Any], score) -> list[tuple[Candidate, float]]:
    """Candidates for one atom, best first.

    `score` is either a weight vector from `fit` or any callable over features.
    """
    if not callable(score):
        weights = list(score)
        def score(f, _w=weights):  # noqa: E306
            return sum(a * b for a, b in zip(f, _w))
    scored = [(c, float(score(features_for(c, label)))) for c in candidates_for(label)]
    return sorted(scored, key=lambda x: -x[1])


def precision_at_1(labels: list[dict[str, Any]], folds: int = 5, seed: int = 0) -> tuple[int, int]:
    """How often the top-ranked span is one the labeler pointed at, atoms held out.

    Only atoms that have both a right and a wrong candidate are scored. An atom
    whose every candidate is correct -- a phone number, a one-line question --
    has nothing to rank and would inflate the number.
    """
    import numpy as np  # noqa: PLC0415

    rankable = [lb for lb in labels
                if (p := training_pairs(lb)) and any(t for _, t in p)
                and not all(t for _, t in p)]
    if not rankable:
        return 0, 0
    order = np.random.default_rng(seed).permutation(len(rankable))
    hits = 0
    for held in np.array_split(order, min(folds, len(rankable))):
        out = set(held.tolist())
        weights = fit([rankable[i] for i in range(len(rankable)) if i not in out])
        for i in sorted(out):
            top = rank(rankable[i], weights)[0][0]
            hits += any(c.text == top.text and t
                        for c, t in training_pairs(rankable[i]))
    return hits, len(rankable)


__all__ = [
    "Candidate", "FEATURE_NAMES", "FIELD_HINTS", "MIN_SPAN", "MAX_SPAN",
    "MIN_OVERLAP", "TIER_WEIGHT", "candidates_for", "features_for",
    "pointer_kind", "best_locatable", "training_pairs", "dataset",
    "pairwise_dataset", "fit",
    "rank", "precision_at_1",
]
