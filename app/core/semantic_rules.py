"""Semantic rules — fire a fuzzy *linguistic* judgment by embedding similarity
instead of a keyword regex, so it generalizes to phrasings nobody wrote a keyword
for ("the vendor's responsibilities encompass:" fires the same as "...the
following services.").

A rule is a small set of POSITIVE prototype phrases (things that SHOULD fire) and
NEGATIVE ones (look similar but should NOT). At call time we embed the candidate
and fire iff its nearest prototype is a positive whose cosine clears ``threshold``.

Design principles:
  * STRUCTURE stays structural. This is only for linguistic judgments (is-this-a
    -lead-in / exclusion / boilerplate / section-type). Don't use it for things a
    flag already answers (hidden column, numPr list item, sheet role by shape).
  * SAFE OFFLINE. The qwen3 embedder lives on a box that sleeps/relays. If it is
    unreachable we fall back to the rule's ``lexical_fallback`` (the old regex),
    so a parse NEVER breaks or silently changes behaviour when embeddings are down.
  * SELF-HEALING. ``positives``/``negatives`` are just example lists — a PM/intern
    correction becomes a new example (no new regex), and the rule's behaviour shifts.
  * CHEAP. Prototypes embed once (process-cached); candidates hit the existing
    per-text embedding cache, and callers only ask about structurally-gated
    candidates, so the round-trips are bounded.
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

_PROTO_CACHE: dict[str, object] = {}  # rule-name -> (pos_matrix, neg_matrix)


def _np():
    import numpy as np  # local import keeps parser import light
    return np


class SemanticRule:
    def __init__(
        self,
        name: str,
        positives: Sequence[str],
        negatives: Sequence[str] = (),
        threshold: float = 0.62,
        lexical_fallback: Callable[[str], bool] | None = None,
    ) -> None:
        self.name = name
        self.positives = list(positives)
        self.negatives = list(negatives)
        self.threshold = threshold
        self.lexical_fallback = lexical_fallback

    # -- env switches -----------------------------------------------------
    @staticmethod
    def _disabled() -> bool:
        # global kill-switch: force the lexical fallback everywhere (CI / offline
        # determinism / debugging a regression to the embedder).
        return os.environ.get("SOWSMITH_SEMANTIC_RULES", "1") == "0"

    def _reachable(self) -> bool:
        try:
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            return bool(embedding_endpoint_reachable())
        except Exception:
            return False

    def _lexical(self, text: str) -> bool:
        return bool(self.lexical_fallback(text)) if self.lexical_fallback else False

    # -- prototype embedding (cached) -------------------------------------
    def _protos(self):
        cached = _PROTO_CACHE.get(self.name)
        if cached is not None:
            return cached
        from app.core.embedding_retrieval import embed_texts
        np = _np()
        texts = self.positives + self.negatives
        vecs = np.array(embed_texts(texts), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-9)
        pos = vecs[: len(self.positives)]
        neg = vecs[len(self.positives) :]
        # Cache ONLY healthy prototypes. If the embedder is reachable-but-broken
        # (a transient down returns ZERO vectors with no exception), caching them
        # would poison every rule for the whole process — so skip the cache and
        # let the next call retry once the embedder recovers.
        if float(norms.min()) > 1e-6:
            _PROTO_CACHE[self.name] = (pos, neg)
        return pos, neg

    # -- the decision -----------------------------------------------------
    def fires(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        # offline / disabled -> deterministic lexical fallback (never break a parse)
        if self._disabled() or not self._reachable():
            return self._lexical(text)
        try:
            from app.core.embedding_retrieval import embed_texts
            np = _np()
            pos, neg = self._protos()
            q = np.array(embed_texts([text])[0], dtype="float32")
            qn = float(np.linalg.norm(q))
            # A zero / degenerate embedding means the embedder is
            # reachable-but-broken (returns zeros, no exception). Computing cosine
            # on it makes EVERY rule silently False and BYPASSES the fallback — so
            # detect it and degrade to the lexical net like a normal outage.
            if qn < 1e-6 or float(np.linalg.norm(pos)) < 1e-6:
                return self._lexical(text)
            q /= qn
            best_pos = float((pos @ q).max())
            best_neg = float((neg @ q).max()) if len(neg) else -1.0
            # fire iff the nearest prototype is a POSITIVE and it clears the floor
            return best_pos >= self.threshold and best_pos > best_neg
        except Exception:
            return self._lexical(text)

    def score(self, text: str) -> tuple[float, float]:
        """(nearest-positive cosine, nearest-negative cosine) — for calibration."""
        from app.core.embedding_retrieval import embed_texts
        np = _np()
        pos, neg = self._protos()
        q = np.array(embed_texts([text])[0], dtype="float32")
        q /= np.linalg.norm(q) + 1e-9
        bp = float((pos @ q).max())
        bn = float((neg @ q).max()) if len(neg) else -1.0
        return bp, bn


# ════════════════════════════════════════════════════════════════════
# SHARED RULE REGISTRY — one source of truth for the CROSS-CUTTING rules.
#
# These judge the MEANING of a line (is it a list lead-in? a cover vs deadline
# date? a section heading vs a document title?), so they apply to ANY format.
# Defining them here — instead of inside one parser — means every parser pulls
# the SAME rule + examples with one import: a rule improved for one format
# instantly covers the others, and the "fires on docx but not pdf/xlsx" class
# of bug can't recur (lead-in used to be defined in docx_parser only).
#
# Format-STRUCTURAL rules stay in their parser (xlsx money-column header, docx
# subsection lift) — they key off that format's geometry, not meaning.
# ════════════════════════════════════════════════════════════════════
import re as _re

_RULE_CACHE: dict = {}
_FRAMING_LEAD_IN_RE = _re.compile(r"\b(the following|as follows)\b", _re.I)


def lead_in_lexical(text: str) -> bool:
    """Offline keyword net for the lead-in judgment (the structural prefilter is
    what really constrains it; this just needs the forward cue)."""
    return bool(_FRAMING_LEAD_IN_RE.search(text or ""))


def lead_in_rule() -> "SemanticRule":
    """Does a line ANNOUNCE a following list ('the vendor will perform the
    following services.', 'Deliverables:', 'The following are out of scope:')?
    Polarity-agnostic — scope / exclusion / customer / deliverable intros alike."""
    r = _RULE_CACHE.get("list_lead_in")
    if r is None:
        r = SemanticRule(
            name="list_lead_in",
            positives=[
                "PurTera will provide field technicians to perform the following services.",
                "Subject to the other provisions of this SOW, Provider will perform the following services.",
                "The vendor shall complete the following tasks:",
                "Services include:",
                "Scope of work consists of the following activities:",
                "The contractor will perform the work as follows:",
                "PurTera will provide the following deliverables:",
                "The vendor responsibilities encompass the items below:",
                "The following items are excluded from this SOW unless separately quoted:",
                "The following are out of scope:",
                "Customer responsibilities include the following:",
                "The customer is responsible for the following:",
                "The following are the General Conditions for the work to be performed as outlined in the Specifications.",
                # FRAMING INTROS — a (possibly long) sentence that announces the
                # structured list/sections that follow, without a "following:" cue.
                "The intent is that all responses follow the same format described in the sections below.",
                "Each response must be organized into the following sections.",
                "Proposals will be evaluated on the criteria listed below.",
                "All submissions should be structured as outlined below.",
                "Responses must include each of the components described below.",
                "Deliverables:", "Assumptions:", "Requirements:",
                "Notes:", "Exclusions:", "Scope of work:",
                "The estimated Fees for Services outlined below are Fixed Fee.",
                "The fees set forth below are firm fixed price.",
                "The rates listed below apply to all Services.",
                "All pricing shown in the table below is fixed.",
                "The amounts detailed below are Time and Materials.",
            ],
            negatives=[
                "This SOW does not include predictive wireless design or spectrum analysis.",
                "The school currently receives 5 Gbps of internet bandwidth.",
                "Access point placement validation is limited to confirming locations align with floor plans.",
                "All work will be performed during normal business hours.",
                "The vendor agrees to hold the client harmless from any liability.",
                "Payment is due within thirty days of invoice receipt.",
                "The total contract value is fixed at the agreed amount.",
                "Address: 123 Main Street, Macon GA",
                "Phone: 555-0100", "Total: $5,000", "Date: January 1, 2026",
                "Rates in USD.", "Fees are in USD.",
            ],
            threshold=0.62,
            lexical_fallback=lead_in_lexical,
        )
        _RULE_CACHE["list_lead_in"] = r
    return r


def is_framing_lead_in(text: str) -> bool:
    """Structural prefilter (bounds what we embed) + the semantic lead-in rule.
    A list lead-in ends with '.'/':' and is short, regardless of wording."""
    t = (text or "").strip()
    if not t or len(t) > 200 or not t.endswith((".", ":")):
        return False
    words = _re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
    if not (1 <= len(words) <= 25):
        return False
    return lead_in_rule().fires(t)


def operative_date_rule() -> "SemanticRule":
    """Is a date OPERATIVE (deadline / milestone / effective / award / timeline)
    versus a decorative cover-letterhead date? Judge the date's CONTEXT
    (section / surrounding text), never the bare digits."""
    r = _RULE_CACHE.get("operative_date")
    if r is None:
        r = SemanticRule(
            name="operative_date",
            positives=[
                "proposals are due by this date", "submission deadline",
                "bids must be received by", "contract award date",
                "effective date of the agreement", "project timeline and key dates",
                "projected schedule of events and dates", "milestone completion date",
                "questions due date", "vendor interview date", "responses due no later than",
            ],
            negatives=[
                "the date this document or letter was prepared", "cover page letterhead date",
                "memo header date", "date printed at the top of the page",
            ],
            threshold=0.58,
            lexical_fallback=lambda t: any(
                w in (t or "").lower() for w in (
                    "due", "deadline", "award", "effective", "timeline", "milestone",
                    "completion", "submit", "no later than", "projected", "schedule",
                    "interview", "question", "closing", "start", "end date", "by ",
                )
            ),
        )
        _RULE_CACHE["operative_date"] = r
    return r


def section_title_rule() -> "SemanticRule":
    """Is a heading a generic document SECTION (Introduction / General Conditions
    / Scope of Work) versus a real document/deal TITLE (an org / project name)?
    Stops a section heading from being crowned the document root."""
    r = _RULE_CACHE.get("section_title")
    if r is None:
        r = SemanticRule(
            name="section_title",
            positives=[
                "introduction", "general information", "general conditions",
                "scope of work", "proposal format", "evaluation criteria",
                "insurance requirements", "payment terms", "warranty",
                "terms and conditions", "definitions", "background", "addenda",
                "indemnification", "company responsibility", "specifications",
            ],
            negatives=[
                "The Academy for Classical Education", "Request for Proposal for network infrastructure",
                "ACME Corporation wireless upgrade project", "Statement of Work data center migration",
                "City of Macon broadband initiative",
            ],
            threshold=0.60,
            lexical_fallback=lambda t: any(
                w in (t or "").lower() for w in (
                    "introduction", "general", "scope", "conditions", "proposal",
                    "evaluation", "insurance", "payment", "warranty", "terms",
                    "definition", "background", "addend", "indemnif", "responsibilit",
                    "specification", "requirement", "overview", "purpose",
                )
            ),
        )
        _RULE_CACHE["section_title"] = r
    return r
