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

import json
import os
import threading as _threading
from pathlib import Path
from typing import Callable, Iterable, Sequence

_PROTO_CACHE: dict[str, object] = {}  # rule-name -> (pos_matrix, neg_matrix)

#: ONE backend verdict per compile, not one per rule evaluation.
#:
#: ``embedding_endpoint_reachable()`` is a live network probe. Both backends
#: cache it, but the probe runs OUTSIDE their lock, so the parse pool can have
#: several threads probing an unhealthy endpoint at once and reaching different
#: verdicts -- and a TTL expiry flips the answer part-way through a compile.
#: Either way one document is judged by the embedder and the next by the
#: lexical fallback, which is how live 01491cca's title line came out as
#: "SOW - Premise Wiring, Bldg. 704 B-4" parsed alone and "... Bldg. 704"
#: parsed alongside other files. Same bytes, different atom, different
#: ``label_key``.
#:
#: Resolved once, under a lock, and frozen. Every rule in a compile then agrees
#: on which path it is taking. ``reset_semantic_backend()`` re-probes, and the
#: compiler calls it when a compile begins so an outage never outlives one run.
#:
#: This makes a compile SELF-consistent. It does not make two compiles on
#: different days agree: a deal parsed while the embedder is up still reads
#: differently from the same deal parsed while it is down. Pin
#: ``SOWSMITH_SEMANTIC_RULES`` to settle that permanently.
_BACKEND_LOCK = _threading.Lock()
_BACKEND_VERDICT: bool | None = None


def reset_semantic_backend() -> None:
    """Forget the frozen verdict; the next rule re-probes once."""
    global _BACKEND_VERDICT
    with _BACKEND_LOCK:
        _BACKEND_VERDICT = None


def semantic_backend_available() -> bool:
    """Can the embedder serve this compile? Probed at most once per reset."""
    global _BACKEND_VERDICT
    verdict = _BACKEND_VERDICT
    if verdict is not None:
        return verdict
    with _BACKEND_LOCK:
        if _BACKEND_VERDICT is None:
            try:
                from app.core.embedding_retrieval import embedding_endpoint_reachable
                _BACKEND_VERDICT = bool(embedding_endpoint_reachable())
            except Exception:
                _BACKEND_VERDICT = False
        return _BACKEND_VERDICT


# Longest candidate worth pre-embedding. Rules judge LINES (headings, lead-ins,
# labels); a multi-KB prose blob is never a rule candidate and would only bloat
# the prewarm request.
_PREWARM_MAX_CHARS = 400
# Ceiling on one prewarm request so a 500-page dump can't build a giant payload.
_PREWARM_MAX_TEXTS = int(os.environ.get("SOWSMITH_SEMANTIC_PREWARM_MAX", "4000"))


def prewarm(texts: Iterable[str]) -> int:
    """Embed a document's candidate lines in ONE round trip. Returns the count.

    ``SemanticRule.fires`` embeds ONE text per call, so each candidate line
    costs a full HTTP round trip — ~2.5 s against the remote qwen3 embedder.
    A single RFP PDF asks about ~900 distinct lines, which is ~38 minutes of
    purely serial network wait for a parse that should take seconds; that is
    what made a full-pack compile look like a hang.

    ``embed_texts`` already collapses its cache misses into a single
    ``/api/embed`` request and persists them, so handing it the whole
    candidate set up front turns N round trips into 1 — every subsequent
    ``fires()`` call is then a local cache hit.

    Purely a cache-filling optimisation: it changes no decision, and it is
    best-effort — offline, disabled, or failing, it returns 0 and every rule
    behaves exactly as before.
    """
    if SemanticRule._disabled():
        return 0
    seen: set[str] = set()
    batch: list[str] = []
    for raw in texts:
        t = (raw or "").strip()
        if not t or len(t) > _PREWARM_MAX_CHARS or t in seen:
            continue
        seen.add(t)
        batch.append(t)
        if len(batch) >= _PREWARM_MAX_TEXTS:
            break
    if not batch:
        return 0
    try:
        from app.core.embedding_retrieval import embed_texts, embedding_endpoint_reachable
        if not embedding_endpoint_reachable():
            return 0
        embed_texts(batch)
    except Exception:
        return 0
    return len(batch)

# ── trained-threshold registry ──────────────────────────────────────────
# A rule's threshold is hand-set at construction, but the trainer
# (_train_semantic_rules.py) re-fits it against the rule's labelled examples
# (its positives/negatives + accumulated labeler corrections) and writes the
# tuned value here. A rule loads its trained threshold at construction, so
# retraining shifts behaviour with NO code edit. Absent file -> hand-set default.
_THRESHOLD_REGISTRY: dict | None = None


def _threshold_registry_path() -> Path:
    return Path(os.environ.get(
        "SOWSMITH_RULE_THRESHOLDS",
        str(Path(__file__).resolve().parents[2] / "models" / "semantic_rule_thresholds.json"),
    ))


def _load_threshold_registry() -> dict:
    global _THRESHOLD_REGISTRY
    if _THRESHOLD_REGISTRY is None:
        try:
            _THRESHOLD_REGISTRY = json.loads(_threshold_registry_path().read_text(encoding="utf-8"))
        except Exception:
            _THRESHOLD_REGISTRY = {}
    return _THRESHOLD_REGISTRY


def _trained_threshold(name: str, default: float) -> float:
    try:
        ent = _load_threshold_registry().get(name)
        if isinstance(ent, dict) and isinstance(ent.get("threshold"), (int, float)):
            return float(ent["threshold"])
    except Exception:
        pass
    return default


def _log_decision(name: str, text: str, best_pos: float, best_neg: float,
                  threshold: float, decision: bool) -> None:
    """Append one rule decision to the feedback log (JSONL) when
    ``SOWSMITH_RULE_LOG`` points at a path. The reviewer's accept/reject on the
    resulting atom is later joined to these rows to label them — that labelled
    set is what the trainer re-fits the threshold on. Off (no env) -> zero cost."""
    path = os.environ.get("SOWSMITH_RULE_LOG")
    if not path:
        return
    try:
        rec = {"rule": name, "text": (text or "")[:400], "best_pos": round(best_pos, 4),
               "best_neg": round(best_neg, 4), "threshold": round(threshold, 4),
               "decision": bool(decision)}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break a parse


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
        self.default_threshold = threshold
        # A trained threshold (from the eval-gated trainer) overrides the hand-set
        # default with NO code edit; absent registry -> hand-set default.
        self.threshold = _trained_threshold(name, threshold)
        self.lexical_fallback = lexical_fallback

    # -- env switches -----------------------------------------------------
    @staticmethod
    def _disabled() -> bool:
        # global kill-switch: force the lexical fallback everywhere (CI / offline
        # determinism / debugging a regression to the embedder).
        return os.environ.get("SOWSMITH_SEMANTIC_RULES", "1") == "0"

    def _reachable(self) -> bool:
        # Frozen for the compile: see `semantic_backend_available`. Asking the
        # network here, once per rule per atom, let one compile take both paths.
        return semantic_backend_available()

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
        # disabled -> deterministic lexical fallback (never break a parse)
        if self._disabled():
            return self._lexical(text)
        # CACHE BEFORE REACHABILITY.
        #
        # This used to ask `_reachable()` first, so a rule took the regex
        # fallback whenever the endpoint was down -- even for a line whose
        # vector was already on disk. The decision moved with the NETWORK
        # rather than with the document, which is why the same deal read
        # differently depending on whether the embedder happened to be up.
        #
        # The cache is content-addressed by model || sha256(text), so a cached
        # vector is the same vector the endpoint would return and the decision
        # is identical either way. The reachability gate still guards the MISS
        # path, which is what it was actually for: embedding an uncached line
        # against a wedged host blocks for SOWSMITH_EMBED_TIMEOUT (180s).
        from app.core.embedding_retrieval import cached_embedding, embed_texts

        cached = cached_embedding(text)
        if cached is None and not self._reachable():
            return self._lexical(text)
        try:
            np = _np()
            pos, neg = self._protos()
            q = np.array(cached if cached is not None else embed_texts([text])[0],
                         dtype="float32")
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
            decision = best_pos >= self.threshold and best_pos > best_neg
            # log the decision (+ its scores) for the feedback/training loop —
            # no-op unless SOWSMITH_RULE_LOG is set, so prod pays nothing.
            _log_decision(self.name, text, best_pos, best_neg, self.threshold, decision)
            return decision
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
# Forward cues that announce a list/section below: 'the following', 'as follows',
# AND list-announcing verbs ('Services include:', 'Scope consists of:', 'This
# support is limited to:') — the latter were missed offline, so those lead-ins
# got emitted as their own atom AND used as the list's section. The structural
# gate (a bullet/list directly follows) is what actually constrains this, so a
# non-lead-in 'include' sentence with nothing below it is never lifted.
_FRAMING_LEAD_IN_RE = _re.compile(
    r"\b(the following|as follows|includ\w*|consist\w*|compris\w*|"
    r"limited to|are as|listed below|outlined below|described below)\b", _re.I)


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


def is_trained(name: str) -> bool:
    """Has this rule's threshold been fitted from data, or is it my guess?

    A SemanticRule seeded with hand-written prototypes is not a learned thing.
    It is a regex with a cosine on top, and it carries the seeder's blind spots
    with none of a regex's predictability -- plus a dependency on an embedder
    that is often unreachable, in which case the lexical fallback decides and
    the "semantic" rule is the fallback wearing a hat.

    So a rule earns the decision by being trained: `_train_semantic_rules.py`
    fits the threshold leave-one-out against the rule's examples PLUS the
    human-labelled decisions in the SOWSMITH_RULE_LOG, and adopts it only if it
    beats the current threshold's F1. Until that has happened and written a
    registry entry, callers should keep whatever deterministic rule they had.
    """
    try:
        ent = _load_threshold_registry().get(name)
        return isinstance(ent, dict) and isinstance(ent.get("threshold"), (int, float))
    except Exception:
        return False


def list_item_lexical(text: str) -> bool:
    """Offline net for the item judgment: short enough to be an item.

    A word count is what this decision used to be, whole -- ten words for an
    ordinary label, thirty under a section heading. It is kept HERE, as the
    fallback, because a number is an honest last resort and a dishonest
    judgment: "Verify distances." and "Thanks, let me know if you need
    anything else." are both short, and only one of them is an item.
    """
    words = (text or "").split()
    return 1 <= len(words) <= 30


def list_item_rule() -> "SemanticRule":
    """Is this line one of the items under the label above it, or the prose
    that ended the list?

    The structural part -- that there IS a label above, and that this line sits
    under it -- is segmentation, and the parser knows it. What the parser does
    NOT know is whether a line is a fact the list is enumerating or a sentence
    that has moved on, and that is a judgment about meaning. It was a word
    count, which is why "The setup will require Cat 6A cabling, two Ethernet
    connections per workstation, and AV work for conference rooms" (18 words,
    an item) and "That is everything we agreed on the call yesterday" (10
    words, not one) were separated by counting.

    The threshold here is trainable through the eval-gated registry without a
    code edit, which is the point: the number stops being the decision.
    """
    r = _RULE_CACHE.get("list_item_under_label")
    if r is None:
        r = SemanticRule(
            name="list_item_under_label",
            positives=[
                # notes sentences -- full clauses that are each one fact
                "The team discussed the office layout, including 106 workstations, conference rooms, phone rooms, IT room, and pantry.",
                "CAD drawings and plans were shared for review.",
                "The setup will require Cat 6A cabling, two Ethernet connections per workstation, and AV work for conference rooms.",
                "Electrical connections will be provided by the landlord, with furniture vendors handling workstation hookups.",
                "The GC will dictate the schedule for cabling installation, which typically takes four to six weeks.",
                "The survey should be conducted after construction is complete.",
                "Wireless access points will require Ethernet cabling.",
                # bare items under a plain label
                "Board room", "Executive offices", "Print/copy areas", "IT room/closet",
                "Measure cabling pathways.", "Verify distances.", "Determine drop locations.",
                "Cabling to all rooms.", "Two drops per room.",
                "24 Cat6A drops in the IT closet",
            ],
            negatives=[
                # the list is over: closings, pleasantries, questions back
                "That is everything we agreed on the call yesterday afternoon, and nothing else is in scope for this phase.",
                "Thanks, let me know if you need anything else.",
                "Looking forward to speaking with you,",
                "Appreciate you thinking of us.",
                "I'll listen to this over, get some good notes on it, and get it to my team and I'll get back with you.",
                "Can you try listening to the recording below, and see if we can get budgetary numbers together?",
                "Hi Pat,",
                "Please see the attached and let me know your thoughts.",
            ],
            threshold=0.62,
            lexical_fallback=list_item_lexical,
        )
        _RULE_CACHE["list_item_under_label"] = r
    return r


def reads_as_list_item(text: str, *, fallback: Callable[[str], bool] | None = None) -> bool:
    """Structural prefilter (bounds what we embed) + the item rule.

    Falls back to ``fallback`` -- the caller's own deterministic rule -- until
    this rule has a TRAINED threshold. Seeded prototypes decide nothing: mine
    scored "Erick offered a Cisco-funded wireless site survey to determine
    access point needs" as NOT an item, because it sits nearer the negative
    "I'll listen to this over ... and I'll get back with you" than the
    positives -- both are somebody offering to do something. The word count it
    replaced got that line right.
    """
    t = (text or "").strip()
    if not t or len(t) > 400:
        return False
    if len(t.split()) > 40:      # nothing anyone writes as a list item
        return False
    if not is_trained("list_item_under_label"):
        # SHADOW: ask the rule anyway so the decision is logged, then ignore it
        # and return the caller's deterministic answer. Without this the rule
        # can never bootstrap -- it is not consulted because it is not trained,
        # and it is not trained because nothing consulted it, so nothing logged.
        # The logged row also carries what the rule WOULD have said next to what
        # the count did, which is the pair a threshold is fitted on.
        try:
            list_item_rule().fires(t)
        except Exception:
            pass
        return (fallback or list_item_lexical)(t)
    return list_item_rule().fires(t)


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


def _meeting_section_header_lexical(text: str) -> bool:
    """Offline net: line is exactly a known meeting-summary section label."""
    try:
        from app.core.normalizers import detect_section

        return detect_section(text or "") is not None
    except Exception:
        t = (text or "").strip().rstrip(":").lower()
        return t in {
            "executive summary",
            "action items",
            "action item",
            "key decisions",
            "key decision",
            "decisions",
            "decision",
            "open questions",
            "open question",
            "attendees",
            "participants",
            "next steps",
            "agenda",
            "discussion",
            "notes",
            "follow ups",
            "follow-ups",
            "follow up",
        }


def meeting_section_header_rule() -> "SemanticRule":
    """Is a short standalone line a meeting-summary SECTION header
    (Executive Summary / Action Items / Key Decisions) rather than a bullet
    body or a speaker stamp?

    Embeddings lead online; ``detect_section`` is the offline lexical net.
    Callers must structurally gate (short line, not a speaker stamp, list/body
    follows) so ordinary prose mentioning 'action items' never fires.
    """
    r = _RULE_CACHE.get("meeting_section_header")
    if r is None:
        r = SemanticRule(
            name="meeting_section_header",
            positives=[
                "Executive Summary",
                "Action Items",
                "Key Decisions",
                "Open Questions",
                "Attendees",
                "Participants",
                "Next Steps",
                "Agenda",
                "Discussion",
                "Follow Ups",
                "Decisions",
                "Notes",
            ],
            negatives=[
                "Alex Rivera [00:04] Hey, how are you?",
                "We discussed the executive summary briefly.",
                "Jacob to send the full equipment list.",
                "Badge readers and cameras are in scope.",
                "Remote implementation was preferred over on-site work.",
                "The contractor shall provide all materials",
                "Yes",
                "No",
            ],
            threshold=0.55,
            lexical_fallback=_meeting_section_header_lexical,
        )
        _RULE_CACHE["meeting_section_header"] = r
    return r


def is_meeting_section_header(text: str) -> bool:
    """Structural prefilter + meeting_section_header SemanticRule."""
    t = (text or "").strip()
    if not t or len(t) > 60:
        return False
    if t[-1:] in ".,;!?":
        return False
    words = _re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
    if not (1 <= len(words) <= 5):
        return False
    # Speaker stamps are never section headers.
    if _re.search(r"\[\d{1,2}:\d{2}", t):
        return False
    return meeting_section_header_rule().fires(t)
