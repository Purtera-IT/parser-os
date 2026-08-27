"""Deterministic post-classification type-sanity pass.

The LLM ``typed_atom_classifier`` and the table/xlsx extractors label atoms
in isolation, so commercial meta-values leak into the ``quantity`` bucket:
``"28.57% margin"``, ``"260 PMO Cost"``, ``"99 pricing lines"`` are not
deliverable quantities — they're financial figures or spreadsheet row
counts. This pollutes quantity rollups and the scorecards.

This module is a *guardrail*, not an extractor: it runs after
classification and re-types atoms that fail type-specific sanity rules.
It is fully deterministic (no LLM, no I/O), universal (no customer
terminology), and conservative — it only demotes atoms it can prove are
mis-typed, never the reverse.

Two passes:

1. ``demote_nondeliverable_quantities`` — a ``quantity`` atom that is
   really a financial/percentage figure or a spreadsheet meta-count is
   re-typed to ``pricing_assumption`` (financial) and its ``quantity:``
   entity keys are stripped so it stops inflating quantity rollups.

2. ``surface_headline_quantities`` — when a ``requirement`` / ``scope_item``
   / ``service_line`` atom states a strong ``"<N> <deliverable-noun>"``
   count (e.g. "replace approximately 110 existing TVs") and no existing
   ``quantity`` atom carries that count, emit a child ``quantity`` atom so
   the deal's headline figure is structured, not buried in prose.

Promotion gate (v59) — the same guardrail discipline applied to *claims*
the pipeline cannot support:

3. ``strip_unsupported_names`` — no manufactured strings. A human-facing
   display name ("Goleta Office") that appears nowhere in the atom's own
   source text is not a name, it is an invention. The name is removed (or
   falls back to a *supported* alias); the atom and its evidence survive.
   Abstention on the name, not on the evidence.

4. ``cap_authority_to_source`` — authority cannot exceed what the source
   supports. An atom whose source does not resolve to a real artifact in
   the compile's artifact set is capped at ``machine_extractor``, as is an
   atom whose text is plainly serialized structure (a JSON key path) rather
   than document prose. This forecloses *rank laundering*: the pipeline's
   own prior output, round-tripped through a manifest, re-entering the
   compile wearing customer authority.

5. ``demote_unearned_contract_authority`` — a file format is not a contract.
   Several parsers hardcode ``contractual_scope`` (rank 100), so every
   spreadsheet outranked a PM's own confirmed answer (rank 95) — including
   a Rough Order of Magnitude estimate and an internal deal kit. Rank 100
   now requires positive contract evidence on the source document.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.sentences import split_sentences

# --- financial / meta tokens that disqualify a "quantity" --------------

# Currency, percentage and pricing vocabulary. A quantity carrying any of
# these is a commercial figure, not a deliverable count.
_FINANCIAL_RE = re.compile(
    r"(?:\$|%|\bpercent\b|\bpct\b|\bmargin\b|\bmarkup\b|\bcost(?:s)?\b|"
    r"\bprice(?:s|d|ing)?\b|\brate(?:s)?\b|\bfee(?:s)?\b|\btax(?:es)?\b|"
    r"\bdiscount(?:s)?\b|\brevenue\b|\bprofit\b|\bmsrp\b|\busd\b|\bdollar(?:s)?\b|"
    r"\bsubtotal\b|\bgrand\s+total\b|\bpmo\b|\bburden(?:ed)?\b|\bsell\b\s*rate)",
    re.IGNORECASE,
)

# Spreadsheet meta-counts: "99 pricing lines", "14 line items", "5 rows",
# "118 skus". These count *records*, not deliverables.
_META_COUNT_RE = re.compile(
    r"\b\d[\d,]*\s+(?:pricing\s+lines?|line\s+items?|rows?|records?|"
    r"sku(?:s)?|entries|cells?|columns?|sheets?|tabs?)\b",
    re.IGNORECASE,
)

# Deliverable nouns — the presence of one of these (with a number) means
# the atom really is a countable deliverable and must NOT be demoted even
# if a stray financial token also appears.
_DELIVERABLE_NOUN_RE = re.compile(
    r"\b\d[\d,]*\s+(?:[a-z][a-z\-]*\s+){0,3}"
    r"(?:tv(?:s)?|television(?:s)?|display(?:s)?|monitor(?:s)?|screen(?:s)?|"
    r"unit(?:s)?|device(?:s)?|dwelling(?:s)?|room(?:s)?|door(?:s)?|"
    r"camera(?:s)?|cam(?:s)?|switch(?:es)?|router(?:s)?|firewall(?:s)?|"
    r"access\s+point(?:s)?|ap(?:s)?|wap(?:s)?|sensor(?:s)?|reader(?:s)?|"
    r"controller(?:s)?|speaker(?:s)?|panel(?:s)?|jack(?:s)?|outlet(?:s)?|"
    r"drop(?:s)?|port(?:s)?|cable(?:s)?|cord(?:s)?|rack(?:s)?|cabinet(?:s)?|"
    r"server(?:s)?|appliance(?:s)?|workstation(?:s)?|laptop(?:s)?|"
    r"desktop(?:s)?|license(?:s)?|seat(?:s)?|endpoint(?:s)?|mount(?:s)?|"
    r"projector(?:s)?|enclosure(?:s)?|station(?:s)?|piece(?:s)?|each)\b",
    re.IGNORECASE,
)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", None) or getattr(atom, "normalized_text", None) or "")


def _classify_quantity(text: str) -> str:
    """Return 'deliverable', 'financial', 'meta', or 'ok'.

    'deliverable' wins over 'financial' — a real count that happens to sit
    near a price stays a quantity.
    """
    if _DELIVERABLE_NOUN_RE.search(text):
        return "deliverable"
    if _META_COUNT_RE.search(text):
        return "meta"
    if _FINANCIAL_RE.search(text):
        return "financial"
    return "ok"


def _classify_quantity_key(key: str) -> str:
    """Classify a single ``quantity:<tail>`` entity key.

    The tail is a slugified figure label — ``quantity:260_pmo_cost``,
    ``quantity:28_57_margin``, ``quantity:118_pricing_lines``. We
    de-slugify (``_`` -> space) and run the same deliverable/financial/meta
    vocabulary as atom text. A bare numeric tail (``quantity:110``) carries
    no vocabulary and stays ``ok`` so legitimate deliverable counts survive.
    """
    if not key.startswith("quantity:"):
        return "ok"
    tail = key.split(":", 1)[1]
    probe = re.sub(r"[_\-]+", " ", tail).strip()
    if not probe:
        return "ok"
    # Ensure the meta/deliverable regexes (which anchor on "<number> <noun>")
    # can fire even when the slug leads with the noun rather than the count.
    if not re.match(r"^\d", probe):
        probe = "1 " + probe
    return _classify_quantity(probe)


def scrub_nondeliverable_quantity_keys(atoms: list[Any]) -> int:
    """Strip financial/meta ``quantity:`` entity keys from **every** atom.

    ``demote_nondeliverable_quantities`` only fixes atoms *typed* as
    quantity. But commercial atoms (``commercial_total`` /
    ``pricing_assumption``) routinely carry junk ``quantity:`` keys like
    ``quantity:260_pmo_cost`` or ``quantity:28_57_margin`` — financial
    figures and spreadsheet meta-counts that the entity resolver then
    promotes into bogus quantity entities (polluting the Truth Gate and
    quantity rollups). This pass removes those keys wherever they appear,
    regardless of atom type, while preserving genuine deliverable counts
    (bare numeric tails) untouched.

    Mutates in place. Returns the number of keys stripped.
    """
    stripped = 0
    for atom in atoms:
        keys = list(getattr(atom, "entity_keys", None) or [])
        if not keys:
            continue
        kept: list[Any] = []
        removed_here = False
        for k in keys:
            ks = str(k)
            if ks.startswith("quantity:") and _classify_quantity_key(ks) in ("financial", "meta"):
                stripped += 1
                removed_here = True
                continue
            kept.append(k)
        if removed_here:
            atom.entity_keys = kept
            flag = "scrubbed_nondeliverable_quantity_key"
            existing = list(getattr(atom, "review_flags", None) or [])
            if flag not in existing:
                atom.review_flags = sorted(set(existing + [flag]))
    return stripped


# Payment / credit terms: "Net 30 days", "Net 30", "due in 45 days",
# "30 days net". The number is a credit period, not a deliverable count.
_PAYMENT_TERM_RE = re.compile(
    r"\bnet\s*\d{1,3}\b|\b\d{1,3}\s*days?\s+net\b|"
    r"\bdue\s+(?:in|within|net)\s+\d{1,3}\s*days?\b|\bpayment\s+terms?\b",
    re.IGNORECASE,
)

# Time-of-day / work-window values: "8:00 AM to 5:00 PM", "8am-5pm",
# "business hours", "M-F 7-4". The numbers are clock times / a coverage
# window, not a deliverable count. Universal, content-derived.
_TIME_WINDOW_RE = re.compile(
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|"
    r"\bbusiness\s+hours\b|\bnormal\s+business\b|\bworking\s+hours\b|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*[-–]\s*"
    r"(?:mon|tue|wed|thu|fri|sat|sun)",
    re.IGNORECASE,
)


def _retype_quantity(atom: Any, new_type: Any, flag: str) -> None:
    """Re-type a mis-labelled ``quantity`` atom and stop it inflating the
    quantity rollups (strip ``quantity:`` keys, flag for review)."""
    from app.core.schemas import ReviewStatus

    atom.atom_type = new_type
    keys = [k for k in (getattr(atom, "entity_keys", None) or [])
            if not str(k).startswith("quantity:")]
    atom.entity_keys = keys
    existing = list(getattr(atom, "review_flags", None) or [])
    if flag not in existing:
        atom.review_flags = sorted(set(existing + [flag]))
    if getattr(atom, "review_status", None) != ReviewStatus.needs_review:
        atom.review_status = ReviewStatus.needs_review


def demote_nondeliverable_quantities(atoms: list[Any]) -> int:
    """Re-type financial/meta/temporal atoms mis-labelled as ``quantity``.

    A ``quantity`` atom should be a count of a deliverable. Atoms that are
    really financial figures, spreadsheet record-counts, payment terms, or
    time-of-day windows are re-typed to their correct class and stop
    inflating the quantity rollups. Mutates in place; returns the count.
    A proven deliverable count ("110 units") is never demoted.
    """
    from app.core.schemas import AtomType

    # AtomType members can vary by version; resolve safely.
    _payment = getattr(AtomType, "payment_term", None)
    _window = getattr(AtomType, "site_access_window", None) or getattr(
        AtomType, "site_implementation_note", None
    )

    demoted = 0
    for atom in atoms:
        if _atom_type_str(atom) != "quantity":
            continue
        text = _atom_text(atom)
        verdict = _classify_quantity(text)
        if verdict == "deliverable":
            continue  # a real count — never demote
        if _payment is not None and _PAYMENT_TERM_RE.search(text):
            _retype_quantity(atom, _payment, "retyped_quantity_to_payment_term")
            demoted += 1
            continue
        if _window is not None and _TIME_WINDOW_RE.search(text):
            _retype_quantity(atom, _window, "retyped_quantity_to_access_window")
            demoted += 1
            continue
        if verdict in ("financial", "meta"):
            _retype_quantity(atom, AtomType.pricing_assumption,
                             "retyped_quantity_to_pricing_assumption")
            demoted += 1
    return demoted


_HEADLINE_RE = re.compile(
    r"(?:approximately\s+|approx\.?\s+|about\s+|~\s*)?"
    r"(\d[\d,]*)\s+((?:[a-z][a-z\-]*\s+){0,3})"
    r"(tv(?:s)?|television(?:s)?|display(?:s)?|monitor(?:s)?|unit(?:s)?|"
    r"device(?:s)?|camera(?:s)?|switch(?:es)?|access\s+point(?:s)?|ap(?:s)?|"
    r"door(?:s)?|reader(?:s)?|drop(?:s)?|jack(?:s)?|outlet(?:s)?|port(?:s)?|"
    r"speaker(?:s)?|panel(?:s)?|sensor(?:s)?|workstation(?:s)?|laptop(?:s)?|"
    r"endpoint(?:s)?|license(?:s)?|seat(?:s)?|rack(?:s)?|server(?:s)?)",
    re.IGNORECASE,
)

_COUNT_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_COUNT_NOUN_RE = (
    r"camera(?:s)?|switch(?:es)?|router(?:s)?|firewall(?:s)?|"
    r"access\s+point(?:s)?|ap(?:s)?|wap(?:s)?|reader(?:s)?|badge\s+reader(?:s)?|"
    r"doorbell(?:s)?|nvr(?:s)?|unvr(?:s)?|user(?:s)?|people|person(?:s)?|"
    r"door(?:s)?|device(?:s)?|endpoint(?:s)?|license(?:s)?|seat(?:s)?|"
    r"tv(?:s)?|television(?:s)?|display(?:s)?|monitor(?:s)?|unit(?:s)?|"
    r"room(?:s)?|drop(?:s)?|jack(?:s)?|outlet(?:s)?|rack(?:s)?|server(?:s)?"
)

_RICH_COUNT_RE = re.compile(
    rf"\b(?P<num>\d+|{'|'.join(_COUNT_WORDS)})"
    rf"(?:\s*(?:-|to|or)\s*(?P<num2>\d+|{'|'.join(_COUNT_WORDS)}))?"
    rf"\s*(?P<descriptor>(?:[A-Za-z0-9][A-Za-z0-9+/.-]*\s+){{0,4}}?)"
    rf"(?P<noun>{_COUNT_NOUN_RE})\b",
    re.IGNORECASE,
)

# Units of measure / time / dimension. When the token IMMEDIATELY after the
# number is one of these, the number describes a size, duration, weight, or
# rate — NOT a count of the trailing deliverable noun. This prevents
# "65 inch display" (a screen dimension) and "15 minutes per unit" (a config
# duration) from masquerading as "65 displays" / "15 units". Universal,
# content-derived: a measurement word, not a per-deal alias list.
_MEASURE_WORDS = frozenset({
    "inch", "inches", "in", "foot", "feet", "ft", "yard", "yards",
    "meter", "meters", "metre", "metres", "m", "mm", "cm", "km",
    "mile", "miles",
    "second", "seconds", "sec", "secs", "minute", "minutes", "min", "mins",
    "hour", "hours", "hr", "hrs", "day", "days", "week", "weeks",
    "month", "months", "year", "years",
    "pound", "pounds", "lb", "lbs", "kg", "kgs", "gram", "grams",
    "ton", "tons", "tonne", "tonnes", "ounce", "ounces", "oz",
    "gallon", "gallons", "liter", "liters", "litre", "litres",
    "volt", "volts", "v", "watt", "watts", "w", "amp", "amps", "ampere",
    "hz", "khz", "mhz", "ghz", "kbps", "mbps", "gbps",
    "kb", "mb", "gb", "tb", "pb",
    "percent", "pct", "degree", "degrees", "px", "dpi", "ppi",
    # Port count / interface density is a device attribute in phrases like
    # "2 48 port switches"; it must not become "48 switches".
    "port",
})

_SOURCE_TYPES_FOR_HEADLINE = {"requirement", "scope_item", "service_line"}
_MIN_HEADLINE_COUNT = 10
_LOW_COUNT_NOUNS = frozenset({
    "access point", "access points", "ap", "aps", "wap", "waps",
    "badge reader", "badge readers", "reader", "readers",
    "camera", "cameras", "doorbell", "doorbells",
    "switch", "switches", "router", "routers", "firewall", "firewalls",
    "nvr", "nvrs", "unvr", "unvrs", "user", "users", "people", "person", "persons",
})


def _existing_quantity_counts(atoms: list[Any]) -> set[int]:
    counts: set[int] = set()
    for atom in atoms:
        if _atom_type_str(atom) != "quantity":
            continue
        for k in (getattr(atom, "entity_keys", None) or []):
            ks = str(k)
            if ks.startswith("quantity:"):
                try:
                    counts.add(int(float(ks.split(":", 1)[1])))
                except (ValueError, IndexError):
                    pass
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            q = val.get("quantity")
            if isinstance(q, (int, float)) and not isinstance(q, bool):
                counts.add(int(q))
    return counts


def _parse_count_token(raw: str) -> int | None:
    token = (raw or "").strip().lower().replace(",", "")
    if not token:
        return None
    if token in _COUNT_WORDS:
        return _COUNT_WORDS[token]
    try:
        return int(token)
    except ValueError:
        return None


def _canonical_quantity_noun(raw: str) -> str:
    noun = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if noun in {"ap", "aps", "wap", "waps", "access point", "access points"}:
        return "access points"
    if noun in {"reader", "readers", "badge reader", "badge readers"}:
        return "badge readers"
    if noun in {"nvr", "nvrs", "unvr", "unvrs"}:
        return "NVRs"
    if noun in {"person", "persons", "people", "user", "users"}:
        return "users"
    return noun


#: Kept only so existing callers importing this name still resolve; the
#: splitting itself now goes through ``split_sentences``, which does not
#: break "Part No. 77-K298" or "St. Louis" the way this pattern does.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _context_sentence(text: str, span: tuple[int, int]) -> str:
    """Return the sentence (or a bounded window) of ``text`` that contains the
    matched quantity span, so a surfaced quantity atom carries its subject and
    surrounding statement instead of an orphaned "<N> <noun>". Universal — pure
    sentence segmentation, no domain vocabulary."""
    if not text:
        return ""
    start, end = span
    cursor = 0
    for sentence in split_sentences(text):
        seg_start = text.find(sentence, cursor)
        if seg_start < 0:
            seg_start = cursor
        seg_end = seg_start + len(sentence)
        cursor = seg_end
        if seg_start <= start < seg_end:
            picked = sentence.strip()
            break
    else:
        picked = text.strip()
    # A transcript "sentence" can still be a long multi-clause turn; keep it
    # bounded but always context-bearing (never shorter than the bare mention).
    if len(picked) > 320:
        left = max(0, start - 160)
        right = min(len(text), end + 160)
        picked = text[left:right].strip()
    return picked


def _iter_quantity_mentions(text: str) -> list[tuple[int, str, dict[str, Any]]]:
    mentions: list[tuple[int, str, dict[str, Any]]] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in seen_spans)

    for m in _RICH_COUNT_RE.finditer(text):
        n1 = _parse_count_token(m.group("num"))
        if n1 is None:
            continue
        n2 = _parse_count_token(m.group("num2") or "")
        descriptor = re.sub(r"\s+", " ", (m.group("descriptor") or "").strip().lower())
        noun = _canonical_quantity_noun(m.group("noun"))
        descriptor_tokens = [t.strip(".,;:").lower() for t in descriptor.split() if t.strip()]
        if descriptor_tokens and descriptor_tokens[0] in _MEASURE_WORDS:
            continue
        if number_is_naming_label(text, m.start("num")):
            continue
        quantity = max(n1, n2) if n2 is not None else n1
        if quantity < 1 or quantity > 100_000:
            continue
        if quantity < _MIN_HEADLINE_COUNT and noun.lower() not in _LOW_COUNT_NOUNS:
            continue
        metadata: dict[str, Any] = {"kind": "quantity", "quantity": quantity, "noun": noun, "inferred": True}
        if n2 is not None:
            metadata["range_min"] = min(n1, n2)
            metadata["range_max"] = max(n1, n2)
        if descriptor:
            metadata["descriptor"] = descriptor
        if re.search(r"\bspare\b", descriptor, re.I) or re.search(r"\bspare\b", m.group(0), re.I):
            metadata["qualifier"] = "spare"
        metadata["headline"] = f"{quantity} {noun}"
        metadata["context"] = _context_sentence(text, m.span())
        mentions.append((quantity, noun, metadata))
        seen_spans.append(m.span())

    for m in _HEADLINE_RE.finditer(text):
        if _overlaps(*m.span()):
            continue
        raw = m.group(1).replace(",", "")
        try:
            n = int(raw)
        except ValueError:
            continue
        if n < _MIN_HEADLINE_COUNT or n > 100_000:
            continue
        filler = (m.group(2) or "").strip().lower()
        first_token = filler.split()[0] if filler else (m.group(3) or "").strip().lower()
        if first_token in _MEASURE_WORDS:
            continue
        if number_is_naming_label(text, m.start(1)):
            continue
        noun = _canonical_quantity_noun(m.group(3))
        mentions.append((n, noun, {
            "kind": "quantity", "quantity": n, "noun": noun, "inferred": True,
            "headline": f"{n} {noun}", "context": _context_sentence(text, m.span()),
        }))

    return mentions


def surface_headline_quantities(atoms: list[Any], *, project_id: str) -> list[Any]:
    """Emit a ``quantity`` atom for a strong ``<N> <deliverable>`` count
    stated in prose that no existing quantity atom captures.

    Conservative: only counts >= ``_MIN_HEADLINE_COUNT`` from
    requirement/scope/service atoms, deduped against existing quantity
    values and against each other. Returns new atoms (does not mutate the
    input list).
    """
    from app.core.ids import stable_id
    from app.core.schemas import (
        ArtifactType,
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )

    have = _existing_quantity_counts(atoms)
    emitted_counts: set[tuple[int, str]] = set()
    out: list[Any] = []
    train_rows: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) not in _SOURCE_TYPES_FOR_HEADLINE:
            continue
        text = _atom_text(atom)
        for n, noun, metadata in _iter_quantity_mentions(text):
            emitted_key = (n, noun.lower())
            if n in have or emitted_key in emitted_counts:
                continue
            emitted_counts.add(emitted_key)
            artifact_id = getattr(atom, "artifact_id", "") or ""
            atom_id = stable_id("atm", artifact_id, "quantity_headline", str(n), noun)
            # Carry the surrounding statement so the quantity is actionable to a
            # head (subject + context), not an orphaned "<N> <noun>". Falls back
            # to the parent atom text, then to the bare headline.
            headline = metadata.get("headline") or f"{n} {noun}"
            context_text = (metadata.get("context") or text or headline).strip() or headline
            src_refs = list(getattr(atom, "source_refs", None) or [])
            if not src_refs:
                src_refs = [
                    SourceRef(
                        id=stable_id("src", atom_id),
                        artifact_id=artifact_id,
                        artifact_type=ArtifactType.txt,
                        filename=artifact_id or "headline_quantity",
                        locator={"extraction": "headline_quantity"},
                        extraction_method="headline_quantity",
                        parser_version="atom_type_sanity_v1",
                    )
                ]
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.quantity,
                    raw_text=context_text,
                    normalized_text=context_text.lower(),
                    value=metadata,
                    entity_keys=[f"quantity:{n}"],
                    source_refs=src_refs,
                    receipts=[],
                    authority_class=AuthorityClass.machine_extractor,
                    confidence=0.55,
                    confidence_raw=0.55,
                    calibrated_confidence=0.55,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["headline_quantity"],
                    parser_version="atom_type_sanity_v1",
                )
            )
            try:
                from app.core.training_log import TEACHER_STORE, TrainingRow

                train_rows.append(
                    TrainingRow(
                        relation="equipment_quantity_context",
                        label=noun.lower(),
                        raw_text=text[:4000],
                        label_kind="span",
                        teacher=TEACHER_STORE,
                        confidence=0.55,
                        deal_id=project_id,
                        project_id=project_id,
                        provenance={
                            "quantity": n,
                            "noun": noun,
                            "source": "headline_quantity_fallback",
                        },
                    )
                )
            except Exception:
                pass
    if train_rows:
        try:
            from app.core.training_log import log_rows

            log_rows(train_rows)
        except Exception:
            pass
    return out


_MANIFEST_META_BOM_RE = re.compile(
    r"^artifacts\[\d+\]\.(?:attachment_id|blob_url|content_sha256|filename|content_type|size_bytes|mime_type)\s*:",
    re.I,
)


def demote_manifest_metadata_bom_lines(atoms: list[Any]) -> int:
    """Re-type manifest JSON metadata mis-labelled as ``bom_line`` by the classifier."""
    from app.core.schemas import AtomType

    demoted = 0
    for atom in atoms:
        if _atom_type_str(atom) != "bom_line":
            continue
        text = _atom_text(atom)
        if not _MANIFEST_META_BOM_RE.match(text):
            continue
        atom.atom_type = AtomType.scope_item
        flags = list(getattr(atom, "review_flags", None) or [])
        if "retyped_manifest_meta_bom_line" not in flags:
            flags.append("retyped_manifest_meta_bom_line")
        atom.review_flags = flags
        demoted += 1
    return demoted


def demote_email_include_list_microtasks(atoms: list[Any]) -> int:
    """Re-type mistyped Include-list micro-labels back to ``scope_item``.

    The typed classifier can promote ``Okta integration`` to ``task`` or
    ``requirement`` when the email parser stripped the bullet chrome; Include
    bullets are verbatim evidence, not standalone tasks/requirements.
    """
    from app.core.schemas import AtomType

    demoted = 0
    for atom in atoms:
        at = _atom_type_str(atom)
        if at not in {"task", "requirement"}:
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        if val.get("list_section") == "include" and val.get("kind") == "email_body_line":
            # Do not demote intentionally minted quote-line parents that lost
            # list_section — those no longer carry include polarity.
            if val.get("is_quote_line") and val.get("backfill_reason"):
                continue
            atom.atom_type = AtomType.scope_item
            demoted += 1
    return demoted


def demote_customer_quote_requirements(atoms: list[Any]) -> int:
    """Quoted customer utterances are ``customer_instruction``, not ``requirement``."""
    from app.core.schemas import AtomType

    demoted = 0
    for atom in atoms:
        if _atom_type_str(atom) != "requirement":
            continue
        text = _atom_text(atom).strip()
        if not text:
            continue
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("\u201c") and text.endswith("\u201d")
        ):
            atom.atom_type = AtomType.customer_instruction
            flags = list(getattr(atom, "review_flags", None) or [])
            if "retyped_customer_quote" not in flags:
                flags.append("retyped_customer_quote")
            atom.review_flags = flags
            demoted += 1
    return demoted


# ── invariant 1: no manufactured display names ─────────────────────────
#
# A name the pipeline shows a PM ("Goleta Office") must be a string a human
# actually wrote in the document that atom came from. Anything else is a
# fabrication, however plausible it reads — and a plausible fabrication is
# worse than no name at all, because it survives review.

#: Scalar fields that carry a human-facing display name. Deliberately a
#: closed list of *top-level* keys: we never recurse into nested dicts,
#: because a nested ``label`` is usually a classifier verdict
#: (``facility_label.label == "keep_facility"``), not a name.
_DISPLAY_NAME_FIELDS: tuple[str, ...] = (
    "name",
    "facility_name",
    "display_name",
    "site_name",
)

#: Fields holding a list of alternate names / aliases for the same thing.
_NAME_LIST_FIELDS: tuple[str, ...] = ("names", "aliases")

#: ``site_id``-style labels. These are shown to PMs as the site's handle, so
#: an unsupported one is as much an invention as an unsupported ``name``.
_ID_LABEL_FIELDS: tuple[str, ...] = ("site_id",)


def _normalize_name(text: str) -> str:
    """Dress-blind normalization, shared with site detection.

    Reuses :func:`app.core.site_detection._normalize` so a name compares
    equal to its source across the dressings this repo already treats as
    noise: casing, whitespace runs (NBSP included), punctuation, and
    hyphen/underscore/slash/period variants. One normalizer, one notion of
    "the same string" — a second one would rot out of sync with the first.
    """
    from app.core.site_detection import _normalize

    return _normalize(text or "")


def _atom_support_text(atom: Any) -> str:
    """The atom's own source text, normalized once for substring probing."""
    raw = str(getattr(atom, "raw_text", None) or "")
    norm = str(getattr(atom, "normalized_text", None) or "")
    return _normalize_name(f"{raw} {norm}")


def _name_is_supported(name: Any, support: str) -> bool:
    """Is ``name`` present, normalized, in the atom's own source text?

    Non-strings and empties are not names — they are left alone (this pass
    only ever removes what it can *prove* unsupported).
    """
    if not isinstance(name, str):
        return True
    probe = _normalize_name(name)
    if not probe:
        return True
    return probe in support


def strip_unsupported_names(atoms: list[Any]) -> int:
    """Remove display names that appear nowhere in the atom's source text.

    For every atom carrying a name-ish field (``name``/``facility_name``/
    ``display_name``/``site_name``, the ``names``/``aliases`` lists, and
    ``site_id``-style labels), assert the string appears — normalized — in
    that atom's own ``raw_text``/``normalized_text``. When it does not:

    * ``names``/``aliases`` lists are filtered down to the supported entries;
    * a scalar name falls back to a supported alias **already present on the
      atom** if one exists, and is otherwise deleted outright;
    * the atom is flagged ``unsupported_name_stripped`` and sent to review.

    The atom itself is never deleted and a replacement is never invented —
    this abstains on the *name*, not on the evidence. Mutates in place;
    returns the number of atoms changed.
    """
    from app.core.schemas import ReviewStatus

    changed = 0
    for atom in atoms:
        value = getattr(atom, "value", None)
        if not isinstance(value, dict) or not value:
            continue
        support = _atom_support_text(atom)
        if not support:
            continue

        touched = False

        # Pass A: filter alias lists down to what the source actually says.
        # Done first so the scalar fallback below can only ever land on a
        # name that survived this filter.
        supported_aliases: list[str] = []
        for field in _NAME_LIST_FIELDS:
            names = value.get(field)
            if not isinstance(names, list) or not names:
                continue
            kept = [n for n in names if _name_is_supported(n, support)]
            if len(kept) != len(names):
                value[field] = kept
                touched = True
            supported_aliases.extend(n for n in kept if isinstance(n, str) and n.strip())

        # Pass B: scalar display names and id-style labels.
        for field in _DISPLAY_NAME_FIELDS + _ID_LABEL_FIELDS:
            if field not in value:
                continue
            name = value.get(field)
            if _name_is_supported(name, support):
                # A supported scalar is itself a legitimate fallback target.
                if isinstance(name, str) and name.strip():
                    supported_aliases.append(name)
                continue
            replacement = next(
                (a for a in supported_aliases if _normalize_name(a)), None
            )
            if replacement is not None:
                value[field] = replacement
            else:
                del value[field]
            touched = True

        if touched:
            atom.value = value
            flags = list(getattr(atom, "review_flags", None) or [])
            if "unsupported_name_stripped" not in flags:
                atom.review_flags = sorted(set(flags + ["unsupported_name_stripped"]))
            if getattr(atom, "review_status", None) != ReviewStatus.needs_review:
                atom.review_status = ReviewStatus.needs_review
            changed += 1
    return changed


# ── invariant 2: authority cannot exceed what the source supports ──────
#
# Rank laundering: content that is not a customer document acquires a
# customer-document authority rank and outranks the real documents. Seen
# twice now — the "218 PM rows" incident, and the manifest ``context`` key
# carrying the pipeline's own prior output back into the compile as 1,123
# atoms at ``customer_current_authored``. Both are the same disease, so the
# rule is keyed on the general property ("is there a real artifact behind
# this claim, and is this text actually a document?"), never on a string.

#: A serialized key path: a dotted/indexed identifier chain glued directly to
#: a terminating colon, at the very start of the text.
#: ``context.prior_scope.x[3].id: v`` and ``artifacts[5].blob_url: https://…``
#: are structure. Prose is not: a sentence puts a space after its period
#: ("Main Office. Access: escorted"), so a segment that must begin with a
#: letter *immediately* after the dot is a strong discriminator. Segments may
#: carry internal spaces because real key paths interpolate names
#: (``…site_fields.South Warehouse.ceiling_substrate.required``).
_SERIALIZED_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?P<segments>(?:\.[A-Za-z_][A-Za-z0-9_ ]*|\[\d+\])+)"
    r"\s*:",
)

#: Dots a path needs to be structure on its own, absent a ``[N]`` index.
#: Two dots (three segments) is past what prose produces by accident.
_MIN_PATH_DOTS = 2

#: Tail segments that make a dotted token a filename or a hostname rather
#: than a key path — the one shape that otherwise collides with this test.
_NON_STRUCTURE_TAIL_SEGMENTS: frozenset[str] = frozenset({
    "com", "net", "org", "io", "gov", "edu", "co", "uk", "us", "ai", "dev",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "tsv",
    "txt", "json", "xml", "html", "htm", "md", "zip", "eml", "msg",
    "png", "jpg", "jpeg", "gif", "svg", "dwg", "rvt",
})


def _looks_like_serialized_structure(text: str) -> bool:
    """Is this atom text a serialized key path rather than document prose?

    Deterministic and universal — a shape test, not a vocabulary list. It
    deliberately does **not** know the string ``"context."``: the point is to
    foreclose the whole class, so any future key that round-trips machine
    output back into a compile is caught by the same rule.
    """
    m = _SERIALIZED_PATH_RE.match((text or "").lstrip())
    if not m:
        return False
    segments = m.group("segments")
    # An ``[N]`` index is decisive: no filename or hostname carries one, and
    # a key like ``site_registry[0].zip`` is a ZIP *code*, not a ZIP file.
    if "[" in segments:
        return True
    dots = segments.count(".")
    if dots < _MIN_PATH_DOTS:
        return False
    if dots == _MIN_PATH_DOTS:
        # Exactly three segments is also the shape of "report.final.pdf" and
        # "www.example.com" — the one collision worth carving out. Deeper
        # paths are key paths whatever their tail.
        tail = segments.rsplit(".", 1)[-1].strip().lower()
        if tail in _NON_STRUCTURE_TAIL_SEGMENTS:
            return False
    return True


#: Types that assert a deliverable or a commitment. A serialized key path may
#: never stand as one of these; it is floored to ``project_metadata``, which
#: is what a manifest key path actually is. Generic/retained types
#: (``scope_item``, ``entity``, ``site_note``) are left as-is — conservative:
#: keep the atom, drop the claim status.
_CLAIM_TYPE_NAMES = frozenset({
    "requirement",
    "acceptance_criterion",
    "quantity",
    "bom_line",
    "commercial_total",
    "pricing_assumption",
    "payment_term",
    "milestone_phase",
    "exclusion",
    "constraint",
    "assumption",
    "decision",
    "action_item",
    "risk",
    "physical_site",
    "service_line",
})


def _atom_source_artifact_ids(atom: Any) -> set[str]:
    """Every artifact this atom claims to come from (atom + its source refs)."""
    ids: set[str] = set()
    aid = getattr(atom, "artifact_id", None)
    if aid:
        ids.add(str(aid))
    for ref in (getattr(atom, "source_refs", None) or []):
        rid = getattr(ref, "artifact_id", None)
        if rid is None and isinstance(ref, dict):
            rid = ref.get("artifact_id")
        if rid:
            ids.add(str(rid))
    return ids


def _cap_authority(atom: Any, flag: str) -> bool:
    """Lower this atom to ``machine_extractor`` if it currently outranks it.

    Only ever demotes: an atom already at or below rank 40
    (``quoted_old_email``, ``deleted_text``) keeps its class, so this pass
    can never *raise* authority. Returns whether anything changed.
    """
    from app.core.authority import AUTHORITY_RANKS
    from app.core.schemas import AuthorityClass, ReviewStatus

    current = getattr(atom, "authority_class", None)
    ceiling = AUTHORITY_RANKS[AuthorityClass.machine_extractor]
    if AUTHORITY_RANKS.get(current, ceiling) <= ceiling:
        return False
    atom.authority_class = AuthorityClass.machine_extractor
    flags = list(getattr(atom, "review_flags", None) or [])
    if flag not in flags:
        atom.review_flags = sorted(set(flags + [flag]))
    if getattr(atom, "review_status", None) != ReviewStatus.needs_review:
        atom.review_status = ReviewStatus.needs_review
    return True


def cap_authority_to_source(atoms: list[Any], *, artifact_ids: Any) -> int:
    """Cap authority at ``machine_extractor`` when the source cannot carry it.

    Two deterministic conditions, both universal:

    1. **Unresolved source** — the atom's ``artifact_id`` / ``source_refs``
       name no artifact in ``artifact_ids`` (the compile's real artifact
       set). Nothing on disk backs the claim, so it may not outrank a
       machine guess.
    2. **Serialized structure** — the atom's text is a JSON key path rather
       than document prose (see :func:`_looks_like_serialized_structure`).
       Structure is not evidence. Such an atom is additionally floored from
       any *claim* type to ``project_metadata``; the atom and its text are
       kept, only its standing is dropped.

    Never raises authority and never deletes an atom. Mutates in place;
    returns the number of atoms capped.
    """
    from app.core.schemas import AtomType

    known = {str(a) for a in (artifact_ids or ()) if a}
    floor_type = getattr(AtomType, "project_metadata", None)

    capped = 0
    for atom in atoms:
        changed = False

        # 1. Does a real artifact stand behind this atom?
        if known:
            sources = _atom_source_artifact_ids(atom)
            if not sources or not (sources & known):
                changed |= _cap_authority(atom, "authority_capped_unresolved_source")

        # 2. Is the text a document at all, or just serialized structure?
        if _looks_like_serialized_structure(_atom_text(atom)):
            changed |= _cap_authority(atom, "authority_capped_serialized_structure")
            flags = list(getattr(atom, "review_flags", None) or [])
            if "serialized_structure_not_evidence" not in flags:
                atom.review_flags = sorted(
                    set(flags + ["serialized_structure_not_evidence"])
                )
                changed = True
            if floor_type is not None and _atom_type_str(atom) in _CLAIM_TYPE_NAMES:
                atom.atom_type = floor_type
                changed = True

        if changed:
            capped += 1
    return capped


# ── invariant 4: a file format is not a contract ───────────────────────
#
# ``contractual_scope`` (rank 100) means "the signed agreement governs this",
# and it outranks ``pm_confirmed`` (95) — a PM's own explicit answer. Several
# parsers hardcode it (xlsx_parser, markdown_parser), so the FILE FORMAT ends
# up conferring contract authority: on deal 1e130077, 294 atoms from a ROM
# (a Rough Order of Magnitude *estimate*) and an internal deal kit held rank
# 100, and a PM could not outvote a number in a scratch spreadsheet.
#
# This is the same seam law as the rest of this module: one representation
# (the file's format) standing in for a different truth (the document's
# authority). Rather than edit seven hardcoded call sites, the rule is stated
# once here — parsers may keep *proposing* contractual_scope; this gate
# decides whether the proposal survives.
#
# This is floor-level policy, not a per-customer heuristic: no customer
# terminology, no per-deal strings, applies to every parser's output. See
# ``demote_unearned_contract_authority`` for the caveat on its successor.

#: Filename/text markers that FORBID rank 100 outright. A document that
#: announces itself as an estimate, a draft, a working sheet or a quote is
#: not the signed agreement — whatever a parser proposed for it.
_NON_CONTRACT_MARKERS: tuple[str, ...] = (
    "rom", "rough order of magnitude",
    "estimate", "estimated", "budgetary", "budget",
    "draft", "deal kit", "dealkit",
    "calc", "calculation", "worksheet", "workbook",
    "quote", "quotation", "proposal",
    "pricing sheet", "price sheet", "rate sheet",
    "template", "scratch", "sample", "example",
)

#: Markers that, on their own, evidence a real contract.
_CONTRACT_MARKERS: tuple[str, ...] = (
    "agreement", "contract", "executed",
    "terms and conditions", "master service", "master services",
    "purchase order", "po number", "po no",
    "amendment", "addendum",
)

#: "Statement of work" only earns rank 100 with an execution block — an
#: unsigned SOW draft is a proposal.
_SOW_MARKERS: tuple[str, ...] = ("statement of work", "sow")

_SIGNATURE_MARKERS: tuple[str, ...] = (
    "signature", "signed", "signed by", "authorized signature",
    "authorized representative", "accepted by", "acceptance by",
    "in witness whereof", "duly executed", "countersigned",
)

#: Non-contract markers that additionally say *what the document is*: a
#: priced or estimated working document, i.e. exactly what ``vendor_quote``
#: means. Atoms from these land at 65 rather than 90.
_PRICED_WORKING_DOC_MARKERS: frozenset[str] = frozenset({
    "rom", "rough order of magnitude", "estimate", "estimated",
    "budgetary", "budget", "quote", "quotation", "proposal",
    "pricing sheet", "price sheet", "rate sheet",
    "deal kit", "dealkit", "calc", "calculation",
    "worksheet", "workbook",
})


def _marker_probes(*parts: str) -> tuple[str, str]:
    """Return (token-separated, squeezed) dress-blind probes for ``parts``.

    The token form supports word-boundary matching ("rom" must not fire
    inside "eeprom"); the squeezed form lets a multi-word marker match its
    run-together dress ("DEALKIT" -> "dealkit" matches "deal kit").
    """
    norm = _normalize_name(" ".join(p for p in parts if p))
    return norm, norm.replace(" ", "")


def _marker_hit(marker: str, norm: str, squeezed: str) -> bool:
    """Word-boundary marker match, dress-blind.

    Single-word markers match only on token boundaries. Multi-word markers
    additionally match their squeezed form, so "deal kit" / "deal_kit" /
    "DEALKIT" are one marker.
    """
    m = _normalize_name(marker)
    if not m:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(m)}(?![a-z0-9])", norm):
        return True
    if " " in m and m.replace(" ", "") in squeezed:
        return True
    return False


def classify_document_contract_evidence(filename: str = "", text: str = "") -> str:
    """Classify a document as ``"contract"`` / ``"non_contract"`` / ``"unknown"``.

    Deterministic, dress-blind, universal — markers only, no customer
    terminology. Negative evidence wins: a "Draft Master Services Agreement"
    is a draft, not an executed agreement.
    """
    norm, squeezed = _marker_probes(filename, text)
    if not norm:
        return "unknown"

    if any(_marker_hit(m, norm, squeezed) for m in _NON_CONTRACT_MARKERS):
        return "non_contract"
    if any(_marker_hit(m, norm, squeezed) for m in _CONTRACT_MARKERS):
        return "contract"
    if any(_marker_hit(m, norm, squeezed) for m in _SOW_MARKERS) and any(
        _marker_hit(m, norm, squeezed) for m in _SIGNATURE_MARKERS
    ):
        return "contract"
    return "unknown"


def _document_index(documents: Any) -> dict[str, tuple[str, str]]:
    """Normalize ``documents`` into {artifact_id: (filename, text)}.

    Accepts what each caller naturally has: the compiler's
    ``{artifact_id: Path}`` map, a ``{artifact_id: "filename"}`` map, a
    ``{artifact_id: {"filename":..., "text":...}}`` map, or a sequence of
    document objects/dicts carrying ``artifact_id``/``filename``.
    """
    out: dict[str, tuple[str, str]] = {}
    if not documents:
        return out

    def _put(aid: Any, filename: Any, text: Any = "") -> None:
        if aid:
            out[str(aid)] = (str(filename or ""), str(text or ""))

    if isinstance(documents, dict):
        for aid, doc in documents.items():
            if isinstance(doc, dict):
                _put(aid, doc.get("filename") or doc.get("name"), doc.get("text"))
            else:
                # Path or plain string: the basename is the filename.
                name = getattr(doc, "name", None) or str(doc or "")
                _put(aid, name)
        return out

    for doc in documents:
        if isinstance(doc, dict):
            _put(doc.get("artifact_id"), doc.get("filename") or doc.get("name"),
                 doc.get("text"))
        else:
            _put(getattr(doc, "artifact_id", None),
                 getattr(doc, "filename", None) or getattr(doc, "name", None),
                 getattr(doc, "text", None))
    return out


def demote_unearned_contract_authority(atoms: list[Any], *, documents: Any) -> int:
    """Strip ``contractual_scope`` from atoms whose document is not a contract.

    An atom may hold rank 100 only when its source document shows positive
    contract evidence (see :func:`classify_document_contract_evidence`).
    Otherwise it is demoted — never silently, always flagged, and never
    below what the document would otherwise earn:

    * a priced/estimated working document (ROM, deal kit, quote, calc sheet)
      lands at ``vendor_quote`` (65) — which is what such a document *is*;
    * anything else lands at ``customer_current_authored`` (90).

    Both sit below ``pm_confirmed`` (95), so a PM's explicit answer once
    again outranks a number in an internal spreadsheet.

    This is **floor-level policy, not a per-customer heuristic**: it encodes
    what a document has to show to govern a SOW, in universal vocabulary,
    and it applies to every parser's output rather than to one parser's
    hardcoded constant.

    Mutates in place; returns the number of atoms demoted.
    """
    from app.core.authority import AUTHORITY_RANKS
    from app.core.schemas import AuthorityClass, ReviewStatus

    index = _document_index(documents)
    if not index:
        return 0

    verdicts: dict[str, tuple[str, bool]] = {}
    for aid, (filename, text) in index.items():
        verdict = classify_document_contract_evidence(filename, text)
        norm, squeezed = _marker_probes(filename, text)
        priced = any(
            _marker_hit(m, norm, squeezed) for m in _PRICED_WORKING_DOC_MARKERS
        )
        verdicts[aid] = (verdict, priced)

    demoted = 0
    for atom in atoms:
        if getattr(atom, "authority_class", None) != AuthorityClass.contractual_scope:
            continue
        sources = _atom_source_artifact_ids(atom)
        known = [verdicts[s] for s in sorted(sources) if s in verdicts]
        if not known:
            continue  # document unknown to this caller — leave it alone
        # Any source document that IS a contract earns the atom its rank.
        if any(verdict == "contract" for verdict, _ in known):
            continue
        priced = any(is_priced for _, is_priced in known)
        target = (
            AuthorityClass.vendor_quote if priced
            else AuthorityClass.customer_current_authored
        )
        # Only ever demote.
        if AUTHORITY_RANKS[target] >= AUTHORITY_RANKS[AuthorityClass.contractual_scope]:
            continue
        atom.authority_class = target
        flags = list(getattr(atom, "review_flags", None) or [])
        if "unearned_contract_authority_demoted" not in flags:
            atom.review_flags = sorted(
                set(flags + ["unearned_contract_authority_demoted"])
            )
        if getattr(atom, "review_status", None) != ReviewStatus.needs_review:
            atom.review_status = ReviewStatus.needs_review
        demoted += 1
    return demoted


# ── invariant 3b: a number in a naming context is not a count ──────────
#
# "Building 704" names a building; it does not order 704 of anything. The
# mirror of ``_MEASURE_WORDS`` (which guards the token *after* the number):
# these guard the token immediately *before* it. Adjacency is the
# same-clause test — "704 drops in building 12" still counts 704 drops,
# because the naming noun does not touch the 704.
_NAMING_CONTEXT_WORDS: frozenset[str] = frozenset({
    "building", "bldg", "blg", "bld",
    "suite", "ste",
    "room", "rm",
    "floor", "level",
    "wing", "zone", "sector", "block",
    "apartment", "apt",
    "unit", "space",
    "section", "phase",
    "number", "no", "num",
    "lot", "parcel", "gate", "dock", "bay",
})

_NAMING_CONTEXT_RE = re.compile(
    r"(?:\b(?:" + "|".join(sorted(_NAMING_CONTEXT_WORDS)) + r")\.?|#)"
    r"[\s\-:]*$",
    re.IGNORECASE,
)


def number_is_naming_label(text: str, number_start: int) -> bool:
    """Does the number at ``number_start`` name something rather than count it?

    True when the immediately preceding token is a naming noun ("building
    704", "bldg. 704", "suite 210", "room 12", "# 704"). Deterministic and
    universal — a grammatical position, not a per-deal alias list.
    """
    if number_start <= 0:
        return False
    return bool(_NAMING_CONTEXT_RE.search((text or "")[:number_start]))


def apply_type_sanity(
    atoms: list[Any],
    *,
    project_id: str,
    artifact_ids: Any = None,
    documents: Any = None,
) -> tuple[list[Any], int, int]:
    """Run every pass. Returns (atoms, demoted_count, surfaced_count).

    ``atoms`` is returned (possibly extended with surfaced quantities).
    ``artifact_ids`` is the compile's real artifact set; when omitted the
    unresolved-source half of :func:`cap_authority_to_source` is skipped
    (the serialized-structure half still runs) so callers that do not know
    the artifact set cannot accidentally cap every atom. ``documents`` maps
    those artifacts to filenames/text for
    :func:`demote_unearned_contract_authority`; omitted, that pass is a
    no-op rather than a guess.
    """
    demoted = demote_nondeliverable_quantities(atoms)
    demoted += demote_manifest_metadata_bom_lines(atoms)
    demoted += demote_email_include_list_microtasks(atoms)
    demoted += demote_customer_quote_requirements(atoms)
    # Promotion gate: a claim may not outrank its source, and a name the
    # source never states is not a name. Both demote/strip only.
    demoted += strip_unsupported_names(atoms)
    demoted += cap_authority_to_source(atoms, artifact_ids=artifact_ids)
    demoted += demote_unearned_contract_authority(atoms, documents=documents)
    # Universal scrub: strip junk quantity: keys off *any* atom (commercial
    # totals, pricing assumptions) — not just quantity-typed ones — so the
    # entity resolver never promotes "260 pmo cost" into a quantity entity.
    scrub_nondeliverable_quantity_keys(atoms)
    surfaced = surface_headline_quantities(atoms, project_id=project_id)
    if surfaced:
        atoms = atoms + surfaced
    return atoms, demoted, len(surfaced)


__all__ = [
    "apply_type_sanity",
    "cap_authority_to_source",
    "classify_document_contract_evidence",
    "demote_unearned_contract_authority",
    "number_is_naming_label",
    "strip_unsupported_names",
    "demote_customer_quote_requirements",
    "demote_email_include_list_microtasks",
    "demote_manifest_metadata_bom_lines",
    "demote_nondeliverable_quantities",
    "scrub_nondeliverable_quantity_keys",
    "surface_headline_quantities",
]
