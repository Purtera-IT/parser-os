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
"""

from __future__ import annotations

import re
from typing import Any

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
    for sentence in _SENTENCE_SPLIT_RE.split(text):
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
    """Re-type mistyped ``task`` atoms that are email Include-list micro-labels.

    The typed classifier can promote ``Okta integration`` to ``task`` when the
    email parser stripped the bullet chrome; those belong in the umbrella
    quote-line bucket, not as standalone child tasks."""
    from app.core.schemas import AtomType

    demoted = 0
    for atom in atoms:
        if _atom_type_str(atom) != "task":
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        if val.get("list_section") == "include" and val.get("kind") == "email_body_line":
            atom.atom_type = AtomType.scope_item
            demoted += 1
    return demoted


def apply_type_sanity(atoms: list[Any], *, project_id: str) -> tuple[list[Any], int, int]:
    """Run both passes. Returns (atoms, demoted_count, surfaced_count).

    ``atoms`` is returned (possibly extended with surfaced quantities).
    """
    demoted = demote_nondeliverable_quantities(atoms)
    demoted += demote_manifest_metadata_bom_lines(atoms)
    demoted += demote_email_include_list_microtasks(atoms)
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
    "demote_manifest_metadata_bom_lines",
    "demote_nondeliverable_quantities",
    "scrub_nondeliverable_quantity_keys",
    "surface_headline_quantities",
]
