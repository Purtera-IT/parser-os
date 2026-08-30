"""Base health gates: measure the compiled corpus and FAIL when it rots.

Why this file exists
--------------------
On 2026-08-27 two hand-run audits found real, large defects: 4,135 atoms
sitting at rank-100 contract authority with no verified support (63% of
deals), and 964 of 1,100 physical_site atoms carrying a street address with
no city. Both were found by a person deciding to look. Worse, a fix that
shipped the same morning silently dropped ~50 real sites from one deal
(438 -> 388) and was caught only because someone re-measured by hand.

A defect that is only visible when a human chooses to re-measure is not
monitored. This CLI turns those one-off audits into committed, thresholded
numbers that exit nonzero, so a regression becomes a red job within hours
instead of a weird brief noticed weeks later.

Doctrine
--------
**A broken audit must never look like a clean base.** This is the same
silent-zero rule the repo already holds for infra (docs/IMAGE_GATE_LOOP.md,
doctrine 2). Every path that cannot measure -- missing ``az``, no auth, a
download failure, an unparseable or atom-less envelope -- is reported as an
ERROR and exits nonzero. Zero findings from zero envelopes is never a pass.

**The audit does not import the gate it audits.** The measurement rules
below are reimplemented from the invariants in ``app/core/atom_type_sanity``
rather than imported from it. An audit that calls the same function it is
checking cannot detect that function breaking -- both sides would move
together and the metric would stay green. Reimplementation is the point: the
two definitions drifting apart is itself a finding.

Metrics, thresholds, and the rationale for each threshold live in
``METRICS`` below and are explained for humans in ``docs/BASE_HEALTH.md``.

Usage
-----
    python tools/base_health.py                       # table, 40-deal sample
    python tools/base_health.py --sample 80 --json
    python tools/base_health.py --update-baseline     # rewrite the baseline
    python tools/base_health.py --baseline other.json

Exit codes
----------
    0  every metric within threshold
    1  at least one threshold breached
    2  could not measure (no az / no auth / nothing downloaded) -- NOT a pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ── blob coordinates ────────────────────────────────────────────────
# Non-customer-specific: an account, a container, and a deal-id-keyed path.
ACCOUNT = "purpulsedevstg01"
CONTAINER = "orbitbrief-artifacts"
ENVELOPE_SUFFIX = "orbitbrief/latest/envelope.json"
BLOB_PREFIX = "deals/"

#: Envelopes larger than this are skipped by default. A 30MB+ envelope is a
#: compile pathology of its own and blows the streaming budget; ``--max-mb``
#: raises it deliberately.
DEFAULT_MAX_MB = 30

DEFAULT_SAMPLE = 40
DEFAULT_BASELINE = Path(__file__).with_name("base_health_baseline.json")

EXIT_OK, EXIT_BREACH, EXIT_CANNOT_MEASURE = 0, 1, 2


# ── authority ranks ─────────────────────────────────────────────────
# Mirrors app.core.schemas.AuthorityClass ordering. Duplicated on purpose
# (see "the audit does not import the gate it audits" above); the constant is
# an enum ordering, not logic, so the duplication is cheap and stable.
RANK: dict[str, int] = {
    "contractual_scope": 100,
    "pm_confirmed": 95,
    "customer_current_authored": 90,
    "approved_site_roster": 80,
    "vendor_quote": 65,
    "meeting_note": 50,
    "machine_extractor": 40,
    "quoted_old_email": 30,
    "deleted_text": 10,
}
RANK_100 = "contractual_scope"
#: "high authority" for the self-ingestion metric: approved_site_roster and up.
HIGH_AUTHORITY_FLOOR = RANK["approved_site_roster"]

SITE_TYPE = "physical_site"


# ── shape tests (reimplemented invariants) ──────────────────────────

def normalize(text: Any) -> str:
    """Dress-blind normalization: casing, punctuation, and separators.

    Same notion of "the same string" as ``app.core.site_detection._normalize``
    -- hyphen/underscore/slash/period become spaces, other punctuation is
    stripped, whitespace collapses. Reimplemented, not imported, so a change
    to the shipped normalizer shows up here as a divergence rather than being
    silently absorbed.
    """
    s = str(text or "").lower().strip()
    s = re.sub(r"[\-_/.]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


#: Scalar structured fields that carry a human-facing display name.
DISPLAY_NAME_FIELDS = ("name", "facility_name", "display_name", "site_name")
#: Lists of alternate names for the same thing.
NAME_LIST_FIELDS = ("names", "aliases")
#: Handles shown to PMs as the site's identity; an invented one is a
#: fabrication as much as an invented ``name``.
ID_LABEL_FIELDS = ("site_id",)

_SERIALIZED_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?P<segments>(?:\.[A-Za-z_][A-Za-z0-9_ ]*|\[\d+\])+)"
    r"\s*:",
)
_MIN_PATH_DOTS = 2
#: Tails that make a dotted token a filename or hostname, not a key path --
#: the one shape that otherwise collides with the structure test.
_NON_STRUCTURE_TAILS = frozenset({
    "com", "net", "org", "io", "gov", "edu", "co", "uk", "us", "ai", "dev",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "tsv",
    "txt", "json", "xml", "html", "htm", "md", "zip", "eml", "msg",
    "png", "jpg", "jpeg", "gif", "svg", "dwg", "rvt",
})


def looks_like_serialized_structure(text: Any) -> bool:
    """Is this atom text a serialized key path rather than document prose?

    A shape test, not a vocabulary list: it does not know the string
    ``context.``, because the point is to foreclose the whole class of
    machine output round-tripping back into a compile.
    """
    m = _SERIALIZED_PATH_RE.match(str(text or "").lstrip())
    if not m:
        return False
    segments = m.group("segments")
    if "[" in segments:  # an [N] index is decisive; no filename carries one
        return True
    dots = segments.count(".")
    if dots < _MIN_PATH_DOTS:
        return False
    if dots == _MIN_PATH_DOTS:
        # Three segments is also "report.final.pdf" / "www.example.com".
        if segments.rsplit(".", 1)[-1].strip().lower() in _NON_STRUCTURE_TAILS:
            return False
    return True


#: Filename markers that say a document is a working estimate, not a contract.
NON_CONTRACT_MARKERS = (
    "rom", "rough order of magnitude",
    "estimate", "estimated", "budgetary", "budget",
    "draft", "deal kit", "dealkit",
    "calc", "calculation", "worksheet", "workbook",
    "quote", "quotation", "proposal",
    "pricing sheet", "price sheet", "rate sheet",
    "template", "scratch", "sample", "example",
)
#: Markers that, on their own, evidence a real contract. A filename carrying
#: one of these is not counted even if it also carries a non-contract word --
#: "MSA - Amendment 2 (draft cover).pdf" is an agreement, and flagging it
#: would train people to ignore this metric.
CONTRACT_MARKERS = (
    "agreement", "contract", "executed",
    "terms and conditions", "master service", "master services",
    "purchase order", "po number", "po no",
    "amendment", "addendum",
    "signed", "countersigned", "fully executed",
)


def _has_marker(norm_name: str, markers: Iterable[str]) -> str | None:
    """First marker present as whole tokens in an already-normalized string."""
    padded = f" {norm_name} "
    for marker in markers:
        if f" {normalize(marker)} " in padded:
            return marker
    return None


def filename_is_noncontract(filename: Any) -> str | None:
    """Marker naming this file a non-contract source, or None.

    Returns None when the filename also carries contract evidence.
    """
    norm = normalize(filename)
    if not norm:
        return None
    if _has_marker(norm, CONTRACT_MARKERS):
        return None
    return _has_marker(norm, NON_CONTRACT_MARKERS)


#: A ``key: value | key: value`` string is an extractor-composed SUMMARY of a
#: source row, not the row itself.
_COMPOSED_PAIR_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_ ]*:\s*\S")


def is_composed_field_summary(text: Any) -> bool:
    """Is this atom text a machine-composed field summary, not source prose?

    Roster extractors write ``site_id: HC-65 | facility: HC 65 | address:
    6655 US 23`` into ``text``. That string is *derived from* the source row,
    not a copy of it -- the roster's facility-name column is consumed into
    ``structured`` and never appears in ``text``.

    This matters because the envelope does not serialize an atom's
    ``raw_text``: ``text`` is the only evidence an auditor has. For these
    atoms the fabricated-name question is therefore UNDECIDABLE from the
    envelope, and checking anyway produces pure noise -- on one real deal it
    reported all 1,518 site names as fabrications when every one of them was
    a verbatim string in the source workbook.

    So these atoms are excluded from ``fabricated_names`` and counted under
    ``names_undecidable`` instead. Suppressing a check silently would be the
    exact failure this tool exists to prevent, so the undecidable population
    is printed on every run. The real fix is upstream: serialize the source
    row (or a replayable source locator) on the atom, and this exclusion can
    be deleted.
    """
    parts = str(text or "").split("|")
    return len(parts) >= 2 and sum(1 for p in parts if _COMPOSED_PAIR_RE.match(p)) >= 2


def display_names(structured: Any) -> list[str]:
    """Every human-facing name string on one atom's structured payload.

    Top-level keys only -- never recurse. A nested ``label`` is usually a
    classifier verdict (``facility_label.label == "keep_facility"``), not a
    name, and counting it would make this metric noise.
    """
    if not isinstance(structured, dict):
        return []
    out: list[str] = []
    for f in DISPLAY_NAME_FIELDS + ID_LABEL_FIELDS:
        v = structured.get(f)
        if isinstance(v, str) and v.strip():
            out.append(v)
    for f in NAME_LIST_FIELDS:
        v = structured.get(f)
        if isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str) and x.strip())
    return out


def _filled(value: Any) -> bool:
    """Is a structured field actually populated (not None/""/[]/{})?"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


# ── metric specs ────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    #: "corpus_count"  -> raw corpus-wide count must be <= threshold
    #: "corpus_ratio"  -> count/denominator must be <= threshold
    #: "per_deal_drift"-> handled separately against the baseline
    kind: str
    threshold: float
    rationale: str
    denominator: str = ""


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="unsupported_contract_authority",
        label="unsupported contract authority",
        kind="corpus_count",
        # 4,135 corpus-wide this morning; the shipped demotion gate took it
        # to 1. The bar is 5, not 0, because a handful of atoms legitimately
        # sit at rank 100 while their replay is still pending -- but 5 is far
        # enough below the observed post-gate value that any real regression
        # in the demotion pass clears it immediately.
        threshold=5,
        rationale="rank-100 with no verified support; was 4,135, gate took it to 1",
    ),
    MetricSpec(
        key="rank100_from_noncontract_source",
        label="rank-100 from non-contract source",
        kind="corpus_count",
        # Zero-tolerance: a ROM spreadsheet or a deal-kit calc can never be
        # contractual scope. There is no legitimate instance of this, so any
        # count above zero is a defect, not a tuning question.
        threshold=0,
        rationale="an estimate/quote/calc file can never carry contract authority",
    ),
    MetricSpec(
        key="output_document_as_evidence",
        label="our own output admitted as evidence",
        kind="corpus_count",
        # A Deal Kit, quote, proposal or SOW is an ANSWER. Reading one as
        # evidence teaches the head that produces those answers to copy itself,
        # and the resulting eval score is meaningless because the label was in
        # the input. Measured over 1,114 classified documents, ~61% of the
        # document corpus is output -- so this is not a rare edge, it is the
        # single largest contamination route into the quoting heads. There is no
        # legitimate instance: zero, permanently.
        threshold=0,
        rationale="a Deal Kit or SOW is the answer; reading it as evidence teaches the model to copy itself",
    ),
    MetricSpec(
        key="test_fixture_admitted",
        label="mock/test document admitted",
        kind="corpus_count",
        # Four mock documents exist in the corpus ("Mock Document | Fictional
        # data", "FOR HUMAN REVIEWER ONLY"). Before the TEST_FIXTURE type
        # existed, one of them was already routed to `label` -- fictional data
        # about to become a TRAINING TARGET. Anything above zero means invented
        # content is shaping a real model.
        threshold=0,
        rationale="fictional documents must never reach evidence or training labels",
    ),
    MetricSpec(
        key="fabricated_names",
        label="fabricated display names",
        kind="corpus_count",
        # The shipped invariant: a name shown to a PM must be a string a
        # human actually wrote in the document the atom came from. A
        # plausible fabrication is worse than no name, because it survives
        # review. Zero, permanently.
        threshold=0,
        rationale="a shown name must appear in its own source text (shipped invariant)",
    ),
    MetricSpec(
        key="street_without_city",
        label="street with no city",
        kind="corpus_ratio",
        denominator="site_atoms",
        # THRESHOLD IS DELIBERATELY NOT 2%. See docs/BASE_HEALTH.md.
        # This is the unambiguous site-geo bug signature: you cannot parse a
        # street out of a roster row and not know its city -- they are
        # adjacent columns in the same row. The target is 2%.
        #
        # It is set at 65% because most envelopes ON BLOB were produced by an
        # OLDER DEPLOYED BUILD that predates the roster geo fix. Measured
        # 2026-08-27 over 38 deals: 56% corpus-wide (233/416 site atoms),
        # while a locally recompiled envelope of the same deal shows 0.3%
        # (1/388) -- the fix works, the corpus is just stale. A 2% threshold
        # today would be permanently red, and a permanently red gate is an
        # ignored gate. 65% leaves headroom over the measured 56% while still
        # catching a catastrophic regression.
        #
        # RECOMPILE REQUIREMENT: this threshold is a placeholder and must not
        # outlive the stale corpus. Once the corpus is recompiled on a build
        # containing the roster geo fix (app/parsers/site_roster_extractor),
        # set this to 0.02 and delete this paragraph.
        threshold=0.65,
        rationale="street parsed but city empty; capped at the old-build corpus level pending recompile",
    ),
    MetricSpec(
        key="self_ingested_high_authority",
        label="self-ingested structure at high authority",
        kind="corpus_count",
        # Serialized key paths are our own machine output read back in as if
        # a customer had written it. At rank >= approved_site_roster it
        # outranks real documents. Zero: there is no correct instance.
        threshold=0,
        rationale="serialized key paths must never outrank real documents",
    ),
    MetricSpec(
        key="site_count_drift",
        label="site count drift vs baseline",
        kind="per_deal_drift",
        # +/-5% per deal, BOTH directions. A fix that silently drops real
        # sites is exactly as bad as one that invents them -- this is the
        # check that would have caught the 438 -> 388 store loss the morning
        # this file was written. An absolute floor of 2 sites keeps tiny
        # deals (3 sites -> 4) from firing on rounding.
        threshold=0.05,
        rationale="both directions: a silent site LOSS is as bad as an invented site",
    ),
)

#: A drift smaller than this many sites never breaches, whatever the percent.
DRIFT_ABS_FLOOR = 2


# ── per-envelope measurement (pure; unit-tested without blob) ───────

@dataclass
class DealMeasure:
    deal: str
    atoms: int = 0
    site_atoms: int = 0
    #: Names the envelope carries no evidence for either way. Never silently
    #: dropped -- reported alongside the metrics.
    names_undecidable: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    #: Small human-readable examples, for the failure message.
    examples: dict[str, list[str]] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class EnvelopeError(ValueError):
    """An envelope that cannot be measured. Never silently a zero."""


def measure_envelope(envelope: Any, deal: str = "?") -> DealMeasure:
    """Measure one parsed envelope. Raises EnvelopeError if unmeasurable.

    Refusing to measure is a first-class outcome: an envelope that is not a
    dict, carries no ``atoms`` key, or carries an empty atom list is a broken
    compile. Returning zeros for it would make a dead pipeline indistinguish-
    able from a clean base -- which is the exact failure this tool exists to
    prevent.
    """
    if not isinstance(envelope, dict):
        raise EnvelopeError(f"envelope is {type(envelope).__name__}, not an object")
    if "atoms" not in envelope:
        raise EnvelopeError("envelope has no 'atoms' key")
    atoms = envelope.get("atoms")
    if not isinstance(atoms, list):
        raise EnvelopeError(f"'atoms' is {type(atoms).__name__}, not a list")
    if not atoms:
        raise EnvelopeError("envelope has zero atoms (broken compile, not a clean deal)")

    documents = envelope.get("documents") or []
    if not isinstance(documents, list):
        documents = []
    filenames = {
        d.get("artifact_id"): str(d.get("filename") or "")
        for d in documents
        if isinstance(d, dict) and d.get("artifact_id")
    }

    m = DealMeasure(deal=deal, atoms=len(atoms))
    counts = {s.key: 0 for s in METRICS if s.kind != "per_deal_drift"}
    examples: dict[str, list[str]] = {k: [] for k in counts}

    def note(key: str, text: str) -> None:
        counts[key] += 1
        if len(examples[key]) < 3:
            examples[key].append(text[:150])

    # Documents carry a `lifecycle` set by app.core.document_lifecycle: what the
    # document is, and who may read it. Absent means never classified, which is
    # not a failure here -- unknown content quarantines elsewhere. What IS a
    # failure is a classified document routed against its own stage.
    OUTPUT_STAGES = {"QUOTED_OUTPUT", "CONTRACTED"}
    for d in documents:
        if not isinstance(d, dict):
            continue
        life = d.get("lifecycle")
        if not isinstance(life, dict):
            continue
        name = str(d.get("filename") or "?")
        stage = str(life.get("stage") or "")
        adm = str(life.get("admissible_for") or "")
        dtype = str(life.get("type") or "")
        if stage in OUTPUT_STAGES and adm == "evidence":
            note("output_document_as_evidence", f"{name} [{dtype}] read as evidence")
        if dtype == "TEST_FIXTURE" and adm not in ("quarantine", ""):
            note("test_fixture_admitted", f"{name} admitted as {adm}")

    for a in atoms:
        if not isinstance(a, dict):
            continue
        authority = str(a.get("authority_class") or "")
        rank = RANK.get(authority, 0)
        text = str(a.get("text") or "")
        structured = a.get("structured")
        atom_id = str(a.get("id") or "?")

        # 1. rank-100 without verified support
        if authority == RANK_100 and str(a.get("verified") or "") != "verified":
            note("unsupported_contract_authority",
                 f"{atom_id} verified={a.get('verified')!r} :: {text}")

        # 2. rank-100 sourced from a document whose NAME says it is not one
        if authority == RANK_100:
            fname = filenames.get(a.get("artifact_id"), "")
            marker = filename_is_noncontract(fname)
            if marker:
                note("rank100_from_noncontract_source",
                     f"{atom_id} marker={marker!r} <- {fname}")

        # 3. names absent from the atom's own source text
        names = display_names(structured)
        if names and is_composed_field_summary(text):
            # No verbatim source in the envelope -> no verdict either way.
            m.names_undecidable += len(names)
        else:
            support = normalize(text)
            for name in names:
                probe = normalize(name)
                if probe and probe not in support:
                    note("fabricated_names", f"{atom_id} {name!r} not in its own source text")

        # 4. self-ingested serialized structure at high authority
        if rank >= HIGH_AUTHORITY_FLOOR and looks_like_serialized_structure(text):
            note("self_ingested_high_authority", f"{atom_id} [{authority}] :: {text}")

        # 5. site geo: a street with no city
        if str(a.get("atom_type") or "") == SITE_TYPE:
            m.site_atoms += 1
            st = structured if isinstance(structured, dict) else {}
            if _filled(st.get("street_address")) and not _filled(st.get("city")):
                note("street_without_city", f"{atom_id} street={st.get('street_address')!r} city=<empty>")

    m.counts = counts
    m.examples = {k: v for k, v in examples.items() if v}
    return m


# ── corpus roll-up and thresholds ───────────────────────────────────

@dataclass
class Finding:
    metric: str
    value: float
    threshold: float
    breached: bool
    display: str
    detail: str = ""
    deals: list[str] = field(default_factory=list)


def drift_breaches(measures: list[DealMeasure], baseline: dict[str, Any]) -> tuple[list[str], int]:
    """Deals whose site count moved beyond tolerance, and how many were checked.

    Flags BOTH directions on purpose. Deals absent from the baseline are not
    breaches (they are new); they are reported separately by the caller.
    """
    deals_baseline = (baseline or {}).get("deals") or {}
    offenders, checked = [], 0
    for m in measures:
        if not m.ok:
            continue
        entry = deals_baseline.get(m.deal)
        if not isinstance(entry, dict) or "site_atoms" not in entry:
            continue
        checked += 1
        expected = int(entry["site_atoms"])
        delta = m.site_atoms - expected
        tolerance = max(DRIFT_ABS_FLOOR, expected * METRIC_BY_KEY["site_count_drift"].threshold)
        if abs(delta) > tolerance:
            pct = (delta / expected * 100) if expected else float("inf")
            direction = "LOST" if delta < 0 else "GAINED"
            offenders.append(
                f"{m.deal[:8]} {direction} {abs(delta)} sites "
                f"({expected} -> {m.site_atoms}, {pct:+.1f}%)"
            )
    return offenders, checked


METRIC_BY_KEY = {s.key: s for s in METRICS}


def evaluate(measures: list[DealMeasure], baseline: dict[str, Any] | None) -> list[Finding]:
    """Apply every threshold to the corpus roll-up."""
    good = [m for m in measures if m.ok]
    totals = {s.key: 0 for s in METRICS if s.kind != "per_deal_drift"}
    for m in good:
        for k in totals:
            totals[k] += m.counts.get(k, 0)
    site_atoms = sum(m.site_atoms for m in good)

    findings: list[Finding] = []
    for spec in METRICS:
        if spec.kind == "corpus_count":
            value = totals[spec.key]
            offenders = sorted(
                (m for m in good if m.counts.get(spec.key)),
                key=lambda m: -m.counts[spec.key],
            )
            findings.append(Finding(
                metric=spec.key,
                value=value,
                threshold=spec.threshold,
                breached=value > spec.threshold,
                display=f"{value:,}",
                detail=f"<= {spec.threshold:,.0f} corpus-wide",
                deals=[f"{m.deal[:8]} x{m.counts[spec.key]}" for m in offenders[:10]],
            ))
        elif spec.kind == "corpus_ratio":
            denom = site_atoms if spec.denominator == "site_atoms" else len(good)
            value = (totals[spec.key] / denom) if denom else 0.0
            offenders = sorted(
                (m for m in good if m.counts.get(spec.key)),
                key=lambda m: -m.counts[spec.key],
            )
            findings.append(Finding(
                metric=spec.key,
                value=value,
                threshold=spec.threshold,
                # No denominator means nothing was measured, not a clean pass.
                breached=(value > spec.threshold) if denom else False,
                display=f"{value:.1%} ({totals[spec.key]:,}/{denom:,})",
                detail=f"<= {spec.threshold:.0%} of site atoms",
                deals=[f"{m.deal[:8]} x{m.counts[spec.key]}" for m in offenders[:10]],
            ))
        else:  # per_deal_drift
            offenders, checked = drift_breaches(good, baseline or {})
            findings.append(Finding(
                metric=spec.key,
                value=len(offenders),
                threshold=0,
                breached=bool(offenders),
                display=f"{len(offenders)} of {checked} deals" if checked else "no baseline",
                detail=f"+/-{spec.threshold:.0%} per deal (floor {DRIFT_ABS_FLOOR} sites), both directions",
                deals=offenders[:10],
            ))
    return findings


# ── blob streaming ──────────────────────────────────────────────────

class BlobUnavailable(RuntimeError):
    """az is missing, unauthenticated, or the container is unreachable."""


def _az(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    if not shutil.which("az"):
        raise BlobUnavailable(
            "the Azure CLI ('az') is not on PATH -- cannot read envelopes. "
            "Install it and run 'az login'."
        )
    try:
        return subprocess.run(["az", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BlobUnavailable(f"az timed out after {timeout}s: az {' '.join(args[:3])}") from exc


def list_envelopes(max_mb: int = DEFAULT_MAX_MB) -> list[tuple[str, int]]:
    """(blob name, size) for every deal's latest envelope. Raises BlobUnavailable."""
    r = _az([
        "storage", "blob", "list",
        "--account-name", ACCOUNT, "-c", CONTAINER,
        "--prefix", BLOB_PREFIX, "--auth-mode", "login",
        "--num-results", "*",
        "--query", f"[?ends_with(name, '{ENVELOPE_SUFFIX}')].[name, properties.contentLength]",
        "-o", "tsv",
    ], timeout=600)
    if r.returncode != 0:
        raise BlobUnavailable(
            "could not list blobs (not logged in, or no read access to "
            f"{ACCOUNT}/{CONTAINER}). Run 'az login'.\n  az said: "
            f"{(r.stderr or '').strip().splitlines()[-1] if r.stderr.strip() else '(no stderr)'}"
        )
    rows: list[tuple[str, int]] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            size = int(parts[1])
        except ValueError:
            continue
        if size <= max_mb * 1024 * 1024:
            rows.append((parts[0], size))
    if not rows:
        raise BlobUnavailable(
            f"listed 0 envelopes under {CONTAINER}/{BLOB_PREFIX} -- refusing to "
            "report a clean base from an empty corpus."
        )
    return rows


def pick_sample(rows: list[tuple[str, int]], n: int) -> list[tuple[str, int]]:
    """Size-stratified, deterministic sample: same deals run to run.

    Even spacing over size order spans tiny deals and heavy ones alike, so
    the sample is comparable across days and the baseline stays meaningful.
    """
    rows = sorted(rows, key=lambda r: (r[1], r[0]))
    if n <= 0 or n >= len(rows):
        return rows
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def deal_of(blob_name: str) -> str:
    parts = blob_name.split("/")
    return parts[1] if len(parts) > 1 else blob_name


def stream_measures(sample: list[tuple[str, int]], verbose: bool = True) -> list[DealMeasure]:
    """Download -> measure -> discard, one envelope at a time.

    The corpus is ~1.6GB; nothing accumulates on disk and nothing accumulates
    in memory beyond one envelope and the per-deal counters.
    """
    out: list[DealMeasure] = []
    with tempfile.TemporaryDirectory(prefix="base_health_") as tmp:
        path = os.path.join(tmp, "envelope.json")
        for i, (name, _size) in enumerate(sample, 1):
            deal = deal_of(name)
            r = _az([
                "storage", "blob", "download",
                "--account-name", ACCOUNT, "-c", CONTAINER, "-n", name,
                "--auth-mode", "login", "-f", path, "--no-progress", "--overwrite",
            ])
            if r.returncode != 0:
                tail = (r.stderr or "").strip().splitlines()
                out.append(DealMeasure(deal=deal, error=f"download failed: {tail[-1] if tail else '?'}"))
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        envelope = json.load(fh)
                    m = measure_envelope(envelope, deal=deal)
                except json.JSONDecodeError as exc:
                    m = DealMeasure(deal=deal, error=f"unparseable JSON: {exc}")
                except EnvelopeError as exc:
                    m = DealMeasure(deal=deal, error=str(exc))
                except OSError as exc:
                    m = DealMeasure(deal=deal, error=f"unreadable download: {exc}")
                out.append(m)
            if os.path.exists(path):
                os.remove(path)
            if verbose:
                m = out[-1]
                status = f"ERROR {m.error}" if m.error else (
                    f"atoms={m.atoms:<6} sites={m.site_atoms:<5} "
                    + " ".join(f"{k.split('_')[0]}={v}" for k, v in m.counts.items() if v)
                )
                print(f"[{i:>3}/{len(sample)}] {deal[:8]}  {status}", flush=True)
    return out


# ── baseline ────────────────────────────────────────────────────────

def build_baseline(measures: list[DealMeasure]) -> dict[str, Any]:
    """Per-deal expected site/atom counts. Small and readable on purpose."""
    return {
        "_comment": (
            "Expected per-deal counts for tools/base_health.py site_count_drift. "
            "Regenerate with --update-baseline ONLY when a change is supposed to "
            "move these numbers, and say why in the commit message. "
            "See docs/BASE_HEALTH.md."
        ),
        "deals": {
            m.deal: {"site_atoms": m.site_atoms, "atoms": m.atoms}
            for m in sorted(measures, key=lambda x: x.deal) if m.ok
        },
    }


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


# ── reporting ───────────────────────────────────────────────────────

def render_table(findings: list[Finding], measures: list[DealMeasure],
                 baseline_deals: int) -> str:
    good = [m for m in measures if m.ok]
    errors = [m for m in measures if not m.ok]
    lines = [
        "",
        "=" * 78,
        f"BASE HEALTH — {len(good)} envelopes measured, {len(errors)} errors, "
        f"{sum(m.atoms for m in good):,} atoms, {sum(m.site_atoms for m in good):,} site atoms",
        "=" * 78,
        f"  {'metric':<36} {'value':>20}  {'threshold':<24}",
        f"  {'-' * 36} {'-' * 20}  {'-' * 24}",
    ]
    for f in findings:
        mark = "FAIL" if f.breached else "ok"
        lines.append(f"  {f.metric:<36} {f.display:>20}  {f.detail:<24} [{mark}]")
    lines.append("")
    for f in findings:
        if f.breached and f.deals:
            lines.append(f"  {f.metric} — offending deals:")
            lines.extend(f"     {d}" for d in f.deals)
            lines.append("")
    undecidable = sum(m.names_undecidable for m in good)
    if undecidable:
        lines.append(
            f"  NOT MEASURED: {undecidable:,} display names sit on atoms whose text is an"
        )
        lines.append(
            "     extractor-composed field summary, so the envelope carries no evidence"
        )
        lines.append(
            "     either way. Excluded from fabricated_names, not counted as clean."
        )
        lines.append("")
    if errors:
        lines.append(f"  ERRORS ({len(errors)}) — these are failures, not clean deals:")
        lines.extend(f"     {m.deal[:8]}  {m.error}" for m in errors[:15])
        if len(errors) > 15:
            lines.append(f"     ... and {len(errors) - 15} more")
        lines.append("")
    if not baseline_deals:
        lines.append("  NOTE: no baseline loaded — site_count_drift was not evaluated.")
        lines.append("")
    return "\n".join(lines)


def build_report(findings: list[Finding], measures: list[DealMeasure]) -> dict[str, Any]:
    good = [m for m in measures if m.ok]
    return {
        "deals_measured": len(good),
        "deals_errored": len(measures) - len(good),
        "atoms": sum(m.atoms for m in good),
        "site_atoms": sum(m.site_atoms for m in good),
        "names_undecidable": sum(m.names_undecidable for m in good),
        "metrics": [
            {
                "metric": f.metric,
                "value": f.value,
                "display": f.display,
                "threshold": f.threshold,
                "threshold_detail": f.detail,
                "breached": f.breached,
                "rationale": METRIC_BY_KEY[f.metric].rationale,
                "offenders": f.deals,
            }
            for f in findings
        ],
        "errors": [{"deal": m.deal, "error": m.error} for m in measures if not m.ok],
    }


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Measure corpus base health against committed thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit 0 = clean, 1 = threshold breached, 2 = could not measure",
    )
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                   help=f"deals to measure, size-stratified (default {DEFAULT_SAMPLE}; 0 = all)")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                   help="committed per-deal expected counts")
    p.add_argument("--update-baseline", action="store_true",
                   help="rewrite the baseline from this run and exit 0 (deliberate act)")
    p.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB,
                   help=f"skip envelopes larger than this (default {DEFAULT_MAX_MB})")
    p.add_argument("--json", action="store_true", help="machine-readable output on stdout")
    p.add_argument("--json-out", type=Path, default=None,
                   help="also write the machine-readable report here "
                        "(so CI gets both a table and a parseable artifact from ONE "
                        "pass over the corpus, not two)")
    p.add_argument("--quiet", action="store_true", help="no per-deal progress lines")
    args = p.parse_args(argv)

    try:
        rows = list_envelopes(max_mb=args.max_mb)
        sample = pick_sample(rows, args.sample)
        if not args.quiet and not args.json:
            print(f"measuring {len(sample)} of {len(rows)} envelopes "
                  f"(streamed one at a time)\n", flush=True)
        measures = stream_measures(sample, verbose=not (args.quiet or args.json))
    except BlobUnavailable as exc:
        # Degrade honestly: say what is wrong and exit nonzero. Never a pass.
        print(f"\nCANNOT MEASURE — base health was NOT verified.\n  {exc}\n",
              file=sys.stderr)
        if args.json:
            json.dump({"status": "cannot_measure", "reason": str(exc)}, sys.stdout, indent=2)
            print()
        return EXIT_CANNOT_MEASURE

    good = [m for m in measures if m.ok]
    if not good:
        print("\nCANNOT MEASURE — every envelope failed to measure. "
              "This is an infrastructure failure, not a clean base.\n", file=sys.stderr)
        for m in measures[:10]:
            print(f"  {m.deal[:8]}  {m.error}", file=sys.stderr)
        return EXIT_CANNOT_MEASURE

    if args.update_baseline:
        payload = build_baseline(measures)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote baseline for {len(payload['deals'])} deals -> {args.baseline}")
        if measures != good:
            print(f"  ({len(measures) - len(good)} errored deals were NOT baselined)")
        return EXIT_OK

    baseline = load_baseline(args.baseline)
    findings = evaluate(measures, baseline)
    n_baseline = len((baseline or {}).get("deals") or {})

    report = build_report(findings, measures)
    report["baseline_deals"] = n_baseline
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print(render_table(findings, measures, n_baseline))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    breached = [f.metric for f in findings if f.breached]
    errors = len(measures) - len(good)
    if breached or errors:
        if not args.json:
            if breached:
                print(f"FAIL — thresholds breached: {', '.join(breached)}")
            if errors:
                print(f"FAIL — {errors} envelope(s) could not be measured "
                      "(an unmeasurable envelope is a failure, not a pass)")
        return EXIT_BREACH
    if not args.json:
        print("PASS — every metric within threshold")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
