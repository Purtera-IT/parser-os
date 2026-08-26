"""Every load-bearing magic number, with its receipt.

A threshold is a claim about the world: "50% speaker density separates
transcripts from business documents" is only true of the corpus it was
measured on, on the day it was measured. The number survives in the code;
the claim quietly expires as the corpus grows -- and nothing says so.

This registry makes each constant carry what an atom carries: provenance.
Where it was derived, from how many examples, when, and what would invalidate
it. ``tests/test_calibration_registry.py`` pins every entry to its live usage
site, so a constant cannot drift from its documentation -- change one without
the other and a test names the discrepancy.

``stale_when`` is the honest part. It states the condition under which the
number should be re-derived rather than trusted, so revisiting them is a
checklist instead of an archaeology dig. Entries whose derivation corpus was
tiny say so in plain numbers.

This is deliberately NOT config. None of these are tunables an operator
should touch; they are measurements, and the registry is their lab notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Calibrated:
    """One measured constant and the evidence behind it."""

    value: float
    #: module.symbol (or module (source) for function-locals) this pins.
    used_at: str
    #: What was measured, on what corpus, with the discriminating numbers.
    derivation: str
    #: Size of the corpus the derivation rests on. Small is fine; hidden is not.
    corpus_n: int
    derived: str  # YYYY-MM
    #: The condition that expires this number.
    stale_when: str
    notes: str = ""
    #: Optional: recompute this constant's VALID BOUNDS from the eval corpus.
    #: Returns {"lower": x, "upper": y} -- the range the current corpus
    #: supports. drift_report() checks value against it; when the corpus
    #: grows and the bounds move past the shipped value, the report names it
    #: instead of the number quietly expiring. This is the third category
    #: between config and weights: a slow parameter that re-derives itself.
    re_derive: Callable[[], dict] | None = field(default=None, compare=False)


REGISTRY: dict[str, Calibrated] = {
    "transcript.speaker_density": Calibrated(
        value=0.50,
        re_derive=lambda: _derive_speaker_density_bounds(),
        used_at="app.parsers.transcript_parser (source: turns / len(scan) >= 0.50)",
        derivation=(
            "Measured across 19 real .txt files: business documents score "
            "1.3-15.2% (RFP 3.2-7.6, SOW 1.7, specs 1.3, addendum 10.2, Q&A "
            "15.2); the highest non-transcripts are customer emails at 37-43%, "
            "and those are claimed by EmailParser on their own headers first. "
            "0.50 sits above every measured non-transcript."
        ),
        corpus_n=19,
        derived="2026-08",
        stale_when=(
            "the eval corpus holds 50+ labelled .txt files, or any real "
            "transcript is observed scoring below 0.50 after the own-line "
            "speaker fold"
        ),
    ),
    "prose_floor.min_lines": Calibrated(
        value=3,
        used_at="app.parsers.markdown_parser._MIN_TEXT_LINES",
        derivation=(
            "LINES is the load-bearing signal: every real deal document "
            "measured is at least 4 non-empty lines, and both observed "
            "contentless files are exactly one."
        ),
        corpus_n=19,
        derived="2026-08",
        stale_when="a real one- or two-line deal document is observed",
    ),
    "prose_floor.min_chars": Calibrated(
        value=100,
        re_derive=lambda: _derive_prose_floor_chars(),
        used_at="app.parsers.markdown_parser._MIN_TEXT_CHARS",
        derivation=(
            "Was 200 -- the smallest file in the corpus -- until a genuine "
            "185-char four-line scope note fell under it in a regression "
            "test. Lowered to a guard against three lines of nothing rather "
            "than a claim about document sizes."
        ),
        corpus_n=19,
        derived="2026-08",
        stale_when="any real document under 100 chars carries scope content",
    ),
    "pdf.low_text_page": Calibrated(
        value=80,
        used_at="app.parsers.orbitbrief_pdf (source: LOW_TEXT_PAGE_THRESHOLD = 80)",
        derivation=(
            "Of 80 sub-threshold pages across 35 real PDFs / 478 pages, 76 "
            "carry raster images (genuine OCR material); the 4 that do not "
            "hold 'Page Intentionally Left Blank' and letterhead footers -- "
            "252 chars whose loss costs nothing."
        ),
        corpus_n=478,
        derived="2026-08",
        stale_when="a text-only page under 80 chars is found carrying scope",
    ),
    "pdf.text_rich_page": Calibrated(
        value=1200,
        used_at="app.parsers.orbitbrief_pdf (source: TEXT_RICH_PAGE_THRESHOLD = 1200)",
        derivation=(
            "84 of 478 real pages sit within +/-25% of this cut; 62 of the 84 "
            "route identically either way because the mid-band shape checks "
            "(prose/list/form) absorb it. The threshold is a perf guard, not "
            "a correctness boundary, and was measured to be one."
        ),
        corpus_n=478,
        derived="2026-08",
        stale_when=(
            "a layout model lands in front of this router, or the 22 "
            "genuinely path-changing near-cut pages are shown to parse worse "
            "on their assigned side"
        ),
    ),
    # schematic.min_text_lines (25) / schematic.min_text_chars (400) were
    # retired 2026-08-26 with app.core.schematic_route: pure construction
    # schematics are no longer accepted into Purpulse, and the gate had no
    # production caller (PDF page routing uses pdf.low_text_page /
    # pdf.text_rich_page, which remain deployed and receipted above).
    "router.match_threshold": Calibrated(
        value=0.50,
        used_at="app.parsers.registry.MATCH_THRESHOLD",
        derivation=(
            "The line between a claim and a prior. Every filename hint is "
            "deliberately scored below it (0.45) and every content signal "
            "above it (0.55+), so the constant IS the evidence policy: a "
            "name may raise a claim and may never create one."
        ),
        corpus_n=2500,
        derived="2026-08",
        stale_when=(
            "any signal is added scoring within 0.05 of it -- the gap "
            "between prior (0.45) and weakest claim (0.55) is the safety "
            "margin, not the number itself"
        ),
    ),
    "speaker_fold.min_folds": Calibrated(
        value=4,
        used_at="app.core.normalizers.fold_standalone_speaker_lines (source: len(folds) >= 4)",
        derivation=(
            "Four own-line speaker turns before the Otter fold may rewrite a "
            "document. Attendee rosters fold to zero (a name under a name is "
            "not a turn), spec headings appear once each; four recurrences "
            "is conversation, not coincidence."
        ),
        corpus_n=19,
        derived="2026-08",
        stale_when="a real transcript with fewer than 4 turns is observed",
    ),
}


# ── derivations: bounds recomputed from the in-repo eval corpus ──────────


def _derive_prose_floor_chars() -> dict:
    """The floor must admit every real prose document and reject the
    contentless ones -- both live in the routing eval corpus, so the valid
    range is computable, not remembered."""
    import tempfile
    from pathlib import Path

    from app.eval.routing_eval import DEFAULT_CASES

    root = Path(tempfile.mkdtemp(prefix="calib_prose_"))
    real_docs: list[int] = []
    contentless: list[int] = []
    for case in DEFAULT_CASES:
        target = case.build(root)
        if target.suffix.lower() not in (".txt", ".md"):
            continue
        size = len(target.read_text(encoding="utf-8", errors="replace").strip())
        if case.expected_parser == "MarkdownParser":
            real_docs.append(size)
        elif case.expected_parser == "NONE":
            contentless.append(size)
    return {
        "lower": (max(contentless) + 1) if contentless else 1,
        "upper": min(real_docs) if real_docs else 10_000,
        "evidence": f"{len(real_docs)} real prose docs, {len(contentless)} contentless",
    }


def _derive_speaker_density_bounds() -> dict:
    """The density cut must sit above every business document and below every
    transcript in the corpus. Measured with the SAME detector the router uses,
    so the bound and the code cannot diverge in method."""
    import tempfile
    from pathlib import Path

    from app.core.normalizers import detect_speaker
    from app.eval.routing_eval import DEFAULT_CASES

    def density(path: Path) -> float:
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        if not lines:
            return 0.0
        return sum(1 for ln in lines if detect_speaker(ln) is not None) / len(lines)

    root = Path(tempfile.mkdtemp(prefix="calib_density_"))
    business: list[float] = []
    transcripts: list[float] = []
    for case in DEFAULT_CASES:
        target = case.build(root)
        if target.suffix.lower() != ".txt":
            continue
        if case.expected_parser == "MarkdownParser":
            business.append(density(target))
        elif case.expected_parser == "TranscriptParser" and "otter" not in case.name:
            # The own-line (Otter) dialect scores 0 colon-density BY DESIGN --
            # the speaker fold handles it -- so it is excluded from a bound
            # about the colon dialect. The first version of this filter
            # checked the wrong substring ("own"), let the Otter case in, and
            # drift_report() flagged impossible bounds [0.037, 0.0] on its
            # very first run: the mechanism caught a bug in its own author's
            # derivation before anything else got to rely on it.
            transcripts.append(density(target))
    return {
        "lower": max(business) if business else 0.0,
        "upper": min(transcripts) if transcripts else 1.0,
        "evidence": f"{len(business)} business docs, {len(transcripts)} colon-dialect transcripts",
    }


def drift_report() -> list[str]:
    """Run every re_derive and name the constants whose corpus outgrew them.

    Empty list == every derivable constant still sits inside the bounds its
    corpus supports. Anything else is a named drift, not a silent expiry.
    """
    problems: list[str] = []
    for key, entry in REGISTRY.items():
        if entry.re_derive is None:
            continue
        try:
            bounds = entry.re_derive()
        except Exception as exc:  # noqa: BLE001 - a broken derivation is itself a finding
            problems.append(f"{key}: derivation failed ({type(exc).__name__}: {exc})")
            continue
        low, high = bounds.get("lower", float("-inf")), bounds.get("upper", float("inf"))
        if not (low <= entry.value <= high):
            problems.append(
                f"{key}: shipped {entry.value} but the corpus now supports "
                f"[{low}, {high}] ({bounds.get('evidence', '')}) -- re-derive it"
            )
    return problems


def report(stale_only: bool = False) -> str:  # pragma: no cover - operator surface
    """Human listing; ``python -m app.core.calibration`` prints it."""
    del stale_only  # staleness is a judgment call the entry states, not computes
    lines = ["calibration registry -- every magic number and its receipt", ""]
    for key, entry in sorted(REGISTRY.items()):
        lines.append(f"  {key} = {entry.value}   ({entry.derived}, n={entry.corpus_n})")
        lines.append(f"    at    : {entry.used_at}")
        lines.append(f"    stale : {entry.stale_when}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
