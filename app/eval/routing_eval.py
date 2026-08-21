"""Score the parser router instead of arguing about it.

Routing was a pile of heuristics with no number attached, which is why it
could be wrong for a long time without anyone noticing: RFPs, SOWs and specs
were being parsed as meeting transcripts, and the only way that surfaced was
somebody reading the routing table by hand.

This gives the decision a score. A routing case is a document SHAPE plus the
parser that shape should reach, so the corpus is portable -- the shapes are
reconstructed rather than referenced, and the eval runs on any machine
instead of only where the deal packs happen to be synced.

Two uses:

    from app.eval.routing_eval import evaluate_routing, DEFAULT_CASES
    report = evaluate_routing(DEFAULT_CASES)
    print(report.summary())

and, when real packs ARE present, ``cases_from_directory`` scores them against
an expectation map so the portable corpus and the real one report the same
metric.

What the number is for: it is a baseline to beat. The current router is
hand-written rules, and the honest end state is a learned router that can be
corrected by a PM the way every other decision in this system can. That
replacement is only safe if there is a score to compare against, and this is
it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class RoutingCase:
    """One document shape and the parser it should reach.

    ``build`` writes the document into a directory and returns its path, so a
    case carries its own fixture and nothing has to be checked in as a binary.
    """

    name: str
    expected_parser: str
    build: Callable[[Path], Path]
    #: Why this case exists -- usually the misroute it was written for.
    note: str = ""


@dataclass
class RoutingResult:
    case: RoutingCase
    actual_parser: str
    confidence: float
    reason: str

    @property
    def correct(self) -> bool:
        return self.actual_parser == self.case.expected_parser


@dataclass
class RoutingReport:
    results: list[RoutingResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def failures(self) -> list[RoutingResult]:
        return [r for r in self.results if not r.correct]

    def per_parser(self) -> dict[str, tuple[int, int]]:
        """expected_parser -> (correct, total)."""
        out: dict[str, list[int]] = {}
        for r in self.results:
            slot = out.setdefault(r.case.expected_parser, [0, 0])
            slot[1] += 1
            if r.correct:
                slot[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def summary(self) -> str:
        lines = [
            f"routing accuracy: {self.accuracy * 100:.1f}%  "
            f"({sum(1 for r in self.results if r.correct)}/{len(self.results)})",
            "",
        ]
        for parser, (ok, total) in sorted(self.per_parser().items()):
            lines.append(f"  {parser:<20} {ok}/{total}")
        if self.failures:
            lines += ["", "  misroutes:"]
            for r in self.failures:
                lines.append(
                    f"    {r.case.name:<34} expected {r.case.expected_parser:<18} "
                    f"got {r.actual_parser} ({r.confidence:.2f} {r.reason})"
                )
        return "\n".join(lines)


def _route(path: Path) -> tuple[str, float, str]:
    from app.parsers.registry import choose_parser

    parser, match, _all = choose_parser(path)
    name = type(parser).__name__ if parser is not None else "NONE"
    reason = (match.reasons or ["-"])[0]
    return name, match.confidence, reason


def evaluate_routing(cases: Iterable[RoutingCase], workdir: Path | None = None) -> RoutingReport:
    """Route every case and score it against its expected parser."""
    report = RoutingReport()
    root = workdir or Path(tempfile.mkdtemp(prefix="routing_eval_"))
    root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        target = case.build(root)
        name, confidence, reason = _route(target)
        report.results.append(RoutingResult(case, name, confidence, reason))
    return report


def cases_from_directory(
    directory: Path, expectations: dict[str, str]
) -> list[RoutingCase]:
    """Build cases from real files, keyed by filename -> expected parser.

    For scoring a real deal pack with the same metric the portable corpus
    uses. Files not named in ``expectations`` are skipped, because an
    unlabelled file has no right answer to be measured against.
    """
    out: list[RoutingCase] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        expected = expectations.get(path.name)
        if expected is None:
            continue
        out.append(
            RoutingCase(
                name=path.name,
                expected_parser=expected,
                build=(lambda _d, _p=path: _p),
                note="real pack",
            )
        )
    return out


# ── the portable corpus ──────────────────────────────────────────────────
# Every shape here is one that was observed being misrouted on a real deal.


def _w(rel: str, body: str) -> Callable[[Path], Path]:
    def build(root: Path) -> Path:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        return p

    return build


def _quote_xlsx(rel: str) -> Callable[[Path], Path]:
    def build(root: Path) -> Path:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["Part Number", "Description", "Qty", "Unit Price", "Extended"])
        ws.append(["CAM-IP-001", "IP Camera", 192, 338.71, 65032.32])
        ws.append(["BRK-88-B117", "Mounting bracket", 64, 12.50, 800.00])
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        wb.save(p)
        return p

    return build


DEFAULT_CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        "rfp_as_text", "MarkdownParser",
        _w("rfp_original.txt",
           "Request for Proposals\nStructured Cabling\nSchool District Contact:\n"
           "David Miller, david@example.com\nPrepared by: Consulting Group\n"
           + "The contractor shall install Cat6 UTP drops at each location.\n" * 30),
        "one 'Label:' line claimed this as a transcript at 0.82",
    ),
    RoutingCase(
        "sow_as_text", "MarkdownParser",
        _w("sow_courtroom.txt",
           "Statement of Work\nSection 1: Scope\n"
           + "Contractor shall terminate both ends and certify every drop.\n" * 25),
        "same misroute as the RFP",
    ),
    RoutingCase(
        "transcript_with_timestamps", "TranscriptParser",
        _w("kickoff.txt",
           "\n".join(f"[00:0{i % 10}:12] Speaker {i % 3}: forty sites by Q3." for i in range(30))),
        "timestamps are unambiguous transcript evidence",
    ),
    RoutingCase(
        "transcript_speaker_density", "TranscriptParser",
        _w("teams_export.txt",
           "\n".join(f"{'Cliff Creech' if i % 2 else 'Purtera PM'}: line {i}." for i in range(30))),
        "Teams export: speaker turns, no timestamps",
    ),
    RoutingCase(
        "email_with_headers", "EmailParser",
        _w("customer_email.txt",
           "From: jane@example.com\nSent: 2026-01-15 09:00\nSubject: Scope update\n\n"
           "Please remove West Wing from scope.\n"),
        "RFC-5322 headers are real content evidence",
    ),
    RoutingCase(
        "pm_note_named_like_a_quote", "MarkdownParser",
        _w("pm_note_pricing_schedule_not_scope.txt",
           "PM Note - Structured Cabling Services\n\n"
           "Appendix I is a pricing schedule/catalog, not a project scope schedule.\n"
           "The work order is the project scope source.\n"
           "Vendor quote should be checked against the work order quantities.\n"),
        "claimed by QuoteParser on its NAME; produced zero atoms",
    ),
    RoutingCase(
        "quote_with_neutral_name", "QuoteParser",
        _quote_xlsx("attachment_b.xlsx"),
        "content must claim a quote without filename help",
    ),
    RoutingCase(
        "table_dump_with_scope_phrase", "MarkdownParser",
        _w("roster_dump.txt",
           "\n".join(["Site | Qty | Notes"]
                     + [f"ATL-{i:02d} | {i} | escort required" for i in range(30)])),
        "EmailParser claimed this on one keyword hit",
    ),
    RoutingCase(
        "contentless_text", "NONE",
        _w("random.txt", "just filler words with no structured signals"),
        "the warning IS the coverage record; must stay unclaimed",
    ),
    RoutingCase(
        "prose_with_a_body", "MarkdownParser",
        _w("scope_notes.txt",
           "Scope of Work\n\n"
           "Contractor shall install forty cameras at the Atlanta facility.\n"
           "Escort is required at all times inside the yard.\n"
           "Mid-turn jumpers are excluded from this bill of materials.\n"),
        "a short real document must still be read",
    ),
)


if __name__ == "__main__":  # pragma: no cover - operator entry point
    print(evaluate_routing(DEFAULT_CASES).summary())
