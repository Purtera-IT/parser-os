"""Span-keyed provenance ledger: make silent recall loss visible and
self-classifying (parser issue vs decide issue).

Motivation
----------
The learned feedback store improves *routing* (classification / keep-drop /
merge) because every ``decide()`` call leaves a candidate behind that a human
correction can attach to. Recall misses leave nothing: a paragraph dropped at
parse time produces no atom, no anchor, no signal — the parser can never flag
what it doesn't know it threw away.

This ledger fixes the asymmetry with one primitive: **every raw source unit
gets a stable ``span_id`` at ingest, carried forward through every stage, and
every stage records what it did to each span.** Attribution then collapses to
"what is the last stage a span appears in?"

Each stage is tagged one of two *kinds*:

* ``GATE`` — a deterministic code rule (a prose keep/drop test, a table walk,
  an OCR/extract step). If content dies at a GATE it is a **parser issue**; the
  fix is a localized code change.
* ``SEAM`` — a ``decide()`` call (store -> LLM -> undecided). If content dies at
  a SEAM it is a **decide issue**; the fix is a learned correction or threshold.

So "parser bug or decide bug?" is answered mechanically by the *kind* of stage
that last consumed the span — no guessing. Each drop also records a one-line
``reason`` that doubles as the diagnosis and points at the fix.

This module is a passive side-channel: it never mutates parser output. A parser
or stage records into a ledger only when one is attached, so production parsing
is byte-for-byte unchanged when no ledger is present.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageKind(str, Enum):
    """Whether a stage's decision is deterministic code or a learned seam."""

    GATE = "gate"  # deterministic code rule -> parser issue -> code fix
    SEAM = "seam"  # decide() call -> decide issue -> learned/threshold fix


@dataclass(frozen=True)
class DropRecord:
    """One span killed (or suppressed) at one stage, with the reason why."""

    span_id: str
    stage: str
    kind: StageKind
    rule: str
    reason: str
    raw_text: str
    artifact: str = ""

    @property
    def fix_shape(self) -> str:
        return "code fix (localized)" if self.kind is StageKind.GATE else "learned correction / threshold"


@dataclass
class SpanLedger:
    """Records the lineage of every registered raw span.

    A span is *represented* once any stage emits a downstream artifact (atom)
    for it. A span with no representation and at least one drop record is
    *lost*. Spans that were represented but later suppressed at a SEAM are not
    "lost" — they are decide-dropped, a distinct, learnable bucket.
    """

    spans: dict[str, str] = field(default_factory=dict)  # span_id -> raw_text
    represented: set[str] = field(default_factory=set)
    drops: list[DropRecord] = field(default_factory=list)
    # Coverage canary: flag an artifact if represented/total falls below this.
    coverage_threshold: float = 0.95

    # -- recording -------------------------------------------------------
    def register_span(self, span_id: str, raw_text: str) -> None:
        """Register a raw source unit. Idempotent; keeps first non-empty text."""
        if span_id not in self.spans:
            self.spans[span_id] = raw_text

    def mark_represented(self, span_id: str) -> None:
        self.represented.add(span_id)

    def record_drop(
        self,
        *,
        span_id: str,
        stage: str,
        kind: StageKind,
        rule: str,
        reason: str,
        raw_text: str = "",
        artifact: str = "",
    ) -> None:
        if span_id and span_id not in self.spans and raw_text:
            self.spans[span_id] = raw_text
        self.drops.append(
            DropRecord(
                span_id=span_id,
                stage=stage,
                kind=kind,
                rule=rule,
                reason=reason,
                raw_text=raw_text or self.spans.get(span_id, ""),
                artifact=artifact,
            )
        )

    # -- analysis --------------------------------------------------------
    def coverage(self) -> tuple[int, int, float]:
        """(represented_span_count, total_registered_spans, ratio)."""
        total = len(self.spans)
        rep = len(self.represented & set(self.spans))
        ratio = (rep / total) if total else 1.0
        return rep, total, ratio

    def canary_ok(self) -> bool:
        _, total, ratio = self.coverage()
        return total == 0 or ratio >= self.coverage_threshold

    def lost_records(self) -> list[DropRecord]:
        """Drops whose span never produced any downstream representation."""
        return [d for d in self.drops if d.span_id not in self.represented]

    def gate_losses(self) -> list[DropRecord]:
        return [d for d in self.lost_records() if d.kind is StageKind.GATE]

    def seam_losses(self) -> list[DropRecord]:
        return [d for d in self.lost_records() if d.kind is StageKind.SEAM]

    # -- ingest decide()-side suppression from a compiled envelope -------
    def ingest_suppressed_atoms(self, suppressed: list[dict[str, Any]]) -> int:
        """Fold an envelope's ``suppressed_atoms`` in as SEAM drop records.

        These atoms WERE emitted, then suppressed downstream at a decide()
        seam, so they represent decide-issue losses rather than parser losses.
        The reason is read from decision_provenance / review_flags.
        """
        n = 0
        for a in suppressed:
            raw = (a.get("raw_text") or a.get("normalized_text") or "").strip()
            if not raw:
                continue
            prov = a.get("decision_provenance") or {}
            stage = (
                prov.get("stage")
                or prov.get("suppression_stage")
                or a.get("suppression_stage")
                or "suppression_ledger"
            )
            reason = (
                prov.get("reason")
                or prov.get("rationale")
                or a.get("suppression_reason")
                or ", ".join(a.get("review_flags") or [])
                or "suppressed at decide() seam"
            )
            source = prov.get("source") or prov.get("route") or "decide"
            span_id = f"{a.get('artifact_id','')}:{a.get('id') or a.get('atom_id','')}"
            self.record_drop(
                span_id=span_id,
                stage=str(stage),
                kind=StageKind.SEAM,
                rule=str(source),
                reason=str(reason)[:160],
                raw_text=raw,
                artifact=str(a.get("artifact_id", "")),
            )
            n += 1
        return n

    # -- reporting -------------------------------------------------------
    def report(self, *, sample: int = 6, width: int = 95) -> str:
        rep, total, ratio = self.coverage()
        lines: list[str] = []
        lines.append("=" * width)
        lines.append("LOST-CONTENT REPORT  (span-keyed provenance ledger)")
        lines.append("=" * width)
        verdict = "PASS" if self.canary_ok() else "*** FAIL ***"
        lines.append(
            f"coverage canary: {rep}/{total} spans represented "
            f"({ratio:.1%})  threshold {self.coverage_threshold:.0%}  ->  {verdict}"
        )
        lines.append("")

        for kind, header, fix in (
            (StageKind.GATE, "PARSER ISSUES  (died at a deterministic GATE)", "code fix"),
            (StageKind.SEAM, "DECIDE ISSUES  (suppressed at a learned SEAM)", "store correction"),
        ):
            losses = [d for d in self.lost_records() if d.kind is kind]
            lines.append("-" * width)
            lines.append(f"{header}: {len(losses)}   [{fix}]")
            lines.append("-" * width)
            if not losses:
                lines.append("  (none)")
                lines.append("")
                continue
            grouped: dict[tuple[str, str], list[DropRecord]] = defaultdict(list)
            for d in losses:
                grouped[(d.rule, d.reason)].append(d)
            for (rule, reason), recs in sorted(
                grouped.items(), key=lambda kv: len(kv[1]), reverse=True
            ):
                lines.append(f"  [{rule}] {reason}  (x{len(recs)})")
                for d in recs[:sample]:
                    lines.append(f"      - {d.raw_text[:width - 10]}")
                if len(recs) > sample:
                    lines.append(f"      ... +{len(recs) - sample} more")
            lines.append("")
        return "\n".join(lines)
