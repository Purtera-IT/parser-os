"""What kind of table is this? One question, asked in one place.

Today it is asked three times, in three modules, with three signatures and
three return types:

    looks_like_site_roster(columns, rows, surrounding_text) -> bool
        app/parsers/site_roster_extractor.py, 15 call sites, PDF tables

    identify_schema(columns) -> str | None
        app/core/table_schema_registry.py, 5 call sites, column patterns

    classify_sheet(sheet_name, rows) -> SheetClassification
        app/parsers/sheet_classifier.py, 9 call sites, spreadsheet sheets

Three of those are the SAME judgment -- "is this a site roster?" -- reached by
different evidence depending on which format the table arrived in. A PM who
corrects a misread roster in a PDF teaches the spreadsheet path nothing, because
there is no shared thing to correct.

This module is the front door. It does not reimplement the heuristics: it
delegates to them, so behaviour is unchanged today, and it gives the judgment
the three properties a readout needs and a scattered heuristic cannot have --
one entry point, a confidence that permits abstention, and a recorded reason a
correction can attach to.

When the shared encoder exists, the head attaches HERE, behind the same
signature, and every call site inherits it at once. That is the whole reason to
put a seam in before the model rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: Kinds this judgment can return. Deliberately coarse: the fine-grained
#: distinctions (which rate card, whose BOM) are facets, decided downstream on
#: an atom, not on the table as a whole.
SITE_ROSTER = "site_roster"
RATE_CARD = "rate_card"
BOM = "bom"
SCHEDULE = "schedule"
REFERENCE = "reference"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class TableJudgment:
    """One answer, with enough context to disagree with it usefully."""

    kind: str = UNKNOWN
    confidence: float = 0.0
    #: Which implementation produced this, so a correction knows what it is
    #: correcting and a later head can be compared against it.
    decided_by: str = ""
    reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        """No opinion. Distinct from deciding UNKNOWN with confidence.

        A component that cannot say "I don't know" cannot be safely promoted,
        so abstention is a first-class outcome rather than a default branch.
        """
        return self.kind == UNKNOWN and self.confidence <= 0.0

    def as_correction_target(self) -> dict[str, Any]:
        """The payload a PM correction attaches to.

        Shaped for a TrainingRow: relation identifies the judgment, label is
        what was decided, and provenance records which implementation decided
        it. One shape regardless of whether the table came from a PDF or a
        spreadsheet -- which is the point.
        """
        return {
            "relation": "table_kind",
            "label": self.kind,
            "confidence": self.confidence,
            "provenance": {"decided_by": self.decided_by, "reason": self.reason},
        }


def judge_table(
    *,
    columns: Sequence[str] | None = None,
    rows: Sequence[Any] | None = None,
    sheet_name: str | None = None,
    surrounding_text: str = "",
) -> TableJudgment:
    """Classify a table, whatever format it arrived in.

    Evidence differs by source and that is fine -- a spreadsheet has a sheet
    name, a PDF table has surrounding prose -- but the QUESTION and the answer
    shape do not, which is what makes the answer correctable in one place.

    Order is by precision, not by convenience:

      1. A named sheet is the strongest signal available; the spreadsheet
         classifier already returns a role, a confidence and a reason.
      2. The roster gate reads prose declarations and row shape, which the
         column-pattern registry cannot see.
      3. The schema registry generalises past rosters to rate cards and BOMs.

    Never raises: an interpretation failing is an abstention, not an error, and
    a compile with twenty other artifacts must not die because one table was
    ambiguous.
    """
    cols = [str(c or "") for c in (columns or [])]
    rws = list(rows or [])

    # 1 -------------------------------------------------- spreadsheet sheets
    if sheet_name is not None:
        try:
            from app.parsers.sheet_classifier import classify_sheet

            sc = classify_sheet(sheet_name, rws)
            role = getattr(getattr(sc, "role", None), "value", None) or str(
                getattr(sc, "role", "")
            )
            return TableJudgment(
                kind=_kind_from_sheet_role(role),
                confidence=float(getattr(sc, "confidence", 0.0) or 0.0),
                decided_by="sheet_classifier.classify_sheet",
                reason=str(getattr(sc, "reason", "") or ""),
                signals=dict(getattr(sc, "signals", {}) or {}),
            )
        except Exception:
            pass

    # 2 ------------------------------------------------------- roster gate
    try:
        from app.parsers.site_roster_extractor import looks_like_site_roster

        if looks_like_site_roster(
            columns=cols, rows=rws, surrounding_text=surrounding_text
        ):
            return TableJudgment(
                kind=SITE_ROSTER,
                # The gate is a boolean, so there is no calibrated probability
                # to report. Stated plainly rather than invented: this is
                # exactly the hand-tuned threshold a conformal readout replaces.
                confidence=1.0,
                decided_by="site_roster_extractor.looks_like_site_roster",
                reason="roster gate matched",
                signals={"columns": cols[:8]},
            )
    except Exception:
        pass

    # 3 ---------------------------------------------------- schema registry
    try:
        from app.core.table_schema_registry import identify_schema

        schema = identify_schema(cols)
        if schema:
            return TableJudgment(
                kind=_kind_from_schema(schema),
                confidence=1.0,
                decided_by="table_schema_registry.identify_schema",
                reason=f"matched schema {schema}",
                signals={"schema": schema, "columns": cols[:8]},
            )
    except Exception:
        pass

    return TableJudgment(kind=UNKNOWN, confidence=0.0, decided_by="",
                         reason="no signal", signals={"columns": cols[:8]})


def _kind_from_sheet_role(role: str) -> str:
    r = (role or "").strip().lower()
    if "roster" in r or "site" in r:
        return SITE_ROSTER
    if "rate" in r:
        return RATE_CARD
    if "bom" in r or "material" in r:
        return BOM
    if "schedule" in r:
        return SCHEDULE
    if "reference" in r or "empty" in r:
        return REFERENCE
    return UNKNOWN


def _kind_from_schema(schema: str) -> str:
    s = (schema or "").strip().lower()
    if "roster" in s or "site" in s:
        return SITE_ROSTER
    if "rate" in s:
        return RATE_CARD
    if "bom" in s or "material" in s:
        return BOM
    if "schedule" in s:
        return SCHEDULE
    return UNKNOWN
