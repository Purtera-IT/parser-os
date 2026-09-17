"""Operator control over which compile stages run on a re-parse (PUR-58).

A re-parse normally runs every stage of :func:`app.core.compiler.compile_project`.
An operator iterating on one stage (say ``work_order``) does not want to pay for
every other LLM-backed enrichment on every loop, so a *stage plan* lets them name
the optional stages to run.

What this module does and does NOT do -- be precise about it:

* Only the stages in :data:`SELECTABLE_STAGES` can be switched off. They are
  enrichment/judgement passes that the rest of the pipeline tolerates being
  absent (each is already wrapped so that a failure degrades to "no change").
  Everything else (discovery, parsing, adjudication, classification, graph,
  packetizing, quality gates) is *core* and always runs: skipping those would
  produce a structurally broken compile, so they are not offered.
* "Reuse stored output" is real only for parsing: ``parse_artifacts`` always
  runs but honours the per-artifact parse cache (``use_cache``), so unchanged
  artifacts reuse their stored parse output. The plan record reports whether
  the cache was enabled; the manifest's ``cache_hits``/``reused_artifact_ids``
  say which artifacts actually reused it.
* A selectable stage that is not selected is simply **not run** -- its output
  from a previous compile is NOT replayed. The result therefore lacks that
  stage's contribution, which is why a partial result is labelled everywhere
  (``CompileResult.stage_plan``, an ``INFO: PARTIAL PARSE`` warning, and the
  trace) so it cannot be mistaken for a full parse or a corpus measurement.
* Combinations that cannot work (a selected stage whose required input stage is
  neither selected nor reusable) are refused with a reason via
  :class:`StagePlanError` rather than compiled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

PARTIAL_WARNING_PREFIX = "INFO: PARTIAL PARSE"


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    # Stages whose output this stage consumes. None of the selectable stages'
    # outputs are stored between compiles, so a required stage must be selected.
    needs: tuple[str, ...] = ()


SELECTABLE_STAGES: dict[str, StageSpec] = {
    s.name: s
    for s in (
        StageSpec("email_threading", "link .eml artifacts into conversation threads"),
        StageSpec(
            "quoted_history_dedup",
            "drop quoted email echoes already present in the thread",
            needs=("email_threading",),
        ),
        StageSpec("pdf_image_vision", "describe raster images in PDFs (also env-gated)"),
        StageSpec("document_job_scope", "set aside documents about another job"),
        StageSpec("work_order", "reassemble relevant documents into work lines (also env-gated)"),
        StageSpec("task_hours", "stamp learned hours on task atoms"),
        StageSpec("commercial_terms", "stamp learned commercial kit shape on task atoms"),
    )
}

# Output of these can be reused from storage instead of recomputed.
REUSABLE_STAGES: dict[str, str] = {
    "parse_artifacts": "per-artifact parse cache (use_cache)",
}


class StagePlanError(ValueError):
    """A requested stage plan is invalid or cannot produce a sound compile."""


@dataclass
class StagePlan:
    requested: list[str] | None  # None == full parse
    use_cache: bool = True
    _ran: list[str] = field(default_factory=list)

    @classmethod
    def full(cls, use_cache: bool = True) -> "StagePlan":
        return cls(requested=None, use_cache=use_cache)

    @classmethod
    def from_request(
        cls, stages: Iterable[str] | str | None, *, use_cache: bool = True
    ) -> "StagePlan":
        """Validate an operator request. ``None`` means a full parse.

        Accepts a list of names or a comma-separated string. Raises
        :class:`StagePlanError` on unknown names or impossible combinations.
        """
        if stages is None:
            return cls.full(use_cache=use_cache)
        if isinstance(stages, str):
            stages = stages.split(",")
        names: list[str] = []
        for raw in stages:
            name = str(raw).strip()
            if name and name not in names:
                names.append(name)
        valid = ", ".join(sorted(SELECTABLE_STAGES))
        unknown = [n for n in names if n not in SELECTABLE_STAGES]
        if unknown:
            raise StagePlanError(
                f"unknown stage(s): {', '.join(unknown)}. Selectable stages: {valid}"
            )
        problems: list[str] = []
        for n in names:
            for dep in SELECTABLE_STAGES[n].needs:
                if dep in names:
                    continue
                if dep in REUSABLE_STAGES and use_cache:
                    continue
                problems.append(
                    f"'{n}' consumes the output of '{dep}', which is neither "
                    f"selected nor reusable from stored output; add '{dep}'"
                )
        if problems:
            raise StagePlanError("refused stage plan: " + "; ".join(problems))
        ordered = [s for s in SELECTABLE_STAGES if s in names]
        return cls(requested=ordered, use_cache=use_cache)

    @property
    def partial(self) -> bool:
        return self.requested is not None and set(self.requested) != set(SELECTABLE_STAGES)

    def runs(self, name: str) -> bool:
        """Should ``name`` run? Core (non-selectable) stages always run."""
        if self.requested is None or name not in SELECTABLE_STAGES:
            return True
        return name in self.requested

    def mark_ran(self, name: str) -> None:
        if name not in self._ran:
            self._ran.append(name)

    @property
    def skipped(self) -> list[str]:
        return [s for s in SELECTABLE_STAGES if not self.runs(s)]

    def record(self) -> dict[str, Any]:
        return {
            "partial": self.partial,
            "stages_requested": None if self.requested is None else list(self.requested),
            "stages_ran": [s for s in SELECTABLE_STAGES if self.runs(s)],
            "stages_skipped": self.skipped,
            "stages_reused": (
                {"parse_artifacts": REUSABLE_STAGES["parse_artifacts"]} if self.use_cache else {}
            ),
            "note": (
                "skipped stages were NOT run and their prior output was NOT replayed; "
                "not comparable to a full parse or a corpus result"
                if self.partial
                else "full parse"
            ),
        }

    def warning(self) -> str | None:
        if not self.partial:
            return None
        return (
            f"{PARTIAL_WARNING_PREFIX}: ran only [{', '.join(self.requested or []) or 'core stages'}]; "
            f"skipped [{', '.join(self.skipped)}] -- not a full parse, do not report as a corpus result"
        )


def selectable_stage_names() -> list[str]:
    return list(SELECTABLE_STAGES)


def plan_for_reparse(
    stages: Iterable[str] | str | None = None, *, use_cache: bool = True
) -> StagePlan:
    """Entry point for the one-deal loop: validate a stage selection.

    Returns a :class:`StagePlan`; pass ``plan.requested`` as ``stages=`` to
    ``compile_project`` (or the plan itself). Raises :class:`StagePlanError`
    with the reason if the selection cannot work.
    """
    return StagePlan.from_request(stages, use_cache=use_cache)


def is_partial_result(result: Any) -> bool:
    """True if a CompileResult (or its dict form) came from a partial parse."""
    sp = result.get("stage_plan") if isinstance(result, dict) else getattr(result, "stage_plan", None)
    return bool(sp and sp.get("partial"))
