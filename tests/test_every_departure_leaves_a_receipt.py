# -*- coding: utf-8 -*-
"""A stage that deletes an atom owes the ledger a receipt.

The suppression ledger is the instrument every content-loss audit in this repo
reads, `_tools/_phase3_audit.py` included. It is only a guarantee if every
departure is written to it -- and measured against the compile's actual atom
list rather than `output_count` (which for most stages reports what the stage
DID, not how many atoms came out), four phase-3 stages were deleting silently:

    atom_type_sanity           37 atoms on 010238, 293 on 010237, no capture at all
    stakeholder_dedup          17 atoms, recorded only as telemetry warnings
    open_question_resolution    7 atoms, filed under open_question_quality_filter
    pasted_note_dedup           3 atoms, see below

That is why thirteen of sixteen stages looked innocent on the first pass. One
of atom_type_sanity's 37 was the only atom stating live 010238's account number
and its contract effective and expiry dates.

`pasted_note_dedup` is the sharpest of the four: it HAD a `capture_suppressed`
call, and that call omitted the required keyword-only `reason`. Every
invocation raised TypeError into the stage's `except Exception` and was logged
as the stage failing, so the ledger call had never once run.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.core.suppression_ledger import capture_suppressed


def _compiler_tree() -> ast.Module:
    import app.core.compiler as C
    return ast.parse(Path(inspect.getfile(C)).read_text(encoding="utf-8"))


def test_reason_is_required_so_a_missing_one_cannot_pass_silently() -> None:
    with pytest.raises(TypeError):
        capture_suppressed([], [], stage="x")  # type: ignore[call-arg]


def test_every_capture_call_passes_a_reason() -> None:
    """A required keyword omitted at a call site inside a `try` is invisible:
    it raises, the handler logs "the stage failed", and the atoms leave anyway."""
    missing = [
        n.lineno
        for n in ast.walk(_compiler_tree())
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "capture_suppressed"
        and "reason" not in {k.arg for k in n.keywords}
    ]
    assert not missing, f"capture_suppressed without reason= at lines {missing}"


@pytest.mark.parametrize("stage", [
    "atom_type_sanity",
    "stakeholder_dedup",
    "pasted_note_dedup",
])
def test_the_leaky_stages_now_file(stage: str) -> None:
    """Each of these deleted atoms with no ledger entry. Pin the call site so
    a later edit cannot quietly remove it again."""
    src = Path(inspect.getfile(__import__("app.core.compiler", fromlist=["x"]))).read_text(
        encoding="utf-8")
    assert f'stage="{stage}"' in src, f"{stage} no longer files a suppression receipt"
