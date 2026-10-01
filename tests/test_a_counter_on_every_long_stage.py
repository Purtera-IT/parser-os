"""Every stage that can go long has to say how far through it is.

`typed_atom_classification` was the only stage with an interior counter, and
that choice was never measured -- it was simply the stage somebody looked at
first. Measured across nine live compiles (2,579 progress readings), the real
distribution of wall clock is:

    open_question_resolution     204.1s median   278.1s max   36.8%
    typed_atom_classification    131.9s          323.4s       23.8%
    document_job_scope            71.0s          192.8s       12.8%
    enrich_entities               60.5s           76.5s       10.9%
    bom_owner                     33.7s          107.4s        6.1%
    pdf_image_vision              15.7s          136.4s        2.8%
    source_replay                 10.9s          111.7s        2.0%
    site_geo_fallback             10.0s           13.0s        1.8%
    ...37 further stages under 0.5s median and 2s max

So the one instrumented stage was the SECOND largest, and a compile sitting in
`open_question_resolution` -- the largest, and more than a third of the
pipeline -- showed no estimate at all. These seven counters cover 95.3% of the
median compile.

These tests drive the real functions. A counter is easy to add and easy to add
WRONG: ticked after an early `continue` it stalls on some deals and not others,
given a denominator the loop never approaches it reads 4% and finishes, and
placed in a module that cannot import telemetry it raises on the first atom of
every compile. Each of those is a bug that only shows up live, so each one is
asserted here against the function as it actually runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core import telemetry


class _Atom:
    """The duck the stages type against: atom_type, raw_text, value, keys."""

    def __init__(self, atom_type: str, raw_text: str = "x", **kw: Any):
        self.atom_type = atom_type
        self.raw_text = raw_text
        self.value: Any = kw.pop("value", {})
        self.entity_keys = kw.pop("entity_keys", [])
        self.review_flags: list[str] = []
        self.review_status = None
        self.receipts: list[Any] = []
        for k, v in kw.items():
            setattr(self, k, v)


class _Recorder:
    """Captures every (done, total) the stage reports, in order."""

    def __init__(self, monkeypatch):
        self.seen: list[tuple[int, int]] = []
        real = telemetry.set_stage_progress

        def spy(done: int, total: int) -> None:
            real(done, total)
            self.seen.append(telemetry.stage_progress())

        monkeypatch.setattr(telemetry, "set_stage_progress", spy)

    # -- what every counter has to be true about ---------------------------

    @property
    def total(self) -> int:
        assert self.seen, "the stage never reported progress at all"
        totals = {t for _, t in self.seen}
        assert len(totals) == 1, f"the denominator moved mid-stage: {totals}"
        return totals.pop()

    def assert_counts_up_to(self, expected_total: int) -> None:
        assert self.total == expected_total, (
            f"denominator is {self.total}, but the loop only does "
            f"{expected_total} units of work -- a bar on a total the loop "
            f"never approaches reads a few percent and then finishes"
        )
        dones = [d for d, _ in self.seen]
        assert dones == sorted(dones), f"progress went backwards: {dones}"
        assert dones[0] == 0, f"first reading should be 0, got {dones[0]}"
        assert dones[-1] == expected_total, (
            f"last reading is {dones[-1]} of {expected_total}: the counter "
            f"stops short, so every estimate reads long"
        )
        assert len(dones) >= expected_total, (
            "fewer ticks than units of work -- a tick is being skipped by an "
            "early exit in the loop body"
        )


@pytest.fixture
def rec(monkeypatch):
    telemetry.clear_stage_progress()
    r = _Recorder(monkeypatch)
    yield r
    telemetry.clear_stage_progress()


# --------------------------------------------------------------------------
# open_question_resolution -- the LARGEST stage in the pipeline
# --------------------------------------------------------------------------

class TestTheLargestStage:
    """36.8% of the compile, 204s median, and it had no counter."""

    def test_it_counts_the_questions_not_the_atoms(self, rec, monkeypatch):
        # The denominator is what COSTS: one decide() per open question. On a
        # real deal of 2,500 atoms with 40 questions, counting atoms would sit
        # the bar at 1.6% for the longest stage of the compile and then end.
        from app.core import taught_answers

        class _Store:
            pass

        class _D:
            verdict = "open"
            source = "store"

        monkeypatch.setattr(
            "app.core.decide.get_store", lambda: _Store(), raising=False
        )
        monkeypatch.setattr(
            "app.core.decide.decide", lambda *a, **k: _D(), raising=False
        )

        atoms = [_Atom("scope_item") for _ in range(50)]
        atoms += [_Atom("open_question", f"q{i}?", value={}) for i in range(7)]
        taught_answers.resolve_taught_answers(atoms, project_id="p")
        rec.assert_counts_up_to(7)

    def test_a_question_whose_decision_raised_still_counts(self, rec, monkeypatch):
        # Counting only the successes stalls the bar on exactly the compile
        # that is going wrong -- the one somebody is watching.
        from app.core import taught_answers

        monkeypatch.setattr(
            "app.core.decide.get_store", lambda: object(), raising=False
        )

        def _boom(*a, **k):
            raise RuntimeError("the decide store is down")

        monkeypatch.setattr("app.core.decide.decide", _boom, raising=False)

        atoms = [_Atom("open_question", f"q{i}?", value={}) for i in range(4)]
        taught_answers.resolve_taught_answers(atoms, project_id="p")
        rec.assert_counts_up_to(4)

    def test_a_question_already_closed_is_not_counted_as_work(self, rec, monkeypatch):
        # Key overlap closed it before this stage ran, so it costs nothing and
        # must not be in the denominator.
        from app.core import taught_answers

        class _D:
            verdict = "open"
            source = "store"

        monkeypatch.setattr(
            "app.core.decide.get_store", lambda: object(), raising=False
        )
        monkeypatch.setattr(
            "app.core.decide.decide", lambda *a, **k: _D(), raising=False
        )

        atoms = [_Atom("open_question", "already?", value={"answered": True})]
        atoms += [_Atom("open_question", f"q{i}?", value={}) for i in range(3)]
        taught_answers.resolve_taught_answers(atoms, project_id="p")
        rec.assert_counts_up_to(3)


# --------------------------------------------------------------------------
# bom_owner
# --------------------------------------------------------------------------

class TestBomOwner:
    def test_it_counts_bom_lines_through_their_early_exits(self, rec, monkeypatch):
        # This loop refuses an atom at four different points after the
        # decide(). The tick sits before all of them on purpose.
        from app.core import bom_owner

        class _D:
            verdict = "customer"
            confidence = 0.0  # below the floor: every atom takes an early exit
            source = "model"

        monkeypatch.setattr(
            "app.core.decide.decide", lambda *a, **k: _D(), raising=False
        )

        atoms = [_Atom("bom_line", f"24 x switch {i}") for i in range(6)]
        atoms += [_Atom("scope_item", "not a bom line")]
        atoms += [_Atom("bom_line", "   ")]  # no text: not work
        bom_owner.stamp_bom_owners(atoms, project_id="p")
        rec.assert_counts_up_to(6)


# --------------------------------------------------------------------------
# enrich_entities and source_replay -- counted per atom
# --------------------------------------------------------------------------

class TestPerAtomStages:
    def test_source_replay_counts_every_atom(self, rec, monkeypatch):
        from app.core import source_replay

        monkeypatch.setattr(
            source_replay, "replay_atom_receipts", lambda a, p: [], raising=False
        )
        atoms = [_Atom("scope_item") for _ in range(9)]
        source_replay.attach_receipts_to_atoms(atoms, {})
        rec.assert_counts_up_to(9)

    def test_enrich_entities_reaches_its_total(self, rec):
        # Its loop has five early exits and the tick is at the TOP, reporting
        # what is COMPLETE -- so the final value has to be set after the loop
        # or the counter would stop one short of its total forever.
        from app.core.entity_extraction import enrich_atoms

        from app.domain import load_domain_pack

        pack = load_domain_pack(None)

        atoms = [_Atom("scope_item", f"install {i} drops at the site") for i in range(5)]
        enrich_atoms(atoms, pack)
        rec.assert_counts_up_to(5)


# --------------------------------------------------------------------------
# The cross-stage property: a finished stage must not leak its count
# --------------------------------------------------------------------------

class TestTheCountDoesNotLeak:
    def test_ending_a_stage_clears_the_counter(self):
        # With one instrumented stage a stale count was invisible. With seven,
        # a count that outlived its stage would be read as the NEXT stage's
        # progress, and the estimator would compute a rate for the wrong work.
        t = telemetry.CompileTelemetry(compile_id="c", project_id="p")
        with t.stage("typed_atom_classification", input_count=10) as tok:
            t.set_stage_progress(7, 10)
            assert telemetry.stage_progress() == (7, 10)
            t.end_stage(tok, output_count=10)
        assert telemetry.stage_progress() == (0, 0), (
            "the next stage would inherit this and read as already underway"
        )
