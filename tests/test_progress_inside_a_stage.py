"""A long stage says how far through it is, so a remaining time can be measured.

Progress was written only at STAGE BOUNDARIES. Measured over 485 real compiles,
the median compile spends 47% of its wall clock inside its single longest stage
(p75: 59%) -- so for about half of every compile nothing was observable.

Ten estimators were fitted against that telemetry. The best managed 61% median
error and 41% within 2x; predicting even the dominant stage from its own input
size, the most favourable case available, came out at 53.8%. The ceiling was
the blindness, not the model.

With a count inside the stage the estimate stops being an extrapolation from a
corpus and becomes arithmetic on a rate measured on THIS deal in THIS run.
"""

from __future__ import annotations

from app.core import telemetry


class TestTheCounter:
    def setup_method(self):
        telemetry.clear_stage_progress()

    def teardown_method(self):
        telemetry.clear_stage_progress()

    def test_it_reports_what_it_is_told(self):
        telemetry.set_stage_progress(340, 1200)
        assert telemetry.stage_progress() == (340, 1200)

    def test_zero_of_zero_means_unknown_not_finished(self):
        # A stage that cannot count its work must not read as complete, and
        # must not read as 0% of something either.
        assert telemetry.stage_progress() == (0, 0)
        telemetry.set_stage_progress(5, 0)
        assert telemetry.stage_progress() == (0, 0)

    def test_done_never_exceeds_total(self):
        # A bar past 100% is worse than no bar: it says the model is wrong
        # about the thing it is currently measuring.
        telemetry.set_stage_progress(9999, 1200)
        assert telemetry.stage_progress() == (1200, 1200)

    def test_garbage_is_ignored_rather_than_raising(self):
        # This runs inside the compile loop. It may never be the reason a
        # compile fails.
        telemetry.set_stage_progress("x", "y")  # type: ignore[arg-type]
        assert telemetry.stage_progress() == (0, 0)
        telemetry.set_stage_progress(None, None)  # type: ignore[arg-type]
        assert telemetry.stage_progress() == (0, 0)

    def test_a_negative_count_is_unknown(self):
        telemetry.set_stage_progress(-3, 10)
        assert telemetry.stage_progress() == (0, 0)


class TestItDoesNotOutliveItsStage:
    def test_ending_a_stage_clears_the_count(self):
        # Otherwise the next stage inherits it and reads as already underway --
        # and an estimate built on a stale denominator is worse than none.
        t = telemetry.CompileTelemetry(compile_id="c1", project_id="d1")
        with t.stage("typed_atom_classification", input_count=10) as st:
            telemetry.set_stage_progress(4, 10)
            assert telemetry.stage_progress() == (4, 10)
            t.end_stage(st, output_count=4)
        assert telemetry.stage_progress() == (0, 0)
