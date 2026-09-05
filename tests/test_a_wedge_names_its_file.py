"""A stage that loops over files must say which file it is on.

On 2026-09-05 deal 5bd32822 sat in `parse_artifacts` for 6.8 hours. The
heartbeat proved the process was alive and reported the stage and the elapsed
time — and could not say which of the 35 artifacts had stopped. Both large
spreadsheets were already parsed and the twenty files left were 0-2 KB apiece,
so the size told us nothing either.

Diagnosing it needed `py-spy` inside the container, which needs a TTY that a
non-interactive session does not have. The log tail should have been enough.
"""

from __future__ import annotations

import io
import json

from app.core import telemetry


def _heartbeat_payload(stage_name: str, item: str | None) -> dict:
    """Build the line the watchdog would print, without waiting 30s for it."""
    stream = io.StringIO()
    tel = telemetry.CompileTelemetry(compile_id="c1", project_id="p1")
    # Through the INSTANCE, exactly as the compiler calls it — the module-level
    # function is not reachable on an instance, and that is how the first cut
    # of this would have raised on the first artifact of every compile.
    tel.set_stage_item(item or "")
    with tel.stage(stage_name):
        with telemetry._HB_LOCK:
            _tid, stage, start_perf, cid, pid, _s = telemetry._HB_STACK[-1]
            current = telemetry._HB_ITEM[0]
        payload = {
            "event": "compile_stage_heartbeat",
            "compile_id": cid,
            "project_id": pid,
            "stage": stage,
            "elapsed_s": 0.0,
            **({"item": current} if current else {}),
        }
    print(json.dumps(payload), file=stream)
    tel.set_stage_item("")
    return payload


def test_the_heartbeat_names_the_file_it_is_on() -> None:
    p = _heartbeat_payload("parse_artifacts", "000092-fireflies-transcript.json")
    assert p["item"] == "000092-fireflies-transcript.json"
    assert p["stage"] == "parse_artifacts"


def test_a_stage_with_no_current_item_says_nothing_extra() -> None:
    """Most stages are not per-file; they must not grow an empty field."""
    p = _heartbeat_payload("semantic_dedup", None)
    assert "item" not in p


def test_the_parse_loop_sets_it_and_clears_it() -> None:
    import inspect

    from app.core import compiler

    src = inspect.getsource(compiler)
    body = src[src.index('telemetry.stage("parse_artifacts"'):]
    assert "telemetry.set_stage_item(relative_name)" in body, "the loop must name its file"
    # Cleared at the end, or every later stage inherits the last filename and
    # reads as though it were still parsing it.
    assert 'telemetry.set_stage_item("")' in body


def test_a_long_name_cannot_flood_the_log() -> None:
    telemetry.set_stage_item("x" * 5000)
    assert len(telemetry._HB_ITEM[0]) <= 200
    telemetry.set_stage_item("")


def test_none_is_treated_as_no_item() -> None:
    telemetry.set_stage_item(None)  # type: ignore[arg-type]
    assert telemetry._HB_ITEM[0] == ""


def test_the_compiler_can_reach_it_on_the_object_it_holds() -> None:
    """`telemetry` in the compiler is a CompileTelemetry, not the module. The
    first cut of this called the module function through the instance and would
    have raised AttributeError on the first artifact of every compile."""
    tel = telemetry.CompileTelemetry(compile_id="c1", project_id="p1")
    assert hasattr(tel, "set_stage_item")
    tel.set_stage_item("a.json")
    assert telemetry._HB_ITEM[0] == "a.json"
    tel.set_stage_item("")
