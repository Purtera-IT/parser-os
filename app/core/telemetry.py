from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Callable, Iterator

from app.core.schemas import CompileStageTrace, CompileTrace

# Type alias for the per-stage callback the worker uses to write
# compile-progress.json after every stage finishes.
StageEndCallback = Callable[[CompileStageTrace, list[CompileStageTrace]], None]


# ──────────────────────────────────────────────────────────────────────
# Stage heartbeat watchdog (process-global, single daemon thread).
#
# The start/end stage logs go silent WHILE a stage runs, so a stage with no
# inner progress loop (an openpyxl load, one big embed call, a wedged dedup)
# is invisible until it finishes — and if it hangs, it never finishes. The
# watchdog prints `compile_stage_heartbeat` every SOWSMITH_HEARTBEAT_SECS for
# whatever stage is currently on top of the (LIFO) active stack, so EVERY
# stage proves it's alive and reports elapsed time. A hang shows up as the
# same stage heartbeating with a climbing elapsed_s — unambiguous, and it
# tells you exactly which stage to py-spy.
#
# Compiles in this process run sequentially, so a single shared stack + one
# daemon thread cover the whole run (set SOWSMITH_HEARTBEAT_SECS=0 to disable).
# ──────────────────────────────────────────────────────────────────────
_HB_LOCK = threading.Lock()
# entries: (token_id, stage_name, start_perf, compile_id, project_id, stream)
_HB_STACK: list[tuple[int, str, float, str, str, object]] = []
_HB_THREAD: threading.Thread | None = None


def _heartbeat_interval() -> float:
    try:
        v = float(os.environ.get("SOWSMITH_HEARTBEAT_SECS", "30"))
    except (TypeError, ValueError):
        v = 30.0
    return v if v > 0 else 0.0  # 0 (or negative) disables heartbeats


def _heartbeat_loop() -> None:  # pragma: no cover - timing/daemon thread
    while True:
        interval = _heartbeat_interval()
        threading.Event().wait(interval if interval > 0 else 30.0)
        if interval <= 0:
            continue
        with _HB_LOCK:
            if not _HB_STACK:
                continue
            _tid, stage, start_perf, cid, pid, stream = _HB_STACK[-1]
            elapsed = perf_counter() - start_perf
        try:
            print(
                json.dumps(
                    {
                        "event": "compile_stage_heartbeat",
                        "compile_id": cid,
                        "project_id": pid,
                        "stage": stage,
                        "elapsed_s": round(elapsed, 1),
                    },
                    ensure_ascii=True,
                ),
                file=stream,
                flush=True,
            )
        except Exception:
            pass


def _ensure_heartbeat_thread() -> None:
    global _HB_THREAD
    if _HB_THREAD is not None or _heartbeat_interval() <= 0:
        return
    _HB_THREAD = threading.Thread(
        target=_heartbeat_loop, name="compile-heartbeat", daemon=True
    )
    _HB_THREAD.start()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _StageToken:
    stage_name: str
    started_at: str
    started_perf: float
    input_count: int | None


class CompileTelemetry:
    def __init__(
        self,
        project_id: str,
        compile_id: str = "pending_compile_id",
        stream=None,
        on_stage_end: StageEndCallback | None = None,
    ) -> None:
        self.project_id = project_id
        self.compile_id = compile_id
        self._stream = stream or sys.stderr
        self._compile_start_perf = perf_counter()
        self.stages: list[CompileStageTrace] = []
        # Optional: worker uses this to write compile-progress.json after
        # every stage so the UI can render live pipeline progress.
        self._on_stage_end: StageEndCallback | None = on_stage_end

    def set_compile_id(self, compile_id: str) -> None:
        self.compile_id = compile_id

    def _emit_log(
        self,
        *,
        event: str,
        stage: str,
        duration_ms: float,
        input_count: int | None,
        output_count: int | None,
        warnings: list[str],
        errors: list[str],
    ) -> None:
        payload = {
            "event": event,
            "compile_id": self.compile_id,
            "project_id": self.project_id,
            "stage": stage,
            "duration_ms": round(duration_ms, 3),
            "counts": {"input_count": input_count, "output_count": output_count},
            "warning_count": len(warnings),
            "error_count": len(errors),
        }
        print(json.dumps(payload, ensure_ascii=True), file=self._stream)

    def start_stage(self, stage_name: str, input_count: int | None = None) -> _StageToken:
        # Announce the stage at START, not just at end. The end-only log made a
        # slow/wedged stage invisible (it never prints until it finishes), so an
        # operator couldn't tell where a long compile was stuck. This start
        # banner makes the live position obvious.
        print(
            json.dumps(
                {
                    "event": "compile_stage_started",
                    "compile_id": self.compile_id,
                    "project_id": self.project_id,
                    "stage": stage_name,
                    "counts": {"input_count": input_count},
                },
                ensure_ascii=True,
            ),
            file=self._stream,
            flush=True,
        )
        token = _StageToken(
            stage_name=stage_name,
            started_at=utc_now_iso(),
            started_perf=perf_counter(),
            input_count=input_count,
        )
        # Register on the heartbeat stack so the watchdog can report this
        # stage's liveness/elapsed every interval until end_stage pops it.
        _ensure_heartbeat_thread()
        with _HB_LOCK:
            _HB_STACK.append(
                (
                    id(token),
                    stage_name,
                    token.started_perf,
                    self.compile_id,
                    self.project_id,
                    self._stream,
                )
            )
        return token

    def end_stage(
        self,
        token: _StageToken,
        *,
        output_count: int | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> CompileStageTrace:
        # Pop this token off the heartbeat stack (match by identity so a
        # token popped out of LIFO order — e.g. an exception unwinding nested
        # stages — is still removed cleanly and never heartbeats again).
        _tok_id = id(token)
        with _HB_LOCK:
            for _i in range(len(_HB_STACK) - 1, -1, -1):
                if _HB_STACK[_i][0] == _tok_id:
                    _HB_STACK.pop(_i)
                    break
        warning_rows = sorted(warnings or [])
        error_rows = sorted(errors or [])
        completed_at = utc_now_iso()
        duration_ms = max(0.0, (perf_counter() - token.started_perf) * 1000.0)
        stage = CompileStageTrace(
            stage_name=token.stage_name,
            started_at=token.started_at,
            completed_at=completed_at,
            duration_ms=round(duration_ms, 3),
            input_count=token.input_count,
            output_count=output_count,
            warnings=warning_rows,
            errors=error_rows,
        )
        self.stages.append(stage)
        self._emit_log(
            event="compile_stage_completed",
            stage=token.stage_name,
            duration_ms=stage.duration_ms,
            input_count=token.input_count,
            output_count=output_count,
            warnings=warning_rows,
            errors=error_rows,
        )
        if self._on_stage_end is not None:
            try:
                self._on_stage_end(stage, list(self.stages))
            except Exception:
                # Never let a progress callback failure crash the compile.
                pass
        return stage

    @contextmanager
    def stage(
        self,
        stage_name: str,
        *,
        input_count: int | None = None,
    ) -> Iterator[_StageToken]:
        token = self.start_stage(stage_name, input_count=input_count)
        try:
            yield token
        except Exception as exc:
            self.end_stage(token, output_count=None, warnings=[], errors=[str(exc)])
            raise

    def build_trace(
        self,
        *,
        artifact_count: int,
        atom_count: int,
        entity_count: int,
        edge_count: int,
        packet_count: int,
        parser_atom_counts: dict[str, int],
        packet_family_counts: dict[str, int],
        parser_routing: list[dict],
    ) -> CompileTrace:
        total_duration_ms = max(0.0, (perf_counter() - self._compile_start_perf) * 1000.0)
        return CompileTrace(
            compile_id=self.compile_id,
            project_id=self.project_id,
            stages=list(self.stages),
            total_duration_ms=round(total_duration_ms, 3),
            artifact_count=artifact_count,
            atom_count=atom_count,
            entity_count=entity_count,
            edge_count=edge_count,
            packet_count=packet_count,
            parser_atom_counts=dict(sorted(parser_atom_counts.items())),
            packet_family_counts=dict(sorted(packet_family_counts.items())),
            parser_routing=list(parser_routing),
        )
