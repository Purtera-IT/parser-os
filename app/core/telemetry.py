from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Iterator

from app.core.schemas import CompileStageTrace, CompileTrace


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _StageToken:
    stage_name: str
    started_at: str
    started_perf: float
    input_count: int | None


class CompileTelemetry:
    def __init__(self, project_id: str, compile_id: str = "pending_compile_id", stream=None) -> None:
        self.project_id = project_id
        self.compile_id = compile_id
        self._stream = stream or sys.stderr
        self._compile_start_perf = perf_counter()
        self.stages: list[CompileStageTrace] = []

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
        return _StageToken(
            stage_name=stage_name,
            started_at=utc_now_iso(),
            started_perf=perf_counter(),
            input_count=input_count,
        )

    def end_stage(
        self,
        token: _StageToken,
        *,
        output_count: int | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> CompileStageTrace:
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
