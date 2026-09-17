from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.core.compiler import compile_project
from app.core.stage_plan import StagePlanError, plan_for_reparse
from app.storage.repositories import list_artifacts, save_compile_result

router = APIRouter(prefix="/projects", tags=["compile"])


@router.post("/{project_id}/compile")
def compile_endpoint(
    project_id: str,
    stages: str | None = Query(
        None,
        description=(
            "Comma-separated optional stages to run on this re-parse (PUR-58). "
            "Omit for a full parse. The result is labelled partial via stage_plan."
        ),
    ),
):
    # Validate before touching storage so a bad plan never produces a compile.
    try:
        plan = plan_for_reparse(stages)
    except StagePlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    try:
        artifact_rows = list_artifacts(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None
    artifact_dir = Path(".purtera_artifacts") / project_id
    if not artifact_rows:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    from app.learning.calibration import default_calibrator_path
    result = compile_project(
        project_dir=artifact_dir, project_id=project_id,
        persistence_hook=save_compile_result,
        calibrator_path=default_calibrator_path(),
        **({"stages": plan} if stages is not None else {}),
    )
    return result
