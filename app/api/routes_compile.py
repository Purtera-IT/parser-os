from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.compiler import compile_project
from app.storage.repositories import list_artifacts, save_compile_result

router = APIRouter(prefix="/projects", tags=["compile"])


@router.post("/{project_id}/compile")
def compile_endpoint(project_id: str):
    try:
        artifact_rows = list_artifacts(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None
    artifact_dir = Path(".purtera_artifacts") / project_id
    if not artifact_rows:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    result = compile_project(project_dir=artifact_dir, project_id=project_id, persistence_hook=save_compile_result)
    return result
