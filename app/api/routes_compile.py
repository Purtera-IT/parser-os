from __future__ import annotations

import json
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


@router.get("/{project_id}/compile/progress")
def compile_progress_endpoint(project_id: str):
    """v45.2 — serve the in-flight compile's progress.json for live UI bars.

    The ProgressTracker writes to {artifact_dir}/.orbitbrief/progress.json on
    every stage / substage event.  We tail it here; the file is updated
    atomically so reads never see a half-written JSON object.

    Returns 404 if no compile has ever been started for this project.  When a
    compile has just completed, status will read "completed" and percent
    will read 100 — the frontend can stop polling at that point.
    """
    artifact_dir = Path(".purtera_artifacts") / project_id
    progress_path = artifact_dir / ".orbitbrief" / "progress.json"
    if not progress_path.exists():
        # Also check the env-controlled fallback location
        try:
            from app.core.progress_tracker import get_active_tracker
            tracker = get_active_tracker()
            if tracker is not None and tracker.deal_id == project_id:
                return json.loads(tracker.out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail=f"No compile progress found for project '{project_id}' — "
            "compile has not been started.",
        )
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read compile progress: {e}",
        ) from e
