from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.file_safety import validate_upload
from app.storage.repositories import project_exists, save_artifact

router = APIRouter(prefix="/projects", tags=["artifacts"])


@router.post("/{project_id}/artifacts")
async def upload_artifact(project_id: str, file: UploadFile = File(...)):
    if not project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'")
    if not file.filename:
        raise HTTPException(status_code=400, detail="multipart field 'file' must include a filename")
    target_dir = Path(".purtera_artifacts") / project_id
    target_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    try:
        safe_upload = validate_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    target_path = target_dir / safe_upload.storage_filename
    target_path.write_bytes(content)
    try:
        return save_artifact(
            project_id=project_id,
            source_path=target_path,
            sha256=safe_upload.sha256,
            size_bytes=safe_upload.size_bytes,
            original_filename=safe_upload.sanitized_filename,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None
