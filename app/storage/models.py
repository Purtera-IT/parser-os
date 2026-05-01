from __future__ import annotations

from pydantic import BaseModel


class ProjectRow(BaseModel):
    project_id: str
    name: str
    created_at: str


class ArtifactRow(BaseModel):
    artifact_id: str
    project_id: str
    source_path: str
    stored_path: str
    metadata_json: str


class CompileResultRow(BaseModel):
    project_id: str
    result_json: str
    updated_at: str
