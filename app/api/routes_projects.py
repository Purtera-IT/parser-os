from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.storage.repositories import create_project

router = APIRouter(prefix="/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str


@router.post("")
def post_project(payload: CreateProjectRequest):
    return {"project_id": create_project(name=payload.name)}
