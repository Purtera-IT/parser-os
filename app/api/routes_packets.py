from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.core.schemas import AtomType, AuthorityClass, PacketFamily, PacketStatus, ReviewStatus
from app.storage.repositories import get_atoms, get_edges, get_entities, get_packets

router = APIRouter(prefix="/projects", tags=["compile-results"])


@router.get("/{project_id}/atoms")
def project_atoms(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    atom_type: AtomType | None = Query(default=None),
    authority_class: AuthorityClass | None = Query(default=None),
    entity_key: str | None = Query(default=None),
    review_status: ReviewStatus | None = Query(default=None),
):
    try:
        return get_atoms(
            project_id,
            limit=limit,
            offset=offset,
            atom_type=atom_type.value if atom_type else None,
            authority_class=authority_class.value if authority_class else None,
            entity_key=entity_key,
            review_status=review_status.value if review_status else None,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None


@router.get("/{project_id}/edges")
def project_edges(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_edges(project_id, limit=limit, offset=offset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None


@router.get("/{project_id}/entities")
def project_entities(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_entities(project_id, limit=limit, offset=offset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None


@router.get("/{project_id}/packets")
def project_packets(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    family: PacketFamily | None = Query(default=None),
    status: PacketStatus | None = Query(default=None),
    severity: Literal["low", "medium", "high", "critical"] | None = Query(default=None),
    anchor_key_contains: str | None = Query(default=None),
    review_priority_lte: int | None = Query(default=None, ge=1, le=5),
):
    try:
        return get_packets(
            project_id,
            limit=limit,
            offset=offset,
            family=family.value if family else None,
            status=status.value if status else None,
            severity=severity,
            anchor_key_contains=anchor_key_contains,
            review_priority_lte=review_priority_lte,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'") from None
