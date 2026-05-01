from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.schemas import AUTHORITY_POLICY_VERSION, COMPILER_VERSION, PACKETIZER_VERSION, SCHEMA_VERSION
from app.storage.db import get_connection

router = APIRouter(tags=["health"])


@router.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready() -> dict[str, str]:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        raise HTTPException(status_code=503, detail="Service not ready") from None
    return {"status": "ready"}


@router.get("/version")
def version() -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "packetizer_version": PACKETIZER_VERSION,
        "authority_policy_version": AUTHORITY_POLICY_VERSION,
    }
