"""AtlasDispatch, hosted inside this service.

Atlas plans rollouts: post the post-award documents (signed SOW, runbook, site
roster) and get back a schedule CP-SAT has proved. It runs here rather than as
its own deployment because this service already exists, already runs FastAPI in
a container, and already has the operational story — logs, probes, rollback.
A second Python service to maintain buys nothing that matters yet.

The routes are Atlas's own `APIRouter`, mounted under `/atlas`. There is no
copy of the handlers here: one implementation, two possible hosts.

**Atlas is an optional dependency.** If it is not installed — a slim image, a
dev checkout without it — this service must still boot and serve compiles.
So the import is attempted once at module load and its failure is recorded
rather than raised; the endpoints then answer 503 with the reason, which is a
far better outcome than a container that will not start.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

log = logging.getLogger(__name__)

router = APIRouter(prefix="/atlas", tags=["atlas"])

_IMPORT_ERROR: str | None = None
_atlas_router = None

try:  # pragma: no cover - exercised by the presence/absence of the package
    from service.app import VERSION as _ATLAS_VERSION
    from service.app import router as _atlas_router
except Exception as exc:  # noqa: BLE001 - reported through /atlas/status
    _ATLAS_VERSION = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    log.warning("AtlasDispatch not available: %s", _IMPORT_ERROR)


@router.get("/status")
def atlas_status() -> dict:
    """Whether planning is available here, and why not when it is not.

    Deliberately separate from /health: this service is healthy whether or not
    Atlas is installed, and conflating the two would make a missing optional
    dependency look like an outage.
    """
    return {
        "available": _atlas_router is not None,
        "version": _ATLAS_VERSION,
        "error": _IMPORT_ERROR,
        "allowed_origins": [o for o in
                            os.getenv("ATLAS_ALLOWED_ORIGINS", "").split(",") if o],
    }


if _atlas_router is not None:
    router.include_router(_atlas_router)
else:
    @router.post("/v1/plan")
    @router.get("/v1/jobs/{job_id}")
    def _unavailable(job_id: str = "") -> None:
        raise HTTPException(
            503,
            "AtlasDispatch is not installed in this deployment "
            f"({_IMPORT_ERROR}). Install atlas-dispatch[service] to enable "
            "rollout planning.",
        )
