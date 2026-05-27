from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from app.core.schemas import AUTHORITY_POLICY_VERSION, COMPILER_VERSION, PACKETIZER_VERSION, SCHEMA_VERSION
from app.storage.db import get_connection

router = APIRouter(tags=["health"])


def _resolve_parser_os_sha() -> str:
    """v45.2 — return the commit SHA the image was built from.

    Resolution order (first non-empty wins):
      1. PARSER_OS_SHA env var  (set in Dockerfile via `ENV PARSER_OS_SHA=$GIT_SHA`)
      2. GIT_SHA env var        (alternative name, some deploy scripts use this)
      3. Build-stamp file /app/.git_sha  (written at docker build time)
      4. Live `git rev-parse HEAD` if .git exists  (local dev fallback)
      5. "unknown"

    Lets ops verify the running container actually has the expected code via:
        curl /v1/version | jq -r .parser_os_sha
    """
    for env_var in ("PARSER_OS_SHA", "GIT_SHA"):
        val = os.environ.get(env_var, "").strip()
        if val and val != "unknown":
            return val
    # Build-stamp file (recommended in the Dockerfile snippet)
    try:
        stamp = "/app/.git_sha"
        if os.path.exists(stamp):
            with open(stamp, encoding="utf-8") as fh:
                val = fh.read().strip()
                if val:
                    return val
    except Exception:
        pass
    # Local-dev fallback
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


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
        # v45.2 — git SHA the running image was built from. Lets ops verify
        # the deployed container actually contains the expected commit.
        "parser_os_sha": _resolve_parser_os_sha(),
        # Also surface the build label (e.g. "v45.2") if set at build time.
        "build_label": os.environ.get("PARSER_OS_BUILD_LABEL", "unknown"),
    }
