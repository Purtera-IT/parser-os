from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.ids import stable_id
from app.core.schemas import CandidateAtom, EvidenceAtom
from app.storage.repositories import load_cache_payload, save_cache_payload


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class CachedArtifactResult(BaseModel):
    artifact_id: str
    sha256: str
    parser_name: str
    parser_version: str
    domain_pack_id: str
    domain_pack_version: str
    segments: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[CandidateAtom] = Field(default_factory=list)
    atoms: list[EvidenceAtom] = Field(default_factory=list)
    receipts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str


def cache_key_for_artifact(
    *,
    artifact_id: str,
    sha256: str,
    parser_name: str,
    parser_version: str,
    domain_pack_id: str,
    domain_pack_version: str,
) -> str:
    return stable_id(
        "artifact_cache",
        artifact_id,
        sha256,
        parser_name,
        parser_version,
        domain_pack_id,
        domain_pack_version,
    )


def compute_artifact_sha256(path: Path) -> str:
    return _sha256_file(path)


def load_cached_artifact_result(
    *,
    artifact_id: str,
    sha256: str,
    parser_name: str,
    parser_version: str,
    domain_pack_id: str,
    domain_pack_version: str,
) -> CachedArtifactResult | None:
    key = cache_key_for_artifact(
        artifact_id=artifact_id,
        sha256=sha256,
        parser_name=parser_name,
        parser_version=parser_version,
        domain_pack_id=domain_pack_id,
        domain_pack_version=domain_pack_version,
    )
    payload = load_cache_payload(key)
    if payload is None:
        return None
    try:
        return CachedArtifactResult.model_validate(payload)
    except Exception:
        return None


def save_cached_artifact_result(result: CachedArtifactResult) -> None:
    key = cache_key_for_artifact(
        artifact_id=result.artifact_id,
        sha256=result.sha256,
        parser_name=result.parser_name,
        parser_version=result.parser_version,
        domain_pack_id=result.domain_pack_id,
        domain_pack_version=result.domain_pack_version,
    )
    save_cache_payload(key, result.model_dump(mode="json"))


def build_cached_artifact_result(
    *,
    artifact_id: str,
    sha256: str,
    parser_name: str,
    parser_version: str,
    domain_pack_id: str,
    domain_pack_version: str,
    candidates: list[CandidateAtom],
    atoms: list[EvidenceAtom],
    warnings: list[str],
) -> CachedArtifactResult:
    return CachedArtifactResult(
        artifact_id=artifact_id,
        sha256=sha256,
        parser_name=parser_name,
        parser_version=parser_version,
        domain_pack_id=domain_pack_id,
        domain_pack_version=domain_pack_version,
        segments=[],
        candidates=list(candidates),
        atoms=list(atoms),
        receipts=[],
        warnings=list(warnings),
        created_at=_now_iso(),
    )
