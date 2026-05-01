from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import CompileResult, EvidenceAtom, EvidenceEdge, EvidencePacket
from app.storage.db import get_connection, init_db
from app.storage.models import ArtifactRow, CompileResultRow, ProjectRow

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = _PROJECT_ROOT / ".purtera_artifacts" / "cache"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_project(name: str) -> str:
    init_db()
    project_id = stable_id("proj", name, _now_iso())
    row = ProjectRow(project_id=project_id, name=name, created_at=_now_iso())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO projects (project_id, name, created_at) VALUES (?, ?, ?)",
            (row.project_id, row.name, row.created_at),
        )
        conn.commit()
    return project_id


def project_exists(project_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    return row is not None


def save_artifact(
    project_id: str,
    source_path: Path,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    init_db()
    if not project_exists(project_id):
        raise KeyError(f"Unknown project: {project_id}")
    artifact_id = stable_id("art", project_id, str(source_path))
    metadata = {
        "artifact_id": artifact_id,
        "project_id": project_id,
        "source_path": str(source_path),
        "stored_path": str(source_path),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "original_filename": original_filename,
    }
    row = ArtifactRow(
        artifact_id=artifact_id,
        project_id=project_id,
        source_path=str(source_path),
        stored_path=str(source_path),
        metadata_json=json.dumps(metadata),
    )
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts
            (artifact_id, project_id, source_path, stored_path, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row.artifact_id,
                row.project_id,
                row.source_path,
                row.stored_path,
                row.metadata_json,
            ),
        )
        conn.commit()
    return metadata


def list_artifacts(project_id: str) -> list[dict[str, Any]]:
    init_db()
    if not project_exists(project_id):
        raise KeyError(f"Unknown project: {project_id}")
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT metadata_json FROM artifacts WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    return [json.loads(row["metadata_json"]) for row in rows]


def save_compile_result(result: CompileResult) -> None:
    init_db()
    row = CompileResultRow(
        project_id=result.project_id,
        result_json=result.model_dump_json(),
        updated_at=_now_iso(),
    )
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO compile_results (project_id, result_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (row.project_id, row.result_json, row.updated_at),
        )
        conn.commit()


def _load_compile_result(project_id: str) -> CompileResult:
    init_db()
    if not project_exists(project_id):
        raise KeyError(f"Unknown project: {project_id}")
    with get_connection() as conn:
        row = conn.execute(
            "SELECT result_json FROM compile_results WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    if row is None:
        return CompileResult(
            project_id=project_id,
            atoms=[],
            entities=[],
            edges=[],
            packets=[],
            warnings=["No compile result saved yet."],
        )
    return CompileResult.model_validate_json(row["result_json"])


def _paginate(items: list[Any], *, limit: int, offset: int) -> dict[str, Any]:
    ordered = list(items)
    total = len(ordered)
    paged = ordered[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": paged,
    }


def load_cache_payload(cache_key: str) -> dict[str, Any] | None:
    path = CACHE_ROOT / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def save_cache_payload(cache_key: str, payload: dict[str, Any]) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{cache_key}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def get_packets(
    project_id: str,
    *,
    family: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    anchor_key_contains: str | None = None,
    review_priority_lte: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    packets: list[EvidencePacket] = _load_compile_result(project_id).packets
    if family is not None:
        packets = [packet for packet in packets if packet.family.value == family]
    if status is not None:
        packets = [packet for packet in packets if packet.status.value == status]
    if severity is not None:
        packets = [
            packet
            for packet in packets
            if packet.risk is not None and packet.risk.severity == severity
        ]
    if anchor_key_contains is not None:
        token = anchor_key_contains.strip().lower()
        packets = [packet for packet in packets if token in packet.anchor_key.lower()]
    if review_priority_lte is not None:
        packets = [
            packet
            for packet in packets
            if packet.risk is not None and packet.risk.review_priority <= review_priority_lte
        ]
    packets = sorted(packets, key=lambda packet: packet.id)
    return _paginate(packets, limit=limit, offset=offset)


def get_atoms(
    project_id: str,
    *,
    atom_type: str | None = None,
    authority_class: str | None = None,
    entity_key: str | None = None,
    review_status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    atoms: list[EvidenceAtom] = _load_compile_result(project_id).atoms
    if atom_type is not None:
        atoms = [atom for atom in atoms if atom.atom_type.value == atom_type]
    if authority_class is not None:
        atoms = [atom for atom in atoms if atom.authority_class.value == authority_class]
    if entity_key is not None:
        atoms = [atom for atom in atoms if entity_key in atom.entity_keys]
    if review_status is not None:
        atoms = [atom for atom in atoms if atom.review_status.value == review_status]
    atoms = sorted(atoms, key=lambda atom: atom.id)
    return _paginate(atoms, limit=limit, offset=offset)


def get_edges(project_id: str, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    edges: list[EvidenceEdge] = sorted(_load_compile_result(project_id).edges, key=lambda edge: edge.id)
    return _paginate(edges, limit=limit, offset=offset)


def get_entities(project_id: str, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    entities = sorted(_load_compile_result(project_id).entities, key=lambda entity: entity.id)
    return _paginate(entities, limit=limit, offset=offset)
