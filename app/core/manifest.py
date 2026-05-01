from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from app.domain.schemas import DomainPack
from app.core.ids import canonical_json_hash, stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    AUTHORITY_POLICY_VERSION,
    COMPILER_VERSION,
    PACKETIZER_VERSION,
    SCHEMA_VERSION,
    ArtifactFingerprint,
    ArtifactType,
    CompileManifest,
    CompileResult,
    EvidenceAtom,
)

DETERMINISTIC_SEED = "purtera-evidence-compiler-deterministic-v1"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_type_for_path(path: Path) -> ArtifactType:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return ArtifactType.xlsx
    if suffix == ".csv":
        return ArtifactType.csv
    if suffix == ".docx":
        return ArtifactType.docx
    if suffix in {".eml"}:
        return ArtifactType.email
    if suffix in {".txt", ".md", ".vtt", ".srt", ".json"}:
        return ArtifactType.transcript if "transcript" in path.name.lower() else ArtifactType.txt
    return ArtifactType.txt


def build_artifact_fingerprint(
    path: Path,
    artifact_id: str,
    parsed_atoms: list[EvidenceAtom],
    filename: str | None = None,
    parser_name: str | None = None,
    parser_version: str | None = None,
) -> ArtifactFingerprint:
    stat = path.stat()
    resolved_parser_name = parser_name or "unknown"
    resolved_parser_version = parser_version or "unknown"
    if parsed_atoms and (resolved_parser_name == "unknown" or resolved_parser_version == "unknown"):
        ref = parsed_atoms[0].source_refs[0] if parsed_atoms[0].source_refs else None
        if ref is not None:
            resolved_parser_name = ref.parser or ref.artifact_type.value
            resolved_parser_version = ref.parser_version

    return ArtifactFingerprint(
        artifact_id=artifact_id,
        filename=filename or path.name,
        artifact_type=_artifact_type_for_path(path),
        sha256=_sha256_file(path),
        size_bytes=stat.st_size,
        modified_time=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        parser_name=resolved_parser_name,
        parser_version=resolved_parser_version,
    )


def compute_input_signature(artifact_fingerprints: list[ArtifactFingerprint]) -> str:
    payload = [
        {
            "filename": af.filename,
            "sha256": af.sha256,
            "parser_name": af.parser_name,
            "parser_version": af.parser_version,
        }
        for af in sorted(artifact_fingerprints, key=lambda x: (x.filename, x.sha256, x.artifact_id))
    ]
    return canonical_json_hash(payload)


def compute_compile_id(project_id: str, artifact_fingerprints: list[ArtifactFingerprint]) -> str:
    artifact_hashes = sorted(f"{af.filename}:{af.sha256}" for af in artifact_fingerprints)
    return stable_id(
        "cmp",
        project_id,
        artifact_hashes,
        COMPILER_VERSION,
        PACKETIZER_VERSION,
        AUTHORITY_POLICY_VERSION,
    )


def create_manifest(
    project_id: str,
    artifact_fingerprints: list[ArtifactFingerprint],
    *,
    domain_pack: DomainPack | None = None,
) -> CompileManifest:
    parser_versions: dict[str, str] = {}
    for af in sorted(artifact_fingerprints, key=lambda x: (x.parser_name, x.filename, x.artifact_id)):
        parser_versions.setdefault(af.parser_name, af.parser_version)
    compile_id = compute_compile_id(project_id, artifact_fingerprints)
    return CompileManifest(
        compile_id=compile_id,
        project_id=project_id,
        schema_version=SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
        packetizer_version=PACKETIZER_VERSION,
        authority_policy_version=AUTHORITY_POLICY_VERSION,
        artifact_fingerprints=sorted(artifact_fingerprints, key=lambda x: x.artifact_id),
        parser_versions=dict(sorted(parser_versions.items(), key=lambda kv: kv[0])),
        started_at=_now_iso(),
        completed_at=None,
        deterministic_seed=DETERMINISTIC_SEED,
        input_signature=compute_input_signature(artifact_fingerprints),
        output_signature=None,
        domain_pack_id=domain_pack.pack_id if domain_pack else None,
        domain_pack_version=domain_pack.version if domain_pack else None,
    )


def compute_output_signature(result: CompileResult) -> str:
    atoms_payload = []
    for atom in result.atoms:
        atom_payload = atom.model_dump(mode="json")
        receipts = []
        for receipt in atom.receipts:
            receipt_payload = receipt.model_dump(mode="json")
            # Replay snippets are for human audit and may differ in inconsequential
            # whitespace; normalize for deterministic semantic signatures.
            receipt_payload["extracted_snippet"] = (
                normalize_text(receipt.extracted_snippet) if receipt.extracted_snippet else None
            )
            receipts.append(receipt_payload)
        atom_payload["receipts"] = sorted(receipts, key=lambda x: x["source_ref_id"])
        atoms_payload.append(atom_payload)

    payload = {
        "atoms": atoms_payload,
        "entities": [entity.model_dump(mode="json") for entity in result.entities],
        "edges": [edge.model_dump(mode="json") for edge in result.edges],
        "packets": [packet.model_dump(mode="json") for packet in result.packets],
    }
    return canonical_json_hash(payload)


def finalize_manifest(manifest: CompileManifest, output_signature: str) -> CompileManifest:
    manifest.completed_at = _now_iso()
    manifest.output_signature = output_signature
    return manifest
