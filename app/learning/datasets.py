from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.schemas import CompileResult
from app.learning.features import build_atom_feature_row, build_packet_feature_row


def _bool_to_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "correct"}:
            return 1
        if lowered in {"0", "false", "no", "incorrect"}:
            return 0
    return None


def _load_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_packet_labels(path: Path) -> dict[str, int]:
    payload = _load_payload(path)
    labels: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("reviews"), list):
            rows.extend([row for row in payload["reviews"] if isinstance(row, dict)])
        if isinstance(payload.get("packet_labels"), list):
            for row in payload["packet_labels"]:
                if not isinstance(row, dict):
                    continue
                merged = dict(row)
                human = row.get("human_label")
                if isinstance(human, dict):
                    merged.update(human)
                rows.append(merged)
        if isinstance(payload.get("labels"), list):
            rows.extend([row for row in payload["labels"] if isinstance(row, dict)])
    elif isinstance(payload, list):
        rows.extend([row for row in payload if isinstance(row, dict)])

    for row in rows:
        packet_id = row.get("packet_id") or row.get("id")
        if not packet_id:
            continue
        label = _bool_to_label(row.get("correct_packet"))
        if label is None:
            label = _bool_to_label(row.get("is_correct"))
        if label is None:
            label = _bool_to_label(row.get("label"))
        if label is None:
            continue
        labels[str(packet_id)] = label
    return labels


def load_atom_labels(path: Path) -> dict[str, int]:
    payload = _load_payload(path)
    labels: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("atom_labels"), list):
        rows.extend([row for row in payload["atom_labels"] if isinstance(row, dict)])
    if isinstance(payload, list):
        rows.extend([row for row in payload if isinstance(row, dict) and ("atom_id" in row or "id" in row)])
    for row in rows:
        atom_id = row.get("atom_id") or row.get("id")
        if not atom_id:
            continue
        label = _bool_to_label(row.get("is_correct"))
        if label is None:
            label = _bool_to_label(row.get("label"))
        if label is None:
            continue
        labels[str(atom_id)] = label
    return labels


def build_packet_training_rows(labels_path: Path, compile_results: list[CompileResult]) -> tuple[list[dict], list[int]]:
    labels = load_packet_labels(labels_path)
    features: list[dict] = []
    y: list[int] = []
    for result in compile_results:
        for packet in result.packets:
            if packet.id not in labels:
                continue
            features.append(build_packet_feature_row(packet, result.atoms))
            y.append(labels[packet.id])
    return features, y


def build_atom_training_rows(labels_path: Path, compile_results: list[CompileResult]) -> tuple[list[dict], list[int]]:
    labels = load_atom_labels(labels_path)
    features: list[dict] = []
    y: list[int] = []
    for result in compile_results:
        for atom in result.atoms:
            if atom.id not in labels:
                continue
            features.append(build_atom_feature_row(atom))
            y.append(labels[atom.id])
    return features, y
