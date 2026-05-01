from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.schemas import CompileResult, PacketStatus, ReviewStatus
from app.learning.datasets import (
    build_atom_training_rows,
    build_packet_training_rows,
)
from app.learning.features import build_atom_feature_row, build_packet_feature_row, summarize_label_balance


def _train_pipeline(features: list[dict], labels: list[int]) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline


def train_calibrator(
    labels_path: Path,
    compile_results: list[CompileResult],
    model_path: Path,
) -> dict[str, Any]:
    packet_features, packet_labels = build_packet_training_rows(labels_path, compile_results)
    atom_features, atom_labels = build_atom_training_rows(labels_path, compile_results)

    if not packet_features:
        raise ValueError("No packet labels matched compile results; cannot train calibrator")
    if len(set(packet_labels)) < 2:
        raise ValueError("Packet labels need both positive and negative examples for calibration")

    packet_model = _train_pipeline(packet_features, packet_labels)
    atom_model = None
    if atom_features and len(set(atom_labels)) >= 2:
        atom_model = _train_pipeline(atom_features, atom_labels)

    artifact = {
        "version": "calibrator_v1",
        "packet_model": packet_model,
        "atom_model": atom_model,
        "packet_label_balance": summarize_label_balance(packet_labels),
        "atom_label_balance": summarize_label_balance(atom_labels),
        "feature_spec": {
            "packet": sorted(packet_features[0].keys()) if packet_features else [],
            "atom": sorted(atom_features[0].keys()) if atom_features else [],
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return {
        "model_path": str(model_path),
        "packet_samples": len(packet_labels),
        "atom_samples": len(atom_labels),
        "packet_label_balance": artifact["packet_label_balance"],
        "atom_label_balance": artifact["atom_label_balance"],
    }


def load_calibrator(model_path: Path) -> dict[str, Any]:
    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise ValueError("Calibrator artifact must be a dictionary")
    if "packet_model" not in payload:
        raise ValueError("Calibrator artifact missing packet_model")
    return payload


def apply_calibration(
    result: CompileResult,
    model_path: Path,
    *,
    abstain_threshold: float = 0.70,
) -> CompileResult:
    artifact = load_calibrator(model_path)
    packet_model = artifact.get("packet_model")
    atom_model = artifact.get("atom_model")

    calibrated = result.model_copy(deep=True)
    if packet_model is not None:
        for packet in calibrated.packets:
            row = build_packet_feature_row(packet, calibrated.atoms)
            probability = float(packet_model.predict_proba([row])[0][1])
            packet.confidence_raw = float(packet.confidence_raw if packet.confidence_raw is not None else packet.confidence)
            packet.calibrated_confidence = probability
            packet.confidence = probability
            if probability < abstain_threshold and packet.status in {PacketStatus.active}:
                packet.status = PacketStatus.needs_review
                flags = set(packet.review_flags)
                flags.add("calibration_abstain")
                packet.review_flags = sorted(flags)

    if atom_model is not None:
        for atom in calibrated.atoms:
            row = build_atom_feature_row(atom)
            probability = float(atom_model.predict_proba([row])[0][1])
            atom.confidence_raw = float(atom.confidence_raw if atom.confidence_raw is not None else atom.confidence)
            atom.calibrated_confidence = probability
            atom.confidence = probability
            if probability < abstain_threshold and atom.review_status == ReviewStatus.auto_accepted:
                atom.review_status = ReviewStatus.needs_review
                flags = set(atom.review_flags)
                flags.add("calibration_abstain")
                atom.review_flags = sorted(flags)

    return calibrated
