"""Compile-time ML capability manifest — proves which heads were armed and fired."""
from __future__ import annotations

import os
from typing import Any

# Env flags that arm ML heads (subset — observability only)
_ML_ENV_FLAGS = (
    "SOWSMITH_ML_PROFILE",
    "SOWSMITH_TRAINING_LOG_DB",
    "SOWSMITH_FEEDBACK_STORE_DB",
    "SOWSMITH_EMBED_CACHE_DB",
    "SOWSMITH_ATOM_TYPE_DEFLECT",
    "SOWSMITH_TYPE_HEAD_DEFLECT",
    "SOWSMITH_TYPE_HEAD_GPU",
    "SOWSMITH_TYPE_HEAD_GPU_CONF",
    "SOWSMITH_RUBRIC_GATE",
    "SOWSMITH_RUBRIC_GATE_CONF",
    "SOWSMITH_CONTRASTIVE_TYPE",
    "SOWSMITH_SPAN_GPU",
    "SOWSMITH_SPAN_AUGMENT",
    "SOWSMITH_SPAN_SKIP",
    "SOWSMITH_ENRICH_STORE_DEFLECT",
    "SOWSMITH_HEAD_REGISTRY_DIR",
    "SOWSMITH_NEURAL_EDGE_GATE",
    "SOWSMITH_MULTI_ENTITY_DISABLE",
    "SOWSMITH_RETRIEVAL_DISABLE",
    "SOWSMITH_TEACHER_CACHE",
    "SOWSMITH_WORKER_RETRAIN",
    "SOWSMITH_PDF_IMAGE_VISION",
)


def detect_runtime() -> str:
    explicit = os.environ.get("SOWSMITH_RUNTIME", "").strip()
    if explicit:
        return explicit
    # Heuristic: worker warm/job set ML_PROFILE=full; service stays lite
    if os.environ.get("SOWSMITH_ML_PROFILE", "").strip().lower() in ("full", "1", "true"):
        job = os.environ.get("CONTAINER_APP_JOB_NAME", "")
        if job:
            return "worker-job"
        return "worker-warm"
    return "service"


def _env_active() -> list[str]:
    active: list[str] = []
    for key in _ML_ENV_FLAGS:
        val = os.environ.get(key, "").strip()
        if not val:
            continue
        if key.endswith("_DISABLE") and val in ("1", "true", "yes", "on"):
            active.append(f"{key}=1")
        elif not key.endswith("_DISABLE") and val not in ("0", "false", "no", "off"):
            active.append(key)
    return sorted(active)


def _artifacts_loaded() -> dict[str, Any]:
    try:
        from app.learning.fetch_ml import artifact_self_check

        checks = artifact_self_check()
    except Exception:
        checks = {}
    rows = 0
    try:
        from app.core.training_log import get_training_log

        log = get_training_log()
        if log is not None:
            rows = log.count()
    except Exception:
        pass
    return {**checks, "training_log_rows": rows}


def _deflect_counts() -> dict[str, Any]:
    out: dict[str, Any] = {"typed_atom": {}, "span_gpu": {}, "enrich_store": {}}
    try:
        from app.core.typed_atom_classifier import get_last_deflect_stats

        stats = get_last_deflect_stats()
        if stats:
            out["typed_atom"] = {
                "store": stats.get("deflected", {}).get("store", 0),
                "student": stats.get("deflected", {}).get("student", 0),
                "type_head": stats.get("deflected", {}).get("type_head", 0),
                "type_head_gpu": stats.get("deflected", {}).get("type_head_gpu", 0),
                "contrastive": stats.get("deflected", {}).get("contrastive", 0),
                "rubric_gate": stats.get("deflected", {}).get("rubric_gate", 0),
                "deflected_total": stats.get("deflected_total", 0),
                "reached_llm": stats.get("reached_llm"),
            }
    except Exception:
        pass
    try:
        from app.core.decide import get_decide_stats

        ds = get_decide_stats()
        out["decide"] = {
            "store_hits": int(ds.get("store_hits", 0)),
            "llm_calls": int(ds.get("llm_calls", 0)),
            "llm_call_rate": round(float(ds.get("llm_call_rate", 0)), 4),
        }
    except Exception:
        pass
    return out


def _head_registry_served() -> bool:
    try:
        from app.learning.head_registry import get_head_registry

        reg = get_head_registry()
        if reg is None:
            return False
        relations = list(reg.relations())
        if any(reg.champion_version(r) for r in relations):
            return True
        # Registry loaded with trained inline heads (pre-promotion audit trail).
        return bool(relations) and os.environ.get("SOWSMITH_HEAD_REGISTRY_DIR", "").strip()
    except Exception:
        return False


def _ml_profile_phase() -> int:
    try:
        return int(os.environ.get("SOWSMITH_ML_PROFILE_PHASE", "1"))
    except ValueError:
        return 1


def _embed_stats() -> dict[str, Any]:
    try:
        from app.core.embedding_retrieval import get_last_embed_stats

        stats = get_last_embed_stats()
        if stats:
            return dict(stats)
    except Exception:
        pass
    return {}


def _worker_build_ids() -> dict[str, str]:
    return {
        "parser_os_sha": os.environ.get("PARSER_OS_SHA", "").strip() or "unknown",
        "worker_sha": os.environ.get("PARSER_OS_WORKER_SHA", "").strip() or "unknown",
    }


def build_compile_capabilities() -> dict[str, Any]:
    profile = os.environ.get("SOWSMITH_ML_PROFILE", "lite").strip().lower()
    if profile in ("1", "true", "yes", "on"):
        profile = "full"
    return {
        "runtime": detect_runtime(),
        "ml_profile": profile,
        "ml_profile_phase": _ml_profile_phase(),
        "build": _worker_build_ids(),
        "artifacts_loaded": _artifacts_loaded(),
        "env_active": _env_active(),
        "deflect_counts": _deflect_counts(),
        "embed_stats": _embed_stats(),
        "head_registry_served": _head_registry_served(),
        "skip_reasons": [],
    }


def build_version_capabilities() -> dict[str, Any]:
    """Snapshot for GET /v1/version — no compile-specific deflect counts."""
    cap = build_compile_capabilities()
    cap.pop("deflect_counts", None)
    return cap
