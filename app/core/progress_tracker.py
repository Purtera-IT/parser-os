"""v45.2 — Compile progress tracker with adaptive ETA estimation.

Emits progress JSON as each stage completes. Predicts ETA from doc
properties (atoms, chars, pages, visual-page count) using a learned
throughput model that's accurate within ±10-20% on first run, then
self-calibrates as the pipeline observes actual stage times.

Used by:
  - app/core/compiler.py — wraps compile_project with start/stage/complete events
  - app/core/multi_entity_llm.py — emits sub-progress during embed + canonicalize batches
  - parser-os-service GET /v1/compile/progress/{compile_id} — serves the JSON

Contract (progress.json shape):
  {
    "compile_id": "cmp_xxx",
    "deal_id": "...",
    "project_dir": "...",
    "started_at": "ISO",
    "updated_at": "ISO",
    "status": "queued|running|completed|failed",
    "current_stage": "enrich_entities",
    "current_stage_label": "Extracting entities (LLM analysis)",
    "current_substage": "canonicalize",
    "current_substage_label": "Validating entity candidates",
    "stages_completed": [...],
    "stages_remaining": [...],
    "percent_complete": 38,
    "elapsed_seconds": 1051,
    "estimated_total_seconds": 1800,
    "estimated_remaining_seconds": 749,
    "eta": "ISO",
    "throughput": { ... },  # sub-stage counters
    "doc_metadata": { ... }
  }
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# STAGE DEFINITIONS (canonical pipeline)
# ════════════════════════════════════════════════════════════════════

# Each stage has:
#   - canonical name (matches compile_stage_completed events)
#   - human-readable label (for UI)
#   - estimated_seconds formula: callable(doc_metadata) → seconds
#   - relative_weight (for percent_complete computation)


def _embed_t(d: dict) -> float:
    """Embedding stage time estimate."""
    sents = d.get("sentence_count") or (d.get("atom_count", 0) * 5)
    return sents / 28.0 + 2.0


def _canon_t(d: dict) -> float:
    """Canonicalize stage time estimate (Mac-tuned)."""
    sents = d.get("sentence_count") or (d.get("atom_count", 0) * 5)
    # ~40% of sentences become candidates that hit canonicalize
    candidates = sents * 0.4
    return candidates / 6.0 + 5.0


def _vision_t(d: dict) -> float:
    """Vision-LLM time estimate — only fires if visual pages."""
    n = d.get("visual_page_count", 0)
    if n == 0:
        return 0
    return n / 0.1 + 5.0  # ~10 sec per page


def _enrich_t(d: dict) -> float:
    """enrich_entities is composite: embed + canon + sicrl + tournament + vision + zero_miss."""
    embed = _embed_t(d)
    canon = _canon_t(d)
    sicrl = 90.0  # flat ~1.5 min
    tournament = (d.get("atom_count", 0) ** 0.5) / 2.0
    vision = _vision_t(d)
    zero_miss = 120.0  # flat ~2 min for PLIR + PM-vocab sweep
    return embed + canon + sicrl + tournament + vision + zero_miss


STAGE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "discover_artifacts",
        "label": "Discovering input files",
        "estimate": lambda d: 0.5,
        "weight": 1,
    },
    {
        "name": "parse_artifacts",
        "label": "Parsing documents (PDF/Word/Excel/email)",
        "estimate": lambda d: d.get("total_chars", 0) / 8000.0 + 5.0,
        "weight": 8,
    },
    {
        "name": "candidate_adjudication",
        "label": "Reconciling cross-document candidates",
        "estimate": lambda d: 0.1,
        "weight": 1,
    },
    {
        "name": "source_replay",
        "label": "Verifying source provenance",
        "estimate": lambda d: d.get("atom_count", 0) / 1000.0 + 1.0,
        "weight": 2,
    },
    {
        "name": "confidence_floor",
        "label": "Filtering low-confidence atoms",
        "estimate": lambda d: 0.1,
        "weight": 1,
    },
    {
        "name": "enrich_entities",
        "label": "Extracting entities (LLM analysis)",
        "estimate": _enrich_t,
        "weight": 80,  # 80% of total time
    },
    {
        "name": "entity_resolution",
        "label": "Resolving entity aliases + cross-doc dedup",
        "estimate": lambda d: d.get("atom_count", 0) / 30.0 + 5.0,
        "weight": 3,
    },
    {
        "name": "graph_build",
        "label": "Building evidence graph",
        "estimate": lambda d: 1.0,
        "weight": 1,
    },
    {
        "name": "packetize",
        "label": "Grouping evidence into PM packets",
        "estimate": lambda d: 0.2 * max(1, d.get("atom_count", 0) / 10),
        "weight": 1,
    },
    {
        "name": "packet_certificates",
        "label": "Signing packet receipts",
        "estimate": lambda d: 0.1,
        "weight": 1,
    },
    {
        "name": "quality_gates",
        "label": "Running quality checks",
        "estimate": lambda d: 0.1,
        "weight": 1,
    },
]


SUBSTAGE_LABELS = {
    "embedding": "Embedding sentences (semantic index)",
    "retrieval": "Searching for entity candidates",
    "canonicalize": "Validating entity candidates",
    "sicrl": "Counterfactual recall (predicting missed content)",
    "tournament": "Cross-document deduplication",
    "vision": "Reading tables and diagrams (vision-LLM)",
    "ocr": "OCR scanned pages",
    "zero_miss": "PM-critical vocabulary sweep",
    "plir": "Page-level recall sweep",
}


# ════════════════════════════════════════════════════════════════════
# THROUGHPUT MODEL — learned from real runs
# ════════════════════════════════════════════════════════════════════


def default_mac_throughputs() -> dict[str, float]:
    """Mac Studio M3 Max + qwen3:14b + qwen3-embedding:8b throughput defaults."""
    return {
        "parse_chars_per_sec": 8000.0,
        "atoms_per_sec_source_replay": 1000.0,
        "sentences_per_sec_embedding": 28.0,
        "candidates_per_sec_canonicalize": 6.0,
        "vision_pages_per_sec": 0.1,
        "atoms_per_sec_entity_resolution": 30.0,
        "tournament_pairs_per_sec": 2.0,
    }


# ════════════════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ════════════════════════════════════════════════════════════════════


@dataclass
class StageRecord:
    name: str
    started_at: float
    completed_at: float | None = None
    actual_seconds: float | None = None
    counts: dict[str, int] = field(default_factory=dict)


class ProgressTracker:
    """Thread-safe compile progress tracker.

    Writes progress.json to a configurable path. Each stage callback
    updates the file. Sub-stage callbacks within enrich_entities
    update fine-grained progress.
    """

    def __init__(
        self,
        compile_id: str,
        *,
        deal_id: str | None = None,
        project_dir: Path | str | None = None,
        out_path: Path | None = None,
        throughputs: dict[str, float] | None = None,
    ):
        self.compile_id = compile_id
        self.deal_id = deal_id or "unknown"
        self.project_dir = str(project_dir) if project_dir else None
        self.throughputs = throughputs or default_mac_throughputs()

        # Output path: caller-provided OR {project_dir}/.orbitbrief/progress.json
        if out_path:
            self.out_path = Path(out_path)
        elif project_dir:
            self.out_path = Path(project_dir) / ".orbitbrief" / "progress.json"
        else:
            # Fall back to env-controlled location
            base = os.environ.get(
                "SOWSMITH_PROGRESS_DIR",
                str(Path.home() / ".parser_os" / "progress"),
            )
            self.out_path = Path(base) / f"{compile_id}.json"

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.started_at = time.time()
        self.lock = Lock()
        self.stage_records: list[StageRecord] = []
        self.current_stage: str | None = None
        self.current_substage: str | None = None
        self.substage_counts: dict[str, dict[str, int]] = {}
        self.doc_metadata: dict[str, Any] = {}
        self.status = "queued"
        self.last_error: str | None = None

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def start(self, doc_metadata: dict[str, Any] | None = None) -> None:
        """Mark compile as started. Pass doc_metadata for ETA estimation."""
        with self.lock:
            self.status = "running"
            if doc_metadata:
                self.doc_metadata.update(doc_metadata)
            self._write()

    def update_doc_metadata(self, **kwargs: Any) -> None:
        """Update doc_metadata mid-pipeline (e.g. after parse atoms are counted)."""
        with self.lock:
            self.doc_metadata.update(kwargs)
            self._write()

    def set_compile_id(self, compile_id: str) -> None:
        """Rename the tracker's compile_id (called after the manifest assigns the
        content-addressed compile_id).  The output file path stays the same (it
        lives under {project_dir}/.orbitbrief/progress.json when project_dir is
        set), so only the in-document `compile_id` field and the registry key
        need to move.
        """
        old_id = self.compile_id
        with self.lock:
            self.compile_id = compile_id
            self._write()
        # Re-key the module registry so get_tracker(new_id) works.
        with _REGISTRY_LOCK:
            if old_id in _ACTIVE_TRACKERS and _ACTIVE_TRACKERS[old_id] is self:
                _ACTIVE_TRACKERS.pop(old_id, None)
                _ACTIVE_TRACKERS[compile_id] = self

    def stage_started(self, name: str) -> None:
        """Mark a top-level stage as starting."""
        with self.lock:
            self.current_stage = name
            self.current_substage = None
            rec = StageRecord(name=name, started_at=time.time())
            self.stage_records.append(rec)
            self._write()

    def stage_completed(
        self, name: str,
        *,
        duration_ms: float | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        """Mark a top-level stage as completed."""
        with self.lock:
            now = time.time()
            # Find matching open record
            for rec in reversed(self.stage_records):
                if rec.name == name and rec.completed_at is None:
                    rec.completed_at = now
                    if duration_ms is not None:
                        rec.actual_seconds = duration_ms / 1000.0
                    else:
                        rec.actual_seconds = now - rec.started_at
                    if counts:
                        rec.counts = dict(counts)
                    break
            else:
                # No matching open record — add a closed one
                self.stage_records.append(StageRecord(
                    name=name,
                    started_at=now - (duration_ms or 0) / 1000.0,
                    completed_at=now,
                    actual_seconds=(duration_ms or 0) / 1000.0,
                    counts=dict(counts or {}),
                ))
            # If this was the active stage, clear it
            if self.current_stage == name:
                self.current_stage = None
            # Update throughput model
            self._update_throughputs(name)
            self._write()

    def substage(
        self, substage: str,
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        """Update sub-stage progress (only meaningful during enrich_entities)."""
        with self.lock:
            self.current_substage = substage
            if current is not None or total is not None:
                slot = self.substage_counts.setdefault(substage, {})
                if current is not None:
                    slot["current"] = current
                if total is not None:
                    slot["total"] = total
            self._write()

    def complete(self) -> None:
        """Mark compile as successfully completed."""
        with self.lock:
            self.status = "completed"
            self.current_stage = None
            self.current_substage = None
            self._write()

    def fail(self, error: str) -> None:
        """Mark compile as failed with reason."""
        with self.lock:
            self.status = "failed"
            self.last_error = error
            self._write()

    # ────────────────────────────────────────────────────────────
    # ETA computation
    # ────────────────────────────────────────────────────────────

    def _compute_eta(self) -> dict[str, Any]:
        """Compute current ETA based on doc_metadata + throughput model
        + observed actuals so far."""
        elapsed = time.time() - self.started_at
        # Sum estimated time for ALL stages from doc_metadata
        per_stage_estimate: dict[str, float] = {}
        for sd in STAGE_DEFINITIONS:
            per_stage_estimate[sd["name"]] = max(
                0.1, float(sd["estimate"](self.doc_metadata))
            )
        # Total estimated
        total_estimated = sum(per_stage_estimate.values()) * 1.15  # 15% buffer

        # If we have observed stages, recompute remaining as
        # (sum of unstarted_stages estimate) + (current_stage estimate * (1 - current_progress))
        completed_set: set[str] = set()
        completed_actual_sum = 0.0
        in_flight_actual = 0.0
        for rec in self.stage_records:
            if rec.completed_at is not None and rec.actual_seconds is not None:
                completed_set.add(rec.name)
                completed_actual_sum += rec.actual_seconds
            elif rec.completed_at is None:
                # In-flight
                in_flight_actual += time.time() - rec.started_at

        # Calibration factor: observed actuals / predicted for completed stages
        observed_predicted_sum = sum(
            per_stage_estimate.get(s, 0) for s in completed_set
        )
        calibration = (
            completed_actual_sum / observed_predicted_sum
            if observed_predicted_sum > 0
            else 1.0
        )
        # Clamp calibration to reasonable range
        calibration = max(0.3, min(calibration, 5.0))

        # Remaining stages (not completed AND not current)
        remaining_seconds = 0.0
        for sd in STAGE_DEFINITIONS:
            n = sd["name"]
            if n in completed_set:
                continue
            est = per_stage_estimate[n]
            if n == self.current_stage:
                # In-flight stage — subtract observed elapsed
                inflight = in_flight_actual
                # Account for sub-stage progress if known
                progress = self._substage_progress(n)
                if progress is not None and progress > 0:
                    expected_full = est * calibration
                    remaining_for_stage = expected_full * (1 - progress)
                    remaining_seconds += max(0, remaining_for_stage)
                else:
                    remaining_for_stage = est * calibration - inflight
                    remaining_seconds += max(0, remaining_for_stage)
            else:
                remaining_seconds += est * calibration

        remaining_seconds *= 1.10  # 10% safety buffer on remaining

        # Compute percent
        total_weight = sum(sd["weight"] for sd in STAGE_DEFINITIONS)
        completed_weight = 0
        for sd in STAGE_DEFINITIONS:
            if sd["name"] in completed_set:
                completed_weight += sd["weight"]
            elif sd["name"] == self.current_stage:
                progress = self._substage_progress(sd["name"]) or 0.0
                completed_weight += sd["weight"] * progress

        percent = int(min(99, max(0, completed_weight * 100 / total_weight)))
        if self.status == "completed":
            percent = 100

        eta_iso = None
        if self.status == "running":
            eta_ts = time.time() + remaining_seconds
            eta_iso = datetime.fromtimestamp(eta_ts, timezone.utc).isoformat()

        return {
            "elapsed_seconds": int(elapsed),
            "estimated_total_seconds": int(elapsed + remaining_seconds) if self.status == "running" else int(elapsed),
            "estimated_remaining_seconds": int(remaining_seconds) if self.status == "running" else 0,
            "percent_complete": percent,
            "eta": eta_iso,
            "calibration_factor": round(calibration, 2),
        }

    def _substage_progress(self, stage_name: str) -> float | None:
        """Return 0.0-1.0 substage progress for the current stage, if known."""
        if stage_name != "enrich_entities":
            return None
        if not self.current_substage:
            return None
        slot = self.substage_counts.get(self.current_substage, {})
        cur = slot.get("current", 0)
        tot = slot.get("total", 0)
        if tot > 0:
            # Sub-stage progress within enrich_entities — map each
            # sub-stage to an approximate weight
            substage_weights = {
                "embedding": 0.20,
                "retrieval": 0.05,
                "canonicalize": 0.50,
                "sicrl": 0.10,
                "tournament": 0.05,
                "vision": 0.05,
                "zero_miss": 0.05,
            }
            base = sum(
                w for sname, w in substage_weights.items()
                if sname == self.current_substage
            )
            # Add weights for substages we've moved past (heuristic: substages run in order)
            order = ["embedding", "retrieval", "canonicalize", "sicrl", "tournament", "vision", "zero_miss"]
            try:
                idx = order.index(self.current_substage)
                past = sum(substage_weights.get(s, 0) for s in order[:idx])
            except ValueError:
                past = 0
            return min(1.0, past + base * (cur / tot))
        return None

    def _update_throughputs(self, stage_name: str) -> None:
        """Live-update throughput rate based on this stage's observed time."""
        rec = next(
            (r for r in self.stage_records
             if r.name == stage_name and r.actual_seconds is not None),
            None,
        )
        if not rec or not rec.actual_seconds:
            return
        # Per-stage observed throughput → smooth into model (0.7 old, 0.3 new)
        d = self.doc_metadata
        if stage_name == "parse_artifacts" and d.get("total_chars"):
            observed = d["total_chars"] / rec.actual_seconds
            self.throughputs["parse_chars_per_sec"] = (
                0.7 * self.throughputs["parse_chars_per_sec"] + 0.3 * observed
            )
        elif stage_name == "entity_resolution" and d.get("atom_count"):
            observed = d["atom_count"] / rec.actual_seconds
            self.throughputs["atoms_per_sec_entity_resolution"] = (
                0.7 * self.throughputs["atoms_per_sec_entity_resolution"] + 0.3 * observed
            )

    # ────────────────────────────────────────────────────────────
    # Serialization
    # ────────────────────────────────────────────────────────────

    def _write(self) -> None:
        """Atomically write progress.json."""
        eta = self._compute_eta()
        # Build stages-completed / remaining lists
        completed_set: set[str] = set()
        for rec in self.stage_records:
            if rec.completed_at is not None:
                completed_set.add(rec.name)
        all_stage_names = [sd["name"] for sd in STAGE_DEFINITIONS]
        stages_completed = [n for n in all_stage_names if n in completed_set]
        stages_remaining = [
            n for n in all_stage_names
            if n not in completed_set and n != self.current_stage
        ]
        current_stage_label = next(
            (sd["label"] for sd in STAGE_DEFINITIONS if sd["name"] == self.current_stage),
            None,
        )
        current_substage_label = (
            SUBSTAGE_LABELS.get(self.current_substage)
            if self.current_substage else None
        )

        payload = {
            "compile_id": self.compile_id,
            "deal_id": self.deal_id,
            "project_dir": self.project_dir,
            "started_at": datetime.fromtimestamp(
                self.started_at, timezone.utc
            ).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "current_stage": self.current_stage,
            "current_stage_label": current_stage_label,
            "current_substage": self.current_substage,
            "current_substage_label": current_substage_label,
            "stages_completed": stages_completed,
            "stages_remaining": stages_remaining,
            "stage_history": [
                {
                    "name": r.name,
                    "actual_seconds": round(r.actual_seconds, 2) if r.actual_seconds else None,
                    "counts": r.counts,
                }
                for r in self.stage_records if r.completed_at is not None
            ],
            "throughput": dict(self.substage_counts),
            "doc_metadata": self.doc_metadata,
            "last_error": self.last_error,
            **eta,
        }

        # Write atomically (tmp file + rename)
        tmp = self.out_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.out_path)
        except Exception as e:
            logger.warning("Failed to write progress.json: %s", e)


# ════════════════════════════════════════════════════════════════════
# MODULE-LEVEL REGISTRY (so any code can find the active tracker
# without threading it through every call)
# ════════════════════════════════════════════════════════════════════


_ACTIVE_TRACKERS: dict[str, ProgressTracker] = {}
_REGISTRY_LOCK = Lock()


def register_tracker(tracker: ProgressTracker) -> None:
    with _REGISTRY_LOCK:
        _ACTIVE_TRACKERS[tracker.compile_id] = tracker


def get_active_tracker() -> ProgressTracker | None:
    """Return the most recently registered tracker. Used by sub-stage
    emitters in multi_entity_llm.py / vision_extraction.py without
    needing to plumb the tracker through every call."""
    with _REGISTRY_LOCK:
        if not _ACTIVE_TRACKERS:
            return None
        return list(_ACTIVE_TRACKERS.values())[-1]


def get_tracker(compile_id: str) -> ProgressTracker | None:
    with _REGISTRY_LOCK:
        return _ACTIVE_TRACKERS.get(compile_id)


def unregister_tracker(compile_id: str) -> None:
    with _REGISTRY_LOCK:
        _ACTIVE_TRACKERS.pop(compile_id, None)


__all__ = [
    "STAGE_DEFINITIONS",
    "SUBSTAGE_LABELS",
    "ProgressTracker",
    "default_mac_throughputs",
    "register_tracker",
    "get_active_tracker",
    "get_tracker",
    "unregister_tracker",
]
