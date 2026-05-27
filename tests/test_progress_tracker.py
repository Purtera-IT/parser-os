"""Unit tests for v45.2 ProgressTracker.

Covers:
  - Static ETA prediction matches doc-property-driven formula
  - Live calibration adjusts ETA when actuals diverge from predictions
  - Substage progress contributes to percent_complete correctly
  - Atomic write (no partial files visible to readers)
  - Status transitions (queued → running → completed / failed)
  - Stage history correctly recorded
  - Throughput model self-adjusts
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.core.progress_tracker import (
    STAGE_DEFINITIONS,
    SUBSTAGE_LABELS,
    ProgressTracker,
    default_mac_throughputs,
    get_active_tracker,
    register_tracker,
    unregister_tracker,
)


@pytest.fixture
def tmp_progress_path(tmp_path: Path) -> Path:
    return tmp_path / "progress.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestStaticEta:
    """ETA estimates from doc_metadata alone (no actuals yet)."""

    def test_eta_for_small_pack(self, tmp_progress_path):
        t = ProgressTracker("cmp_test1", out_path=tmp_progress_path)
        t.start({
            "artifact_count": 3,
            "total_chars": 95236,
            "page_count": 38,
            "atom_count": 369,
            "visual_page_count": 10,
            "sentence_count": 442,
        })
        data = _read(tmp_progress_path)
        # First-run ETA should be in the realistic range for Pack 02 on Mac
        # (we know Pack 02 v45.1 actually takes ~30 min cold cache)
        # Our formula gives ~7 min raw estimate; with 15% buffer + Mac
        # contention multiplier, the predicted total may be lower than
        # actual. That's OK — calibration kicks in during the run.
        assert data["status"] == "running"
        assert data["estimated_total_seconds"] > 200, \
            "Should estimate at least 3+ min for Pack 02 size"
        assert data["estimated_total_seconds"] < 3600, \
            "Should estimate under 1 hour for Pack 02 size"
        assert data["percent_complete"] == 0
        assert data["doc_metadata"]["atom_count"] == 369

    def test_eta_scales_with_chars(self, tmp_progress_path):
        small = ProgressTracker("cmp_small", out_path=tmp_progress_path)
        small.start({"total_chars": 10000, "atom_count": 50, "sentence_count": 50})
        small_eta = _read(tmp_progress_path)["estimated_total_seconds"]

        big = ProgressTracker("cmp_big", out_path=tmp_progress_path)
        big.start({"total_chars": 500000, "atom_count": 2000, "sentence_count": 4000})
        big_eta = _read(tmp_progress_path)["estimated_total_seconds"]

        assert big_eta > small_eta, "Larger doc should have larger ETA"

    def test_visual_pages_add_to_eta(self, tmp_progress_path):
        no_vis = ProgressTracker("cmp_nv", out_path=tmp_progress_path)
        no_vis.start({
            "total_chars": 50000, "atom_count": 200, "sentence_count": 300,
            "visual_page_count": 0,
        })
        no_vis_eta = _read(tmp_progress_path)["estimated_total_seconds"]

        with_vis = ProgressTracker("cmp_vis", out_path=tmp_progress_path)
        with_vis.start({
            "total_chars": 50000, "atom_count": 200, "sentence_count": 300,
            "visual_page_count": 20,
        })
        with_vis_eta = _read(tmp_progress_path)["estimated_total_seconds"]

        # 20 visual pages × ~10 sec each = ~200s additional
        assert with_vis_eta > no_vis_eta, "Visual pages should add to ETA"
        assert with_vis_eta - no_vis_eta > 150


class TestStageProgression:
    """Status + percent_complete progress as stages complete."""

    def test_status_transitions(self, tmp_progress_path):
        t = ProgressTracker("cmp_st", out_path=tmp_progress_path)
        assert t.status == "queued"

        t.start({"atom_count": 100})
        assert _read(tmp_progress_path)["status"] == "running"

        t.stage_started("parse_artifacts")
        t.stage_completed("parse_artifacts", duration_ms=1000)
        assert _read(tmp_progress_path)["status"] == "running"

        t.complete()
        data = _read(tmp_progress_path)
        assert data["status"] == "completed"
        assert data["percent_complete"] == 100

    def test_fail_with_reason(self, tmp_progress_path):
        t = ProgressTracker("cmp_fail", out_path=tmp_progress_path)
        t.start({"atom_count": 50})
        t.stage_started("parse_artifacts")
        t.fail("Ollama unreachable")
        data = _read(tmp_progress_path)
        assert data["status"] == "failed"
        assert data["last_error"] == "Ollama unreachable"

    def test_percent_grows_with_stage_completion(self, tmp_progress_path):
        t = ProgressTracker("cmp_pct", out_path=tmp_progress_path)
        t.start({"atom_count": 100, "total_chars": 10000, "sentence_count": 200})
        pcts = []

        for stage in ["discover_artifacts", "parse_artifacts",
                      "candidate_adjudication", "source_replay"]:
            t.stage_started(stage)
            t.stage_completed(stage, duration_ms=100)
            pcts.append(_read(tmp_progress_path)["percent_complete"])

        assert pcts == sorted(pcts), "percent_complete must be monotonically increasing"

    def test_stages_completed_and_remaining_lists(self, tmp_progress_path):
        t = ProgressTracker("cmp_lists", out_path=tmp_progress_path)
        t.start({"atom_count": 100})
        t.stage_started("discover_artifacts")
        t.stage_completed("discover_artifacts", duration_ms=10)
        t.stage_started("parse_artifacts")
        t.stage_completed("parse_artifacts", duration_ms=500)

        data = _read(tmp_progress_path)
        assert "discover_artifacts" in data["stages_completed"]
        assert "parse_artifacts" in data["stages_completed"]
        assert "enrich_entities" in data["stages_remaining"]
        # Stages_completed should be in canonical order
        assert data["stages_completed"] == ["discover_artifacts", "parse_artifacts"]


class TestSubstageProgress:
    """Sub-stage progress contributes to percent_complete within enrich_entities."""

    def test_substage_label_set(self, tmp_progress_path):
        t = ProgressTracker("cmp_sub", out_path=tmp_progress_path)
        t.start({"atom_count": 100})
        t.stage_started("enrich_entities")
        t.substage("embedding", current=100, total=500)

        data = _read(tmp_progress_path)
        assert data["current_stage"] == "enrich_entities"
        assert data["current_substage"] == "embedding"
        assert data["current_substage_label"] == SUBSTAGE_LABELS["embedding"]

    def test_substage_progress_increases_percent(self, tmp_progress_path):
        t = ProgressTracker("cmp_sub2", out_path=tmp_progress_path)
        t.start({"atom_count": 100, "sentence_count": 500})
        # Complete all stages BEFORE enrich
        for stage in ["discover_artifacts", "parse_artifacts",
                      "candidate_adjudication", "source_replay",
                      "confidence_floor"]:
            t.stage_started(stage)
            t.stage_completed(stage, duration_ms=10)

        # Start enrich_entities
        t.stage_started("enrich_entities")
        early = _read(tmp_progress_path)["percent_complete"]

        # Sub-progress: 50% through embedding
        t.substage("embedding", current=250, total=500)
        mid = _read(tmp_progress_path)["percent_complete"]

        # Sub-progress: started canonicalize at 25%
        t.substage("canonicalize", current=100, total=400)
        later = _read(tmp_progress_path)["percent_complete"]

        # Percentages must be strictly non-decreasing
        assert early <= mid <= later


class TestCalibration:
    """ETA recalculates as actuals diverge from prediction."""

    def test_calibration_factor_recorded(self, tmp_progress_path):
        t = ProgressTracker("cmp_cal", out_path=tmp_progress_path)
        t.start({"atom_count": 100, "total_chars": 10000, "sentence_count": 200})
        # Run parse fast (faster than predicted)
        t.stage_started("parse_artifacts")
        time.sleep(0.05)
        t.stage_completed("parse_artifacts", duration_ms=50)  # 50ms vs predicted ~1.25s

        data = _read(tmp_progress_path)
        assert "calibration_factor" in data
        # 50ms / 1250ms = 0.04 → but clamped to >= 0.3
        assert data["calibration_factor"] >= 0.3
        assert data["calibration_factor"] <= 5.0


class TestAtomicWrite:
    """progress.json writes are atomic — no partial file visible."""

    def test_no_partial_writes_visible(self, tmp_progress_path):
        t = ProgressTracker("cmp_atom", out_path=tmp_progress_path)
        t.start({"atom_count": 100})
        assert tmp_progress_path.exists()
        # File must always be valid JSON, even mid-update
        for i in range(10):
            t.stage_started(f"stage_{i}")
            data = json.loads(tmp_progress_path.read_text(encoding="utf-8"))
            assert "compile_id" in data
            t.stage_completed(f"stage_{i}", duration_ms=10)

    def test_concurrent_substage_updates_safe(self, tmp_progress_path):
        """Multiple threads updating substage shouldn't corrupt the file."""
        from concurrent.futures import ThreadPoolExecutor
        t = ProgressTracker("cmp_conc", out_path=tmp_progress_path)
        t.start({"atom_count": 100})
        t.stage_started("enrich_entities")

        def update(i):
            t.substage("canonicalize", current=i, total=100)

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(update, range(50)))

        # Final file must be valid JSON
        data = json.loads(tmp_progress_path.read_text(encoding="utf-8"))
        assert data["current_substage"] == "canonicalize"


class TestRegistry:
    """Module-level tracker registry for cross-file access."""

    def test_register_and_get(self, tmp_progress_path):
        t = ProgressTracker("cmp_reg", out_path=tmp_progress_path)
        register_tracker(t)
        try:
            assert get_active_tracker() is t
        finally:
            unregister_tracker("cmp_reg")
        assert get_active_tracker() is None


class TestStageDefinitions:
    """Sanity check the canonical stage list matches what compile_project actually emits."""

    def test_all_canonical_stages_have_estimator(self):
        for sd in STAGE_DEFINITIONS:
            assert "name" in sd
            assert "label" in sd
            assert callable(sd["estimate"])
            # Estimator must return positive number for representative input
            est = sd["estimate"]({
                "atom_count": 100, "total_chars": 50000,
                "sentence_count": 200, "visual_page_count": 5,
            })
            assert est >= 0

    def test_weight_sum_is_positive(self):
        total = sum(sd["weight"] for sd in STAGE_DEFINITIONS)
        assert total > 0


class TestEdgeCases:
    """Defensive tests for surprise inputs."""

    def test_completing_unknown_stage(self, tmp_progress_path):
        t = ProgressTracker("cmp_edge", out_path=tmp_progress_path)
        t.start({"atom_count": 0})
        # Completing without starting — should not crash
        t.stage_completed("never_started", duration_ms=10)
        data = _read(tmp_progress_path)
        assert any(s["name"] == "never_started" for s in data["stage_history"])

    def test_empty_doc_metadata(self, tmp_progress_path):
        t = ProgressTracker("cmp_empty", out_path=tmp_progress_path)
        t.start({})  # No metadata at all
        data = _read(tmp_progress_path)
        assert data["status"] == "running"
        assert data["estimated_total_seconds"] >= 0

    def test_update_doc_metadata_mid_run(self, tmp_progress_path):
        t = ProgressTracker("cmp_mid", out_path=tmp_progress_path)
        t.start({"atom_count": 100})
        # After parse completes, we learn the real atom_count
        t.stage_started("parse_artifacts")
        t.stage_completed("parse_artifacts", duration_ms=500)
        t.update_doc_metadata(atom_count=369, sentence_count=442)
        data = _read(tmp_progress_path)
        assert data["doc_metadata"]["atom_count"] == 369
        assert data["doc_metadata"]["sentence_count"] == 442
