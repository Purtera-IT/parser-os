"""Regression tests for ``app.core.production_report`` + the
``parser-os report`` CLI.

The production report is the single hand-off package a tester gets back
from a compile run.  These tests lock in:

- The summary helpers handle the real CompileResult shape
- The executive REPORT.md renders a non-empty inputs table when an
  artifact_fingerprint is present
- The plain-English failure-analysis fires the right buckets
- Gold-compare auto-detect works when ``labels/gold_standard.json`` is
  present
- The whole bundle (REPORT.md + result.json + compare.json + ZIP)
  builds end-to-end against a tiny synthetic project
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import openpyxl
import pytest

from app.core.production_report import (
    _failure_analysis,
    _render_executive_report,
    _summarize_artifacts,
    _summarize_atoms,
    _summarize_packets,
    build_production_report,
)


# ─── Pure-function summary helpers ─────────────────────────────


class TestSummarizers:
    def test_summarize_atoms_aggregates_types_and_keys(self) -> None:
        atoms = [
            {
                "atom_type": "scope_item",
                "authority_class": "contractual_scope",
                "confidence": 0.9,
                "entity_keys": ["device:ip_camera", "site:building_a"],
                "receipts": [{"replay_status": "verified"}],
            },
            {
                "atom_type": "exclusion",
                "authority_class": "customer_current_authored",
                "confidence": 0.6,
                "entity_keys": ["device:ip_camera"],
                "receipts": [{"replay_status": "verified"}],
            },
            {
                "atom_type": "scope_item",
                "authority_class": "contractual_scope",
                "confidence": 0.85,
                "entity_keys": [],
                "receipts": [{"replay_status": "unsupported"}],
            },
        ]
        s = _summarize_atoms(atoms)
        assert s["count"] == 3
        assert s["by_type"] == {"scope_item": 2, "exclusion": 1}
        assert s["distinct_entity_keys"] == 2  # device:ip_camera, site:building_a
        assert s["entity_keys_by_prefix"] == {"device": 1, "site": 1}
        assert s["low_confidence_count"] == 1  # 0.6 < 0.7
        assert s["receipts"] == {"verified": 2, "unsupported": 1}
        # Average rounds to 3 decimals
        assert abs(s["avg_confidence"] - 0.783) < 0.01

    def test_summarize_packets_counts_contradictions(self) -> None:
        packets = [
            {"family": "scope_inclusion", "status": "active", "contradicting_atom_ids": []},
            {
                "family": "customer_override",
                "status": "needs_review",
                "contradicting_atom_ids": ["atm_1"],
            },
            {
                "family": "customer_override",
                "status": "needs_review",
                "contradicting_atom_ids": ["atm_2"],
            },
        ]
        s = _summarize_packets(packets)
        assert s["count"] == 3
        assert s["by_family"] == {"scope_inclusion": 1, "customer_override": 2}
        assert s["by_status"] == {"active": 1, "needs_review": 2}
        assert s["with_contradictions"] == 2

    def test_summarize_artifacts_pulls_from_fingerprints_and_routing(self) -> None:
        manifest = {
            "artifact_fingerprints": [
                {
                    "artifact_id": "art_1",
                    "filename": "artifacts/foo.pdf",
                    "artifact_type": "pdf",
                    "size_bytes": 12345,
                    "parser_name": "orbitbrief_pdf",
                    "parser_version": "v3",
                }
            ],
            "parser_routing": [
                {
                    "artifact_id": "art_1",
                    "chosen_parser": "orbitbrief_pdf",
                    "confidence": 0.95,
                    "reasons": ["pdf_extension"],
                    "cache_hit": False,
                }
            ],
        }
        atoms = [
            {"artifact_id": "art_1"},
            {"artifact_id": "art_1"},
            {"artifact_id": "art_1"},
        ]
        rows = _summarize_artifacts(manifest, atoms)
        assert len(rows) == 1
        row = rows[0]
        assert row["filename"] == "artifacts/foo.pdf"
        assert row["atom_count"] == 3
        assert row["routing_confidence"] == 0.95
        assert row["parser_name"] == "orbitbrief_pdf"


# ─── Failure analysis (plain-English bullet list) ─────────────


class TestFailureAnalysis:
    def _summary_template(self) -> dict[str, dict]:
        return {
            "atom_summary": {
                "count": 10,
                "low_confidence_count": 0,
                "receipts": {"verified": 10},
            },
            "packet_summary": {
                "count": 5,
                "with_contradictions": 0,
            },
        }

    def test_clean_compile_emits_success(self) -> None:
        s = self._summary_template()
        findings = _failure_analysis(
            result={"quality": {}},
            compare_report=None,
            atom_summary=s["atom_summary"],
            packet_summary=s["packet_summary"],
        )
        # Just one ✅ finding
        assert any("No anomalies" in f for f in findings)

    def test_failed_replay_surfaces_first(self) -> None:
        s = self._summary_template()
        s["atom_summary"]["receipts"] = {"verified": 8, "failed": 2}
        findings = _failure_analysis(
            result={"quality": {}},
            compare_report=None,
            atom_summary=s["atom_summary"],
            packet_summary=s["packet_summary"],
        )
        assert any("failed source-replay" in f for f in findings)

    def test_default_pack_routing_warns(self) -> None:
        s = self._summary_template()
        findings = _failure_analysis(
            result={"quality": {"pack_routing_source": "default"}},
            compare_report=None,
            atom_summary=s["atom_summary"],
            packet_summary=s["packet_summary"],
        )
        assert any("default_pack" in f.lower() for f in findings)

    def test_low_entity_resolution_rate_warns(self) -> None:
        s = self._summary_template()
        findings = _failure_analysis(
            result={"quality": {"entity_resolution_rate": 0.30}},
            compare_report=None,
            atom_summary=s["atom_summary"],
            packet_summary=s["packet_summary"],
        )
        assert any("Entity-resolution rate" in f for f in findings)

    def test_gold_compare_failure_surfaced(self) -> None:
        s = self._summary_template()
        compare = {
            "overall": {"pass": 3, "total_checked": 5, "pass_fraction": 0.6},
            "metrics": {
                "atom_count": {"verdict": "pass"},
                "packet_families": {
                    "verdict": "fail",
                    "missing": ["compliance_clause", "action_item"],
                },
                "compliance_atoms": {
                    "verdict": "fail",
                    "actual": 5,
                    "expected_min": 15,
                },
            },
        }
        findings = _failure_analysis(
            result={"quality": {}},
            compare_report=compare,
            atom_summary=s["atom_summary"],
            packet_summary=s["packet_summary"],
        )
        joined = " ".join(findings)
        assert "Gold compare" in joined
        assert "packet_families" in joined
        assert "compliance_atoms" in joined


# ─── Report rendering ──────────────────────────────────────────


class TestRenderReport:
    def test_inputs_table_present_when_fingerprints_exist(
        self, tmp_path: Path
    ) -> None:
        result = {
            "compile_id": "cmp_test",
            "project_id": "test_proj",
            "manifest": {
                "input_signature": "in_sig",
                "output_signature": "out_sig",
                "artifact_fingerprints": [
                    {
                        "artifact_id": "art_1",
                        "filename": "artifacts/sample.pdf",
                        "artifact_type": "pdf",
                        "size_bytes": 12345,
                        "parser_name": "orbitbrief_pdf",
                        "parser_version": "v3",
                    }
                ],
                "parser_routing": [
                    {
                        "artifact_id": "art_1",
                        "chosen_parser": "orbitbrief_pdf",
                        "confidence": 0.95,
                        "reasons": ["pdf_extension"],
                    }
                ],
            },
            "atoms": [{"artifact_id": "art_1", "atom_type": "scope_item"}],
            "packets": [],
            "trace": {
                "stages": [
                    {
                        "stage_name": "parse_artifacts",
                        "duration_ms": 100,
                        "input_count": 1,
                        "output_count": 1,
                    }
                ],
                "total_duration_ms": 100,
            },
            "quality": {},
        }
        atom_summary = _summarize_atoms(result["atoms"])
        packet_summary = _summarize_packets(result["packets"])
        artifact_summary = _summarize_artifacts(result["manifest"], result["atoms"])
        md = _render_executive_report(
            result=result,
            compare_report=None,
            atom_summary=atom_summary,
            packet_summary=packet_summary,
            artifact_summary=artifact_summary,
            project_dir=tmp_path,
            out_dir=tmp_path,
        )
        # Header + inputs table + atom histogram all present
        assert "# Parser-OS production report" in md
        assert "`artifacts/sample.pdf`" in md
        assert "in_sig" in md
        assert "out_sig" in md

    def test_no_gold_section_when_compare_absent(self, tmp_path: Path) -> None:
        result = {
            "compile_id": "cmp_test",
            "project_id": "p",
            "manifest": {},
            "atoms": [],
            "packets": [],
            "trace": {},
            "quality": {},
        }
        md = _render_executive_report(
            result=result,
            compare_report=None,
            atom_summary=_summarize_atoms([]),
            packet_summary=_summarize_packets([]),
            artifact_summary=[],
            project_dir=tmp_path,
            out_dir=tmp_path,
        )
        # Gold-compare line absent; success header used instead
        assert "Gold-compare verdict" not in md
        assert "Compile completed" in md

    def test_gold_emoji_reflects_pass_fraction(self, tmp_path: Path) -> None:
        empty = {
            "compile_id": "c", "project_id": "p", "manifest": {}, "atoms": [],
            "packets": [], "trace": {}, "quality": {},
        }
        for pf, expected_emoji in [(0.95, "🟢"), (0.60, "🟡"), (0.30, "🔴")]:
            compare = {
                "overall": {"pass": 1, "total_checked": 1, "pass_fraction": pf},
                "metrics": {},
            }
            md = _render_executive_report(
                result=empty,
                compare_report=compare,
                atom_summary=_summarize_atoms([]),
                packet_summary=_summarize_packets([]),
                artifact_summary=[],
                project_dir=tmp_path,
                out_dir=tmp_path,
            )
            assert expected_emoji in md, f"pf={pf} expected {expected_emoji}"


# ─── End-to-end: build_production_report on a synthetic project ─


class TestBuildProductionReport:
    def _make_synthetic_project(self, tmp_path: Path) -> Path:
        project = tmp_path / "synth_project"
        artifacts = project / "artifacts"
        labels = project / "labels"
        artifacts.mkdir(parents=True)
        labels.mkdir(parents=True)
        # One tiny XLSX so xlsx_parser produces atoms deterministically.
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["ID", "Site", "Device", "Quantity", "Part Number"])
        ws.append([1, "Building A", "IP Camera", 12, "CW9166I-B"])
        ws.append([2, "Building B", "IP Camera", 8, "CW9166I-B"])
        wb.save(artifacts / "schedule.xlsx")
        # Tiny gold file so compare.json gets written.
        (labels / "gold_standard.json").write_text(
            json.dumps(
                {
                    "case_id": "synth",
                    "expected_min_atom_count": 1,
                    "expected_min_packet_count": 1,
                }
            ),
            encoding="utf-8",
        )
        return project

    def test_full_bundle_writes_all_artifacts(self, tmp_path: Path) -> None:
        project = self._make_synthetic_project(tmp_path)
        out_dir = tmp_path / "report_out"
        summary = build_production_report(
            project_dir=project,
            out_dir=out_dir,
            no_cache=True,
            skip_orbitbrief=True,  # keep test fast; envelope tested elsewhere
            zip_bundle=True,
        )
        # Required outputs
        assert (out_dir / "REPORT.md").is_file()
        assert (out_dir / "result.json").is_file()
        assert (out_dir / "compare.json").is_file()  # gold present
        # ZIP exists and contains REPORT.md + result.json + compare.json
        zip_path = Path(summary["zip_path"])
        assert zip_path.is_file()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert any(n.endswith("REPORT.md") for n in names)
            assert any(n.endswith("result.json") for n in names)
            assert any(n.endswith("compare.json") for n in names)
        # Summary fields populated
        assert summary["compile_id"]
        assert summary["input_signature"]
        assert summary["output_signature"]
        assert summary["atom_count"] >= 1
        assert summary["gold_pass_fraction"] is not None

    def test_no_gold_skips_compare_json(self, tmp_path: Path) -> None:
        project = self._make_synthetic_project(tmp_path)
        # Remove the gold so compare.json shouldn't be produced
        (project / "labels" / "gold_standard.json").unlink()
        out_dir = tmp_path / "report_out_no_gold"
        summary = build_production_report(
            project_dir=project,
            out_dir=out_dir,
            no_cache=True,
            skip_orbitbrief=True,
            zip_bundle=False,
        )
        assert (out_dir / "REPORT.md").is_file()
        assert not (out_dir / "compare.json").is_file()
        assert summary["compare_json"] is None
        assert summary["gold_pass_fraction"] is None

    def test_zip_bundle_disabled_when_no_zip_passed(self, tmp_path: Path) -> None:
        project = self._make_synthetic_project(tmp_path)
        out_dir = tmp_path / "report_out_no_zip"
        summary = build_production_report(
            project_dir=project,
            out_dir=out_dir,
            no_cache=True,
            skip_orbitbrief=True,
            zip_bundle=False,
        )
        assert summary["zip_path"] is None
        assert (out_dir / "REPORT.md").is_file()
