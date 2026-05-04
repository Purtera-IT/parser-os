"""Regression tests for Week-4 production-readiness features.

Covers:
- ``app.core.gold_compare`` — gold-vs-compiled metric verdicts (P3.3)
- ``app.core.quality_metrics`` — quality-score computation (P3.4)
- ``app.domain.project_config`` — project.yaml schema + scaffold (P3.1)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.gold_compare import compare_to_gold


# ─── Gold compare ─────────────────────────────────────────────────────
class TestGoldCompare:
    def test_pass_when_atom_count_meets_threshold(self) -> None:
        gold = {"case_id": "TEST", "expected_min_atom_count": 5}
        compiled = {"atoms": [{"id": str(i)} for i in range(10)]}
        report = compare_to_gold(gold=gold, compiled=compiled)
        assert report["metrics"]["atom_count"]["verdict"] == "pass"
        assert report["overall"]["pass"] >= 1

    def test_fail_when_atom_count_below_threshold(self) -> None:
        gold = {"expected_min_atom_count": 100}
        compiled = {"atoms": [{"id": str(i)} for i in range(10)]}
        report = compare_to_gold(gold=gold, compiled=compiled)
        assert report["metrics"]["atom_count"]["verdict"] == "fail"
        assert report["metrics"]["atom_count"]["actual"] == 10
        assert report["metrics"]["atom_count"]["expected_min"] == 100

    def test_skipped_when_threshold_absent(self) -> None:
        gold = {}
        compiled = {"atoms": [], "packets": [], "edges": []}
        report = compare_to_gold(gold=gold, compiled=compiled)
        # Every check should skip — overall pass_fraction defaults to 0
        assert report["overall"]["total_checked"] == 0
        assert report["overall"]["skipped"] >= 5

    def test_quantity_conflict_count(self) -> None:
        gold = {"expected_quantity_conflict_edges_within_artifact": 6}
        compiled = {
            "atoms": [],
            "packets": [],
            "edges": [
                {"metadata": {"edge_family": "part_number_quantity_conflict"}}
                for _ in range(6)
            ],
        }
        report = compare_to_gold(gold=gold, compiled=compiled)
        assert report["metrics"]["quantity_conflict_edges"]["verdict"] == "pass"

    def test_distinct_sites(self) -> None:
        gold = {"expected_min_distinct_sites": 3}
        compiled = {
            "atoms": [
                {"entity_keys": ["site:a", "device:cam"]},
                {"entity_keys": ["site:b"]},
                {"entity_keys": ["site:c", "vendor:cisco"]},
                {"entity_keys": ["site:a"]},  # duplicate — must dedup
            ],
            "packets": [],
            "edges": [],
        }
        report = compare_to_gold(gold=gold, compiled=compiled)
        assert report["metrics"]["distinct_sites"]["verdict"] == "pass"
        assert report["metrics"]["distinct_sites"]["actual"] == 3

    def test_entity_keys_must_include(self) -> None:
        gold = {
            "expected_entity_keys_must_include": [
                "vendor:cisco", "site:school_a", "device:ip_camera",
            ],
        }
        compiled = {
            "atoms": [
                {"entity_keys": ["vendor:cisco", "device:ip_camera"]},
                {"entity_keys": ["site:school_b"]},
            ],
            "packets": [],
            "edges": [],
        }
        report = compare_to_gold(gold=gold, compiled=compiled)
        check = report["metrics"]["entity_keys_must_include"]
        assert check["verdict"] == "fail"
        assert "site:school_a" in check["missing_sample"]

    def test_threshold_with_plus_suffix(self) -> None:
        gold = {"expected_min_packet_count": "12+"}
        compiled = {"atoms": [], "packets": [{"id": str(i)} for i in range(15)], "edges": []}
        report = compare_to_gold(gold=gold, compiled=compiled)
        assert report["metrics"]["packet_count"]["verdict"] == "pass"
        assert report["metrics"]["packet_count"]["expected_min"] == 12


# ─── Quality metrics ──────────────────────────────────────────────────
class TestQualityMetrics:
    def _build_result(self) -> "object":
        from app.core.schemas import (
            ArtifactType,
            AtomType,
            AuthorityClass,
            CompileManifest,
            CompileResult,
            EntityRecord,
            EvidenceAtom,
            EvidenceEdge,
            EdgeType,
            EvidencePacket,
            PacketFamily,
            PacketStatus,
            ReviewStatus,
            SourceRef,
        )

        ref = SourceRef(
            id="src1",
            artifact_id="art1",
            artifact_type=ArtifactType.pdf,
            filename="x.pdf",
            locator={"page": 1},
            extraction_method="t",
            parser_version="v1",
        )

        def mk_atom(aid: str, keys: list[str]) -> EvidenceAtom:
            return EvidenceAtom(
                id=aid,
                project_id="t",
                artifact_id="art1",
                atom_type=AtomType.scope_item,
                raw_text=f"text {aid}",
                normalized_text=f"text {aid}",
                value={},
                entity_keys=keys,
                source_refs=[ref],
                receipts=[],
                authority_class=AuthorityClass.contractual_scope,
                confidence=0.9,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version="orbitbrief_pdf_v3",
            )

        atoms = [
            mk_atom("a1", ["device:ip_camera", "site:school"]),
            mk_atom("a2", ["device:unknown"]),  # unknown — not real
            mk_atom("a3", ["vendor:cisco"]),
            mk_atom("a4", []),  # no keys
        ]
        edges = [
            EvidenceEdge(
                id="edge1",
                project_id="t",
                from_atom_id="a1",
                to_atom_id="a3",
                edge_type=EdgeType.supports,
                reason="r",
                confidence=0.9,
                metadata={"edge_family": "value_support"},
            ),
            EvidenceEdge(
                id="edge2",
                project_id="t",
                from_atom_id="a1",
                to_atom_id="a3",
                edge_type=EdgeType.contradicts,
                reason="r",
                confidence=0.95,
                metadata={"edge_family": "part_number_quantity_conflict"},
            ),
        ]
        packets = [
            EvidencePacket(
                id="pkt1",
                project_id="t",
                family=PacketFamily.scope_inclusion,
                anchor_type="device",
                anchor_key="device:ip_camera",
                governing_atom_ids=["a1"],
                supporting_atom_ids=["a1"],
                contradicting_atom_ids=[],
                related_edge_ids=[],
                confidence=0.9,
                status=PacketStatus.active,
                review_flags=[],
                anchor_signature={
                    "anchor_type": "device",
                    "canonical_key": "device:ip_camera",
                    "entity_keys": ["device:ip_camera"],
                    "normalized_topic": "device:ip_camera",
                    "hash": "abc123",
                },
                reason="r",
            ),
            EvidencePacket(
                id="pkt2",
                project_id="t",
                family=PacketFamily.scope_inclusion,
                anchor_type="device",
                anchor_key="device:unknown",
                governing_atom_ids=["a2"],
                supporting_atom_ids=["a2"],
                contradicting_atom_ids=[],
                related_edge_ids=[],
                confidence=0.5,
                status=PacketStatus.needs_review,
                review_flags=["unknown_anchor"],
                anchor_signature={
                    "anchor_type": "device",
                    "canonical_key": "device:unknown",
                    "entity_keys": [],
                    "normalized_topic": "device:unknown",
                    "hash": "def456",
                },
                reason="r",
            ),
        ]
        manifest = CompileManifest(
            compile_id="cmp1",
            project_id="t",
            started_at="2026-01-01T00:00:00Z",
            deterministic_seed="seed",
            input_signature="sig_in",
            domain_pack_id="security_camera",
            domain_pack_version="2.0.0",
            parser_routing=[
                {
                    "artifact_id": "art1",
                    "filename": "x.pdf",
                    "chosen_parser": "orbitbrief_pdf",
                    "parser_version": "orbitbrief_pdf_v3",
                    "confidence": 0.95,
                    "reasons": ["pdf_extension"],
                },
            ],
        )
        return CompileResult(
            project_id="t",
            atoms=atoms,
            entities=[],
            edges=edges,
            packets=packets,
            warnings=[],
            manifest=manifest,
        )

    def test_quality_basics(self) -> None:
        from app.core.quality_metrics import compute_quality
        result = self._build_result()
        q = compute_quality(result, pack_routing_source="source_notes", pack_routing_confidence=0.9)
        # 3 of 4 atoms have a real entity_key (a1, a3 do; a2 has only :unknown; a4 empty)
        assert q.entity_resolution_rate == round(2 / 4, 4)
        # 1 of 2 packets has a real anchor (pkt1 yes, pkt2 has unknown_anchor)
        assert q.packet_specificity == 0.5
        assert q.quantity_conflict_edge_count == 1
        assert q.pack_id == "security_camera"
        assert q.pack_routing_source == "source_notes"
        assert q.parser_routing_confidence_avg == 0.95
        assert q.parser_atom_yield_rate == 1.0  # the orbitbrief_pdf parser produced ≥1 atom
        assert q.atoms_per_artifact == 4.0


# ─── ProjectConfig ────────────────────────────────────────────────────
class TestProjectConfig:
    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        from app.domain.project_config import load_project_config
        assert load_project_config(tmp_path) is None

    def test_load_yaml(self, tmp_path: Path) -> None:
        from app.domain.project_config import load_project_config
        (tmp_path / "project.yaml").write_text(textwrap.dedent("""
            domain_pack: security_camera_pack
            service_line: security_camera
            customer: acme
            parserignore_extra:
              - "*.draft.pdf"
        """).strip(), encoding="utf-8")
        cfg = load_project_config(tmp_path)
        assert cfg is not None
        assert cfg.domain_pack == "security_camera_pack"
        assert cfg.service_line == "security_camera"
        assert cfg.customer == "acme"
        assert cfg.parserignore_extra == ["*.draft.pdf"]

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        from app.domain.project_config import load_project_config
        (tmp_path / "project.yaml").write_text(
            "service_line: wireless\n"
            "future_unknown_field: 42\n",
            encoding="utf-8",
        )
        cfg = load_project_config(tmp_path)
        assert cfg is not None
        assert cfg.service_line == "wireless"

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        from app.domain.project_config import load_project_config
        (tmp_path / "project.yaml").write_text(":\n  - not a mapping\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_project_config(tmp_path)

    def test_write_default_template(self, tmp_path: Path) -> None:
        from app.domain.project_config import write_default_project_yaml
        target = write_default_project_yaml(tmp_path)
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        assert "domain_pack" in text
        assert "service_line" in text

    def test_write_default_skips_existing(self, tmp_path: Path) -> None:
        from app.domain.project_config import write_default_project_yaml
        existing = tmp_path / "project.yaml"
        existing.write_text("custom: yes\n", encoding="utf-8")
        write_default_project_yaml(tmp_path)
        assert existing.read_text(encoding="utf-8") == "custom: yes\n"
