from __future__ import annotations

from pathlib import Path

from app.learning.rule_miner import mine_rule_suggestions


def _candidate_labels() -> list[dict]:
    return [
        {
            "id": "alias_1",
            "label_type": "entity_alias",
            "canonical_key": "ip_camera",
            "alias": "CCTV cam",
            "approved": True,
            "example_id": "packet_1",
        },
        {
            "id": "alias_2",
            "label_type": "entity_alias",
            "canonical_key": "ip_camera",
            "alias": "CCTV cam",
            "approved": True,
            "example_id": "packet_2",
        },
        {
            "id": "header_1",
            "label_type": "parser_header_alias",
            "target": "quantity",
            "header": "No. Units",
            "approved": True,
            "example_id": "sheet_a",
        },
        {
            "id": "header_2",
            "label_type": "parser_header_alias",
            "target": "quantity",
            "header": "No. Units",
            "approved": True,
            "example_id": "sheet_b",
        },
    ]


def test_repeated_alias_labels_create_domain_alias_suggestion() -> None:
    suggestions = mine_rule_suggestions(candidate_labels=_candidate_labels(), min_evidence=2)
    alias_rows = [row for row in suggestions if row.suggestion_type == "domain_alias"]
    assert alias_rows
    payload = alias_rows[0].proposed_change
    assert payload["device_aliases"]["ip_camera"] == ["CCTV cam"]


def test_repeated_header_labels_create_parser_header_alias_suggestion() -> None:
    suggestions = mine_rule_suggestions(candidate_labels=_candidate_labels(), min_evidence=2)
    header_rows = [row for row in suggestions if row.suggestion_type == "parser_header_alias"]
    assert header_rows
    assert header_rows[0].proposed_change["parser_header_aliases"]["quantity"] == ["No. Units"]


def test_contradictory_examples_lower_confidence() -> None:
    labels = _candidate_labels() + [
        {
            "id": "alias_3",
            "label_type": "entity_alias",
            "canonical_key": "ip_camera",
            "alias": "CCTV cam",
            "approved": False,
            "example_id": "packet_3",
        },
        {
            "id": "alias_4",
            "label_type": "entity_alias",
            "canonical_key": "ip_camera",
            "alias": "CCTV cam",
            "approved": False,
            "example_id": "packet_4",
        },
    ]
    suggestions = mine_rule_suggestions(candidate_labels=labels, min_evidence=2)
    alias_rows = [row for row in suggestions if row.suggestion_type == "domain_alias"]
    assert alias_rows
    assert alias_rows[0].confidence < 1.0
    assert alias_rows[0].negative_examples


def test_suggestions_require_human_approval() -> None:
    suggestions = mine_rule_suggestions(candidate_labels=_candidate_labels(), min_evidence=2)
    assert suggestions
    assert all(row.requires_human_approval is True for row in suggestions)


def test_no_suggestion_auto_modifies_domain_packs() -> None:
    pack_path = Path("app/domain/default_pack.yaml")
    before = pack_path.read_text(encoding="utf-8")
    _ = mine_rule_suggestions(candidate_labels=_candidate_labels(), min_evidence=2)
    after = pack_path.read_text(encoding="utf-8")
    assert before == after
