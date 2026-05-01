from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from app.core.compiler import compile_project
from app.parsers.registry import choose_parser
from app.parsers.spreadsheet_route_signals import resolve_quote_vs_xlsx_tie


def _save_xlsx(path: Path, headers: list[str], data_row: list[str] | None = None) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    if data_row is not None:
        ws.append(data_row)
    wb.save(path)


def test_site_list_routes_to_xlsx_parser(demo_project: Path) -> None:
    parser, match, _ = choose_parser(demo_project / "site_list.xlsx", domain_pack=None)
    assert parser is not None
    assert match.parser_name == "xlsx"


def test_site_list_device_qty_routes_to_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "site_list.xlsx"
    _save_xlsx(path, ["Site", "Device", "Qty"], ["Main", "Switch-A", "12"])
    parser, match, _ = choose_parser(path, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "xlsx"


def test_vendor_quote_routes_to_quote_parser(demo_project: Path) -> None:
    parser, match, _ = choose_parser(demo_project / "vendor_quote.xlsx", domain_pack=None)
    assert parser is not None
    assert match.parser_name == "quote"
    assert match.confidence >= 0.8


def test_vendor_quote_line_item_qty_included_routes_to_quote(tmp_path: Path) -> None:
    path = tmp_path / "vendor_quote.xlsx"
    _save_xlsx(path, ["Line Item", "Qty", "Included"], ["Cable run", "10", "Yes"])
    parser, match, _ = choose_parser(path, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "quote"


def test_drop_schedule_wide_material_routes_to_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "drop_schedule.xlsx"
    _save_xlsx(
        path,
        ["Plate ID", "Room", "RJ45", "Cat6 UTP", "Cat6 STP"],
        ["P-01", "101", "2", "1", "0"],
    )
    parser, match, _ = choose_parser(path, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "xlsx"


def test_bom_part_description_qty_routes_to_quote(tmp_path: Path) -> None:
    path = tmp_path / "material_bom.xlsx"
    _save_xlsx(path, ["Part Number", "Description", "Qty"], ["PN-1", "Widget", "5"])
    parser, match, _ = choose_parser(path, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "quote"


def test_site_list_still_xlsx_when_ancestor_dir_contains_vendor_token(tmp_path: Path) -> None:
    """Pytest tmp dirs may include 'vendor' (e.g. test names); routing must use path tail only."""
    deep = (
        tmp_path
        / "pytest-of-x"
        / "pytest-0"
        / "test_ip_camera_vendor_mismatch0"
        / "repo"
        / "tests"
        / "fixtures"
        / "demo_project"
    )
    deep.mkdir(parents=True)
    path = deep / "site_list.xlsx"
    _save_xlsx(path, ["Site", "Device", "Qty"], ["Main", "IP Camera", "1"])
    parser, match, _ = choose_parser(path, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "xlsx"


def test_ambiguous_path_logs_tie_decision_reasons(tmp_path: Path) -> None:
    path = tmp_path / "scope_matrix_vendor_estimate.xlsx"
    _save_xlsx(path, ["Site", "Device", "Qty"], ["A", "X", "1"])
    choice, reasons = resolve_quote_vs_xlsx_tie(path)
    joined = " ".join(reasons).lower()
    assert "tie" in joined
    assert choice in ("quote", "xlsx")


def test_customer_email_routes_email(tmp_path: Path) -> None:
    artifact = tmp_path / "customer_email.txt"
    artifact.write_text("From: customer@example.com\nSent: Monday\nSubject: Scope\nNeed exclude west wing", encoding="utf-8")
    parser, match, _ = choose_parser(artifact, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "email"


def test_kickoff_transcript_routes_transcript(tmp_path: Path) -> None:
    artifact = tmp_path / "kickoff_transcript.txt"
    artifact.write_text("Decisions:\n- Main campus first\nAction Items:\nAlex: schedule install", encoding="utf-8")
    parser, match, _ = choose_parser(artifact, domain_pack=None)
    assert parser is not None
    assert match.parser_name == "transcript"


def test_random_txt_produces_warning_no_crash(tmp_path: Path) -> None:
    artifact = tmp_path / "random.txt"
    artifact.write_text("just filler words with no structured signals", encoding="utf-8")
    result = compile_project(tmp_path, allow_errors=True)
    assert any("No parser matched artifact" in warning for warning in result.warnings)


def test_compile_trace_includes_parser_routing(tmp_path: Path) -> None:
    artifact = tmp_path / "customer_email.txt"
    artifact.write_text("From: customer@example.com\nSent: Monday\nSubject: Scope", encoding="utf-8")
    result = compile_project(tmp_path, allow_errors=True)
    assert result.trace is not None
    assert result.trace.parser_routing
    routing = result.trace.parser_routing[0]
    assert routing["filename"] == "customer_email.txt"
    assert routing["chosen_parser"] == "email"
