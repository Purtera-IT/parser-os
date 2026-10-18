"""A macro workbook is a workbook, and an inventory's site count comes from the
sheet that matches it (010264: 797 addresses, not the 65,902-row directory)."""
from pathlib import Path

from app.core.c3_sow import sheet_site_count
from app.parsers.registry import choose_parser


def test_an_xlsm_routes_to_the_spreadsheet_parser(tmp_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Address", "City", "State"])
    ws.append(["1 Main St", "Flagstaff", "AZ"])
    path = tmp_path / "inventory.xlsm"
    wb.save(path)
    parser, match, _ = choose_parser(path)
    assert parser is not None and match.parser_name == "xlsx"


def test_the_matching_sheet_gives_the_site_count() -> None:
    env = {"documents": [{"artifact_id": "a1", "filename": "9000_Clocks_Inventory_Review_9-2-26.xlsm"}],
           "atoms": [{"artifact_id": "a1", "structured": {"sheet": "DeviceList-08-06_9000"},
                      "text": "This sheet can supply 797 distinct site addresses, from 1,557 rows"},
                     {"artifact_id": "a1", "structured": {"sheet": "Cost Ctr List w Rank"},
                      "text": "This sheet can supply 65,902 distinct site addresses, from 114,783 rows"}]}
    assert sheet_site_count(env) == 797
    assert sheet_site_count({"atoms": []}) is None
