"""A tab-separated .txt is a table (000123's 161-store BevMo roster)."""
from pathlib import Path

from app.parsers.registry import choose_parser


def test_a_tab_separated_text_file_yields_one_site_per_row(tmp_path: Path) -> None:
    rows = ["Store\tLocationName\tAddress\tCity\tState\tZip Code"] + [
        f"{i}\tShop_{i}\t{i}0 Main Street\tTown{i}\tCA\t9000{i}" for i in range(1, 6)]
    path = tmp_path / "locations.txt"
    path.write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
    parser, match, _ = choose_parser(path)
    assert match.parser_name == "xlsx"
    sites = [a for a in parser.parse(path) if str(a.atom_type).endswith("physical_site")]
    assert len(sites) == 5 and sites[0].value["city"] == "Town1"


def test_prose_with_a_stray_tab_stays_text(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("Hello team,\nPlease see the plan.\tThanks\nBest\n", encoding="utf-8")
    _, match, _ = choose_parser(path)
    assert match.parser_name != "xlsx"
