"""A per-place table (City | Zip | size | hours per site) is a site roster (010358's survey workbook)."""
from app.parsers.site_roster_extractor import extract_site_roster, find_header_row, looks_like_site_roster

HEADER = ["City", "Zip", "Region", "Tier", "Floors", "Bldg SqFt", "WLAN Survey Hrs", "Total Hrs", "Extended Cost"]
ROWS = [
    ["HOUSTON", "77002", "Houston Core", "Medium", 3, 42648, 3.9, 9.25, 1000],
    ["HOUSTON", "77015", "Houston Core", "Medium", 2, 51000, 4.3, 9.73, 1100],
    ["HOUSTON", "77015", "Houston Core", "Small", 1, 14500, 2.0, 7.13, 800],
    ["BEAUMONT", "77701", "Beaumont", "Complex", 18, 272300, 26.0, 33.1, 3500],
]


def _dicts(rows):
    return [{HEADER[i]: ("" if v is None else str(v)) for i, v in enumerate(r)} for r in rows]


def test_header_is_found_under_a_banner_and_an_assumptions_block():
    sheet = [["Houston Survey - Hours and Cost by Site"], ["WLAN sqft per 8-hr day", 100000], [], HEADER, *ROWS]
    assert find_header_row(sheet) == 3


def test_every_row_is_a_site_even_two_in_one_zip():
    rows = _dicts(ROWS)
    assert looks_like_site_roster(columns=HEADER, rows=rows, surrounding_text="Site Detail")
    out = extract_site_roster(columns=HEADER, rows=rows, surrounding_text="Site Detail")
    assert [(r.city, r.zip) for r in out] == [("HOUSTON", "77002"), ("HOUSTON", "77015"), ("HOUSTON", "77015"), ("BEAUMONT", "77701")]


def test_one_place_repeated_is_not_a_roster():
    one = _dicts([["HOUSTON", "77002", "Core", "M", 1, 1000, 1, 1, 1]] * 4)
    assert not looks_like_site_roster(columns=HEADER, rows=one, surrounding_text="Rates")


def test_city_and_zip_without_a_size_is_not_enough_on_its_own():
    header = ["City", "Zip", "Invoice", "Amount", "Extended Cost"]
    rows = [{"City": c, "Zip": z, "Invoice": "1", "Amount": "5", "Extended Cost": "5"} for c, z in (("A", "1"), ("B", "2"), ("C", "3"))]
    assert not looks_like_site_roster(columns=header, rows=rows, surrounding_text="Billing")
