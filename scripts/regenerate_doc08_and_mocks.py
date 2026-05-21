"""Regenerate doc 08 (site roster) PDF with full column widths, AND
produce 15+ mock site-roster PDFs covering every shape we want the
parser to handle universally.

Outputs:
  - <out_dir>/08_site_roster_and_facilities_authoritative.pdf
      Fixed-width regeneration of the OPTBOT Marriott site roster.
  - <out_dir>/mock_site_rosters/site_roster_<NN>_<shape>.pdf
      15+ synthetic rosters across these shapes:

        01_standard_table         standard 6-column site_id-first table
        02_id_last_column         site_id is in the LAST column
        03_2_column               site_id + facility only
        04_8_column_full          site_id + everything (8 cols)
        05_underscore_ids         atl_hq_01 (underscore separator)
        06_numeric_ids            S001, S002 (no region prefix)
        07_store_ids              STORE-142, STORE-143
        08_bldg_ids               BLDG-1, BLDG-12
        09_international          TOR, LON, FRA region prefixes
        10_no_headers             headerless table (row-shape inference)
        11_explicit_declaration   "kind=physical_site for all rows" pre-prose
        12_split_address          address split into street/city/state cols
        13_single_site            1-row roster
        14_many_sites             18-row roster (stress test)
        15_extra_columns          mix of canonical + unknown cols
        16_phone_email            contact info instead of escort owner
        17_mixed_id_shapes        ATL-HQ-01 and S100 and BLDG-7 same table
        18_with_continuation      "TBD" rows + continuation block

Run:
  python scripts/regenerate_doc08_and_mocks.py <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_STYLES = getSampleStyleSheet()
_NORMAL = _STYLES["Normal"]
_TITLE = ParagraphStyle(
    "title",
    parent=_STYLES["Title"],
    fontSize=16,
    spaceAfter=8,
    alignment=0,
)
_H2 = ParagraphStyle(
    "h2",
    parent=_STYLES["Heading2"],
    fontSize=12,
    spaceBefore=8,
    spaceAfter=4,
)
_BODY = ParagraphStyle(
    "body",
    parent=_NORMAL,
    fontSize=9.5,
    leading=12,
    spaceAfter=6,
)
_CELL = ParagraphStyle(
    "cell",
    parent=_NORMAL,
    fontSize=8.5,
    leading=11,
)


def _build_table(headers: list[str], rows: list[list[str]], col_widths: list[float]) -> Table:
    data = [[Paragraph(h, _CELL) for h in headers]]
    for r in rows:
        data.append([Paragraph(c or "", _CELL) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    return t


def _build_doc(path: Path, story: list, landscape_mode: bool = True) -> None:
    pagesize = landscape(LETTER) if landscape_mode else LETTER
    doc = SimpleDocTemplate(
        str(path),
        pagesize=pagesize,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=path.stem,
    )
    doc.build(story)


# ── Doc 08 regeneration (Marriott OPTBOT Atlanta) ──────────────────


def regen_doc08(out_dir: Path) -> Path:
    out = out_dir / "08_site_roster_and_facilities_authoritative.pdf"
    story = [
        Paragraph("000087 - OPTBOT Atlanta Office Refresh  |  HubSpot 60355665326", _BODY),
        Paragraph("08 - Site Roster &amp; Facilities (Authoritative)", _TITLE),
        Paragraph(
            "Customer-supplied final supplement - closes site_roster &amp; after-hours gaps",
            _BODY,
        ),
        Paragraph(
            "This document is the authoritative site roster for the OPTBOT Atlanta Office Refresh program. "
            "It supersedes informal site references in discovery files where addresses or facility IDs were incomplete. "
            "PurTera and OPTBOT Facilities will use this roster for SOW site tables, escort scheduling, and cutover planning.",
            _BODY,
        ),
        Paragraph("1. Authoritative physical site roster (site_roster v5)", _H2),
        Paragraph(
            "The table below is the customer-approved site_roster. Each row is a physical_site with verified "
            "street address, primary MDF/IDF, and facility contact. kind=physical_site for all rows.",
            _BODY,
        ),
    ]

    headers = ["Site ID", "Facility name", "Street address", "MDF / IDF", "Access window", "Escort owner"]
    rows = [
        ["ATL-HQ-01",   "OPTBOT Atlanta HQ",        "1200 Peachtree St NE, Atlanta GA 30309",     "MDF-3A / IDF 2-7",  "Mon-Fri 07:00-18:00", "OPTBOT Facilities"],
        ["ATL-WEST-02", "OPTBOT West Campus",       "3100 Interstate N Pkwy, Atlanta GA 30339",   "MDF-W1 / IDF W2-3", "Mon-Fri 07:00-18:00", "OPTBOT Facilities"],
        ["ATL-AIR-03",  "OPTBOT Airport Logistics", "6000 N Terminal Pkwy, Atlanta GA 30320",     "MDF-A / IDF A1",     "Mon-Sat 06:00-22:00", "OPTBOT Security"],
        ["ATL-047-04",  "OPTBOT Brady Training",    "047 Brady Ave NW, Atlanta GA 30318",         "MDF-B / IDF B1-2",   "Mon-Fri 08:00-17:00", "OPTBOT Facilities"],
        ["ATL-CP-05",   "OPTBOT College Park Staging", "1850 Sullivan Rd, College Park GA 30337", "MDF-CP / staging",   "Mon-Fri 07:00-15:00", "OPTBOT Logistics"],
    ]
    # Tuned column widths so EVERY cell text fits — total ~10.0 inches
    # (landscape Letter content width is ~10.2 in).
    col_widths = [0.85 * inch, 1.55 * inch, 2.65 * inch, 1.55 * inch, 1.50 * inch, 1.55 * inch]
    story.append(_build_table(headers, rows, col_widths))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. After-hours, escort, and site-staff billing (final)", _H2))
    story.append(Paragraph(
        "Restricted work windows: Before 07:00, after 18:00 weekdays, and all weekends at "
        "ATL-HQ-01 and ATL-WEST-02 require 48-hour notice to OPTBOT Facilities.",
        _BODY,
    ))
    story.append(Paragraph(
        "Escort &amp; badge: OPTBOT Facilities provides escorts, badge sponsorship, and lift access at no charge to PurTera. "
        "PurTera bills only labor for after-hours work at the rates in document 09 (1.5x standard labor).",
        _BODY,
    ))
    story.append(Paragraph(
        "Custodial / unlock: OPTBOT custodial staff unlocks conference rooms; not in PurTera scope.",
        _BODY,
    ))
    story.append(Paragraph(
        "Building supervision: OPTBOT Security on-site supervisor required for ATL-AIR-03 after 20:00; "
        "scheduled by Facilities.",
        _BODY,
    ))

    story.append(Paragraph("3. Site cluster provenance (for OrbitBrief Site Reality)", _H2))
    story.append(Paragraph(
        "Source artifact for this roster: 08_site_roster_and_facilities_authoritative.pdf (this file). "
        "Cross-reference discovery: 03_site_surveys_and_requirements.docx, "
        "05_project_schedule_and_cutover_plan.xlsx. Each site_id above must publish as kind=physical_site "
        "with member evidence from this roster table and address fields.",
        _BODY,
    ))

    story.append(Paragraph("4. Cutover blackout dates", _H2))
    story.append(Paragraph(
        "No work permitted: 2026-11-26 through 2026-11-28 (Thanksgiving), "
        "2026-12-24 through 2027-01-02 (year-end freeze). ATL-AIR-03: no cutover during peak travel weeks "
        "without 14-day written waiver from OPTBOT Security.",
        _BODY,
    ))

    _build_doc(out, story, landscape_mode=True)
    return out


# ── 15+ mock variations ──────────────────────────────────────────


def mock_01_standard(out_dir: Path) -> Path:
    out = out_dir / "site_roster_01_standard.pdf"
    headers = ["Site ID", "Facility name", "Street address", "MDF/IDF", "Access window", "Escort owner"]
    rows = [
        ["NYC-HQ-01",   "Acme New York HQ",         "350 5th Ave, New York NY 10118",     "MDF-1",   "Mon-Fri 08:00-18:00", "Facilities"],
        ["NYC-WEST-02", "Acme Chelsea Office",      "75 9th Ave, New York NY 10011",      "MDF-2",   "Mon-Fri 08:00-18:00", "Facilities"],
        ["BOS-MAIN-03", "Acme Boston Office",       "100 Federal St, Boston MA 02110",    "MDF-3",   "Mon-Fri 08:00-17:00", "Facilities"],
    ]
    story = [
        Paragraph("Acme - Site Roster v1", _TITLE),
        _build_table(headers, rows, [0.9 * inch, 1.7 * inch, 3.0 * inch, 1.0 * inch, 1.5 * inch, 1.4 * inch]),
    ]
    _build_doc(out, story)
    return out


def mock_02_id_last_column(out_dir: Path) -> Path:
    out = out_dir / "site_roster_02_id_last_column.pdf"
    headers = ["Facility", "Address", "MDF", "Site ID"]
    rows = [
        ["DataHub West",   "100 Mission St, San Francisco CA",  "MDF-W", "SFO-WEST-01"],
        ["DataHub East",   "200 Park Ave, New York NY",         "MDF-E", "NYC-EAST-02"],
        ["DataHub Midwest","400 Wacker Dr, Chicago IL",         "MDF-M", "CHI-MID-03"],
    ]
    story = [
        Paragraph("DataHub - Site IDs (table)", _TITLE),
        _build_table(headers, rows, [1.7 * inch, 3.5 * inch, 1.0 * inch, 1.4 * inch]),
    ]
    _build_doc(out, story)
    return out


def mock_03_2_column(out_dir: Path) -> Path:
    out = out_dir / "site_roster_03_2_column.pdf"
    headers = ["Site ID", "Facility"]
    rows = [
        ["LON-HQ-01", "Acme London"],
        ["LON-WEST-02", "Acme Hammersmith"],
        ["LON-EAST-03", "Acme Canary Wharf"],
    ]
    story = [Paragraph("UK sites", _TITLE), _build_table(headers, rows, [1.5 * inch, 4.0 * inch])]
    _build_doc(out, story)
    return out


def mock_04_8_column_full(out_dir: Path) -> Path:
    out = out_dir / "site_roster_04_8_column_full.pdf"
    headers = ["Site ID", "Facility", "Address", "MDF/IDF", "Hours", "Escort", "Contact", "Phone"]
    rows = [
        ["TOR-HQ-01", "Acme Toronto", "100 King St W, Toronto ON", "MDF-1", "Mon-Fri 08:00-18:00", "Security", "J. Smith",  "416-555-0100"],
        ["TOR-NW-02", "Acme North York", "5 Park Home Ave, Toronto ON", "MDF-2", "Mon-Fri 07:00-19:00", "Facilities", "A. Patel",  "416-555-0101"],
    ]
    story = [
        Paragraph("Toronto sites (full schema)", _TITLE),
        _build_table(headers, rows, [0.8, 1.1, 1.8, 0.7, 1.4, 0.9, 0.9, 1.0]),
    ]
    # Convert to inches
    story[-1] = _build_table(headers, rows, [w * inch for w in [0.8, 1.1, 1.8, 0.7, 1.4, 0.9, 0.9, 1.0]])
    _build_doc(out, story)
    return out


def mock_05_underscore_ids(out_dir: Path) -> Path:
    out = out_dir / "site_roster_05_underscore_ids.pdf"
    headers = ["Site Code", "Name", "Address"]
    rows = [
        ["atl_hq_01",   "Atlanta HQ",       "1200 Peachtree St NE, Atlanta GA"],
        ["atl_west_02", "Atlanta West",     "3100 Interstate N Pkwy, Atlanta GA"],
        ["atl_air_03",  "Atlanta Airport",  "6000 N Terminal Pkwy, Atlanta GA"],
    ]
    story = [Paragraph("Atlanta sites (lowercase + underscores)", _TITLE),
             _build_table(headers, rows, [1.2 * inch, 2.0 * inch, 4.0 * inch])]
    _build_doc(out, story)
    return out


def mock_06_numeric_ids(out_dir: Path) -> Path:
    out = out_dir / "site_roster_06_numeric_ids.pdf"
    headers = ["Site ID", "Facility name", "Street address"]
    rows = [
        ["S001", "Acme Site 1", "100 Main St, Anytown USA"],
        ["S002", "Acme Site 2", "200 Main St, Anytown USA"],
        ["S003", "Acme Site 3", "300 Main St, Anytown USA"],
    ]
    story = [Paragraph("Numeric site IDs", _TITLE),
             _build_table(headers, rows, [1.0 * inch, 2.5 * inch, 3.5 * inch])]
    _build_doc(out, story)
    return out


def mock_07_store_ids(out_dir: Path) -> Path:
    out = out_dir / "site_roster_07_store_ids.pdf"
    headers = ["Store #", "Location", "Address"]
    rows = [
        ["STORE-142", "Cherry Creek", "3030 E 1st Ave, Denver CO"],
        ["STORE-143", "Park Meadows", "8401 Park Meadows Center Dr, Lone Tree CO"],
        ["STORE-144", "Flatiron Crossing", "1 Flatiron Crossing Dr, Broomfield CO"],
    ]
    story = [Paragraph("Retail chain stores", _TITLE),
             _build_table(headers, rows, [1.2 * inch, 2.0 * inch, 4.0 * inch])]
    _build_doc(out, story)
    return out


def mock_08_bldg_ids(out_dir: Path) -> Path:
    out = out_dir / "site_roster_08_bldg_ids.pdf"
    headers = ["Building", "Use", "Square footage"]
    rows = [
        ["BLDG-1",  "Office",     "120,000 sf"],
        ["BLDG-12", "Warehouse",  "85,000 sf"],
        ["BLDG-A2", "Datacenter", "20,000 sf"],
    ]
    story = [Paragraph("Campus building roster", _TITLE),
             _build_table(headers, rows, [1.5 * inch, 1.8 * inch, 2.0 * inch])]
    _build_doc(out, story)
    return out


def mock_09_international(out_dir: Path) -> Path:
    out = out_dir / "site_roster_09_international.pdf"
    headers = ["Site ID", "City / Country", "Address"]
    rows = [
        ["TOR-HQ-01", "Toronto, CA",  "100 King St W, Toronto ON M5X 1A1"],
        ["LON-HQ-01", "London, UK",   "1 St Mary Axe, London EC3A 8BF"],
        ["FRA-DC-01", "Frankfurt, DE","Hanauer Landstraße 296, 60314 Frankfurt"],
        ["SGP-OFFICE-01", "Singapore, SG", "1 Marina Boulevard, Singapore"],
    ]
    story = [Paragraph("International rollout sites", _TITLE),
             _build_table(headers, rows, [1.4 * inch, 1.8 * inch, 3.5 * inch])]
    _build_doc(out, story)
    return out


def mock_10_no_headers(out_dir: Path) -> Path:
    # First row has site-shaped tokens — no headers, parser must
    # infer columns positionally.
    out = out_dir / "site_roster_10_no_headers.pdf"
    headers = ["", "", ""]
    rows = [
        ["DAL-HQ-01", "Acme Dallas",  "100 Commerce St, Dallas TX"],
        ["DAL-N-02",  "Acme Frisco",  "200 Main St, Frisco TX"],
        ["DAL-S-03",  "Acme Plano",   "300 Plano Pkwy, Plano TX"],
    ]
    story = [Paragraph("(no headers; row-shape inference test)", _BODY),
             _build_table(headers, rows, [1.2 * inch, 2.0 * inch, 3.5 * inch])]
    _build_doc(out, story)
    return out


def mock_11_explicit_declaration(out_dir: Path) -> Path:
    out = out_dir / "site_roster_11_explicit_declaration.pdf"
    headers = ["Code", "Name", "Where"]
    rows = [
        ["HQ", "Acme HQ", "100 Main St, Anytown USA"],
        ["NW", "Northwest Office", "200 NW Ave, Anytown USA"],
        ["SE", "Southeast Office", "300 SE Blvd, Anytown USA"],
    ]
    story = [
        Paragraph(
            "kind=physical_site for all rows in the table that follows. "
            "Use these as the authoritative facility list for the rollout.",
            _BODY,
        ),
        _build_table(headers, rows, [1.0 * inch, 2.0 * inch, 3.5 * inch]),
    ]
    _build_doc(out, story)
    return out


def mock_12_split_address(out_dir: Path) -> Path:
    out = out_dir / "site_roster_12_split_address.pdf"
    headers = ["Site ID", "Facility", "Street", "City", "State", "Zip"]
    rows = [
        ["SEA-HQ-01",  "Acme Seattle",      "1200 5th Ave", "Seattle",  "WA", "98101"],
        ["SEA-BEL-02", "Acme Bellevue",     "10500 NE 8th St", "Bellevue", "WA", "98004"],
    ]
    story = [Paragraph("Split-address roster", _TITLE),
             _build_table(headers, rows, [1.0 * inch, 1.8 * inch, 2.0 * inch, 1.2 * inch, 0.6 * inch, 0.7 * inch])]
    _build_doc(out, story)
    return out


def mock_13_single_site(out_dir: Path) -> Path:
    out = out_dir / "site_roster_13_single_site.pdf"
    headers = ["Site ID", "Facility", "Address"]
    rows = [
        ["PHX-HQ-01", "Acme Phoenix HQ", "100 Camelback Rd, Phoenix AZ"],
    ]
    story = [Paragraph("Single-site project", _TITLE),
             _build_table(headers, rows, [1.2 * inch, 2.0 * inch, 4.0 * inch])]
    _build_doc(out, story)
    return out


def mock_14_many_sites(out_dir: Path) -> Path:
    out = out_dir / "site_roster_14_many_sites.pdf"
    headers = ["Site ID", "Facility", "Address"]
    rows = [
        ["DEN-HQ-01",   "Acme Denver HQ",       "1700 Lincoln St, Denver CO"],
        ["DEN-DTC-02",  "Acme DTC",             "5900 S Quebec St, Centennial CO"],
        ["DEN-AIR-03",  "Acme Airport",         "8500 Peña Blvd, Denver CO"],
        ["SLC-HQ-04",   "Acme Salt Lake",       "100 S Temple, SLC UT"],
        ["PHX-HQ-05",   "Acme Phoenix",         "100 Camelback Rd, Phoenix AZ"],
        ["LAS-HQ-06",   "Acme Las Vegas",       "1 Convention Center Dr, Las Vegas NV"],
        ["ABQ-HQ-07",   "Acme Albuquerque",     "100 Civic Plaza NW, Albuquerque NM"],
        ["BIL-HQ-08",   "Acme Billings",        "100 N Broadway, Billings MT"],
        ["BOI-HQ-09",   "Acme Boise",           "999 Main St, Boise ID"],
        ["PDX-HQ-10",   "Acme Portland",        "111 SW Columbia, Portland OR"],
        ["SEA-HQ-11",   "Acme Seattle",         "1200 5th Ave, Seattle WA"],
        ["SFO-HQ-12",   "Acme SF",              "1 Market St, San Francisco CA"],
        ["LAX-HQ-13",   "Acme LA",              "100 N Hope St, Los Angeles CA"],
        ["SAN-HQ-14",   "Acme San Diego",       "100 Broadway, San Diego CA"],
        ["MSP-HQ-15",   "Acme Minneapolis",     "100 S 5th St, Minneapolis MN"],
        ["MKE-HQ-16",   "Acme Milwaukee",       "100 E Wisconsin Ave, Milwaukee WI"],
        ["DTW-HQ-17",   "Acme Detroit",         "100 Renaissance Ctr, Detroit MI"],
        ["IND-HQ-18",   "Acme Indianapolis",    "100 Monument Cir, Indianapolis IN"],
    ]
    story = [Paragraph("West & Midwest rollout - 18 sites", _TITLE),
             _build_table(headers, rows, [1.2 * inch, 2.0 * inch, 4.0 * inch])]
    _build_doc(out, story)
    return out


def mock_15_extra_columns(out_dir: Path) -> Path:
    out = out_dir / "site_roster_15_extra_columns.pdf"
    headers = ["Site ID", "Facility", "Address", "Risk class", "Cleared", "Cost code", "Owner"]
    rows = [
        ["MIA-HQ-01", "Acme Miami",   "100 Biscayne Blvd, Miami FL",  "Tier 1", "Yes", "CC-001", "Jane Roe"],
        ["MIA-BCH-02","Acme Beach",   "1 Ocean Dr, Miami Beach FL",   "Tier 2", "No",  "CC-002", "John Doe"],
    ]
    story = [Paragraph("Roster with project-specific extra cols", _TITLE),
             _build_table(headers, rows, [0.9 * inch, 1.4 * inch, 2.3 * inch, 0.9 * inch, 0.7 * inch, 0.9 * inch, 1.2 * inch])]
    _build_doc(out, story)
    return out


def mock_16_phone_email(out_dir: Path) -> Path:
    out = out_dir / "site_roster_16_phone_email.pdf"
    headers = ["Site ID", "Facility", "Address", "Contact", "Phone", "Email"]
    rows = [
        ["AUS-HQ-01", "Acme Austin", "100 Congress Ave, Austin TX", "M. Lopez",  "512-555-0100", "mlopez@acme.test"],
        ["AUS-S-02",  "Acme South",  "200 S Lamar Blvd, Austin TX", "T. Nguyen", "512-555-0101", "tnguyen@acme.test"],
    ]
    story = [Paragraph("Roster with phone/email instead of escort", _TITLE),
             _build_table(headers, rows, [0.9 * inch, 1.5 * inch, 2.4 * inch, 1.0 * inch, 1.2 * inch, 1.7 * inch])]
    _build_doc(out, story)
    return out


def mock_17_mixed_id_shapes(out_dir: Path) -> Path:
    out = out_dir / "site_roster_17_mixed_id_shapes.pdf"
    headers = ["Site ID", "Facility", "Address"]
    rows = [
        ["ATL-HQ-01",    "OPTBOT Atlanta HQ",     "1200 Peachtree St NE"],
        ["S100",         "OPTBOT West Office",    "3100 Interstate N Pkwy"],
        ["BLDG-7",       "OPTBOT Airport Logistics", "6000 N Terminal Pkwy"],
        ["STORE-204",    "OPTBOT Brady Training", "047 Brady Ave NW"],
        ["LON-OFFICE-A", "OPTBOT London EMEA",    "1 St Mary Axe, London"],
    ]
    story = [Paragraph("Heterogeneous site-ID shapes in one roster", _TITLE),
             _build_table(headers, rows, [1.3 * inch, 2.5 * inch, 3.5 * inch])]
    _build_doc(out, story)
    return out


def mock_18_with_continuation(out_dir: Path) -> Path:
    out = out_dir / "site_roster_18_with_continuation.pdf"
    headers = ["Site ID", "Facility", "Address", "Status"]
    rows = [
        ["NYC-HQ-01", "Acme NYC",      "100 Park Ave, NY",         "Live"],
        ["NYC-W-02",  "Acme NYC West", "200 W 14th St, NY",        "Live"],
        ["TBD-1",     "Future site 1", "TBD - lease signing",      "Pending"],
        ["TBD-2",     "Future site 2", "TBD - LOI executed",       "Pending"],
        ["NYC-E-05",  "Acme NYC East", "300 E 42nd St, NY",        "Live"],
    ]
    story = [Paragraph("Roster with TBD continuation rows", _TITLE),
             _build_table(headers, rows, [1.0 * inch, 1.7 * inch, 4.0 * inch, 0.9 * inch])]
    _build_doc(out, story)
    return out


# ── Main ─────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/regenerate_doc08_and_mocks.py <out_dir>", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[regen] out_dir = {out_dir}")
    print(f"[regen] doc 08 -> {regen_doc08(out_dir).name}")

    mock_dir = out_dir / "mock_site_rosters"
    mock_dir.mkdir(parents=True, exist_ok=True)
    builders = [
        mock_01_standard, mock_02_id_last_column, mock_03_2_column,
        mock_04_8_column_full, mock_05_underscore_ids, mock_06_numeric_ids,
        mock_07_store_ids, mock_08_bldg_ids, mock_09_international,
        mock_10_no_headers, mock_11_explicit_declaration, mock_12_split_address,
        mock_13_single_site, mock_14_many_sites, mock_15_extra_columns,
        mock_16_phone_email, mock_17_mixed_id_shapes, mock_18_with_continuation,
    ]
    for b in builders:
        p = b(mock_dir)
        print(f"[regen] mock -> {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
