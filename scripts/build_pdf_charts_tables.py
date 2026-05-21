"""PDFs that mix charts and complex tables — the shapes a managed-services
financial / scope report uses.

  ct_a_bar_chart_plus_table.pdf       Bar chart + adjacent quarterly spend table
  ct_b_pie_chart_breakdown.pdf        Pie chart + breakdown table
  ct_c_rotated_table.pdf              Landscape page with wide multi-col BOM
  ct_d_nested_subheaders.pdf          Table with multi-level column groups
  ct_e_image_only_table.pdf           Image of a table (no text layer)
  ct_f_chart_and_sla_matrix.pdf       Chart + SLA tier matrix combined
  ct_g_color_coded_table.pdf          Status colors (RAG: red/amber/green) in cells
  ct_h_milestone_timeline_table.pdf   Date columns Q1-Q4 with milestone rows
  ct_i_split_table_across_pages.pdf   Table that spans 2 pages
  ct_j_summary_then_detail.pdf        Page 1 summary table + Page 2 detail BOM
  ct_k_charts_with_callouts.pdf       Chart with annotation callouts on the chart
  ct_l_pricing_with_subtotals.pdf     Pricing table with subtotals and grand total
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String


_S = getSampleStyleSheet()
_T = ParagraphStyle("t", parent=_S["Title"], fontSize=14, spaceAfter=10)
_H2 = ParagraphStyle("h2", parent=_S["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
_BODY = ParagraphStyle("b", parent=_S["Normal"], fontSize=10, leading=13, spaceAfter=6)
_CELL = ParagraphStyle("c", parent=_S["Normal"], fontSize=9)


def _tbl(headers, rows, widths_in, header_color="#dde6f0"):
    data = [[Paragraph(h, _CELL) for h in headers]] + [
        [Paragraph(c or "", _CELL) if isinstance(c, str) else Paragraph(str(c) if c is not None else "", _CELL) for c in r]
        for r in rows
    ]
    t = Table(data, colWidths=[w * inch for w in widths_in], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _bar_drawing(title: str, labels: list[str], values: list[float]) -> Drawing:
    d = Drawing(420, 180)
    chart = VerticalBarChart()
    chart.x, chart.y = 50, 30
    chart.width, chart.height = 340, 130
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#3b78c2")
    d.add(chart)
    d.add(String(210, 170, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=11))
    return d


def _pie_drawing(title: str, labels: list[str], values: list[float]) -> Drawing:
    d = Drawing(360, 200)
    pie = Pie()
    pie.x, pie.y = 100, 30
    pie.width, pie.height = 150, 150
    pie.data = values
    pie.labels = labels
    pie.slices.strokeWidth = 0.5
    palette = ["#3b78c2", "#7fb069", "#e8a87c", "#c38d9e", "#85cfb1"]
    for i in range(len(values)):
        pie.slices[i].fillColor = colors.HexColor(palette[i % len(palette)])
    d.add(pie)
    d.add(String(180, 190, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=11))
    return d


def _build(path, story, landscape_mode=False):
    pagesize = landscape(LETTER) if landscape_mode else LETTER
    doc = SimpleDocTemplate(
        str(path), pagesize=pagesize,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    doc.build(story)


def ct_a(out: Path) -> Path:
    p = out / "ct_a_bar_chart_plus_table.pdf"
    story = [
        Paragraph("Q1-Q4 Spend Forecast (ACME 2026 Refresh)", _T),
        Paragraph("Section: Quarterly spend distribution", _H2),
        _bar_drawing("Spend by Quarter (USD)", ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"], [125000, 245000, 180000, 90000]),
        Spacer(1, 12),
        Paragraph("Quarterly spend table", _H2),
        _tbl(
            ["Quarter", "Spend USD", "Notes"],
            [
                ["Q1 2026", "$125,000", "Hardware procurement"],
                ["Q2 2026", "$245,000", "Cutover at ATL-HQ-01 + ATL-WEST-02"],
                ["Q3 2026", "$180,000", "ATL-AIR-03 deployment"],
                ["Q4 2026", "$90,000",  "Post-cutover managed service"],
            ],
            [1.2, 1.3, 4.0],
        ),
    ]
    _build(p, story)
    return p


def ct_b(out: Path) -> Path:
    p = out / "ct_b_pie_chart_breakdown.pdf"
    story = [
        Paragraph("Hardware spend breakdown by category", _T),
        _pie_drawing("FY26 Hardware Spend", ["Switches", "APs", "Cameras", "Cabling", "Other"], [120000, 200000, 80000, 40000, 30000]),
        Spacer(1, 14),
        _tbl(
            ["Category", "Amount", "% of Total"],
            [
                ["Switches", "$120,000", "26%"],
                ["Access Points", "$200,000", "43%"],
                ["Cameras", "$80,000", "17%"],
                ["Cabling", "$40,000", "9%"],
                ["Other", "$30,000", "6%"],
            ],
            [1.6, 1.6, 1.4],
        ),
    ]
    _build(p, story)
    return p


def ct_c(out: Path) -> Path:
    p = out / "ct_c_rotated_table.pdf"
    headers = ["Site", "Item", "Mfr", "Model", "Qty", "Unit Price", "Total"]
    rows = [
        ["ATL-HQ-01",   "Switch",  "Cisco", "C9300-48P-A",     "5",  "$3,500", "$17,500"],
        ["ATL-HQ-01",   "AP",      "Cisco", "WAP-9180AX-K9",   "50", "$1,200", "$60,000"],
        ["ATL-WEST-02", "Switch",  "Cisco", "C9300-48P-A",     "3",  "$3,500", "$10,500"],
        ["ATL-WEST-02", "AP",      "Cisco", "WAP-9180AX-K9",   "30", "$1,200", "$36,000"],
        ["ATL-AIR-03",  "SFP",     "Cisco", "SFP-10G-LR-S=",   "12", "$420",   "$5,040"],
    ]
    story = [
        Paragraph("Landscape BOM (wide 7-column layout)", _T),
        _tbl(headers, rows, [1.2, 1.0, 0.9, 1.5, 0.6, 1.2, 1.2]),
    ]
    _build(p, story, landscape_mode=True)
    return p


def ct_d(out: Path) -> Path:
    p = out / "ct_d_nested_subheaders.pdf"
    # Two-row header with column groups
    headers_row_1 = ["", "Q1 2026", "", "Q2 2026", ""]
    headers_row_2 = ["Site", "Hardware", "Services", "Hardware", "Services"]
    data = [
        ["ATL-HQ-01",   "$40K", "$8K", "$60K", "$15K"],
        ["ATL-WEST-02", "$25K", "$5K", "$30K", "$8K"],
        ["ATL-AIR-03",  "$10K", "$2K", "$80K", "$20K"],
    ]
    full_data = [
        [Paragraph(h, _CELL) for h in headers_row_1],
        [Paragraph(h, _CELL) for h in headers_row_2],
    ] + [[Paragraph(c, _CELL) for c in r] for r in data]
    t = Table(full_data, colWidths=[1.5 * inch] + [1.0 * inch] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#dde6f0")),
        ("SPAN", (1, 0), (2, 0)),  # Q1 spans hardware/services
        ("SPAN", (3, 0), (4, 0)),  # Q2 spans hardware/services
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
    ]))
    story = [
        Paragraph("Multi-level column headers (HW/Services × Q1/Q2)", _T),
        t,
    ]
    _build(p, story)
    return p


def ct_e(out: Path) -> Path:
    # Image-only "table" — render a table to a PNG and embed
    p = out / "ct_e_image_only_table.pdf"
    img_path = out / "_ct_e_table.png"
    # Render a tiny table as PNG via reportlab
    try:
        from reportlab.graphics import renderPM
        d = Drawing(400, 120)
        d.add(String(10, 100, "BOM (rendered as image)", fontName="Helvetica-Bold", fontSize=10))
        for i, line in enumerate(["WAP-9180AX-K9   50   $1,200", "C9300-48P-A      5  $3,500", "SFP-10G-LR-S=  12     $420"]):
            d.add(String(10, 80 - i * 18, line, fontName="Helvetica", fontSize=10))
        renderPM.drawToFile(d, str(img_path), fmt="PNG")
        story = [
            Paragraph("Scanned-style image BOM", _T),
            Paragraph("Below: a table rendered as a single image (no text layer).", _BODY),
            Image(str(img_path), width=4 * inch, height=1.2 * inch),
            Paragraph("(In a real scanned PDF, the OCR fallback would recover the data.)", _BODY),
        ]
    except Exception:
        story = [Paragraph("(image rendering skipped)", _BODY)]
    _build(p, story)
    return p


def ct_f(out: Path) -> Path:
    p = out / "ct_f_chart_and_sla_matrix.pdf"
    story = [
        Paragraph("SLA Tier Pricing & Performance", _T),
        _bar_drawing("Monthly Fee USD", ["Bronze", "Silver", "Gold"], [2500, 5000, 8500]),
        Spacer(1, 14),
        _tbl(
            ["Tier", "Monthly Fee", "P1 Response", "P2 Response", "Uptime"],
            [
                ["Bronze", "$2,500", "4 hours",   "8 hours",   "99.0%"],
                ["Silver", "$5,000", "2 hours",   "4 hours",   "99.5%"],
                ["Gold",   "$8,500", "1 hour",    "2 hours",   "99.9%"],
            ],
            [1.0, 1.2, 1.2, 1.2, 1.0],
        ),
    ]
    _build(p, story)
    return p


def ct_g(out: Path) -> Path:
    p = out / "ct_g_color_coded_table.pdf"
    headers = ["Site", "Status", "Risk", "Owner"]
    rows = [
        ["ATL-HQ-01",   "Green",  "Low",     "Facilities"],
        ["ATL-WEST-02", "Amber",  "Medium",  "Operations"],
        ["ATL-AIR-03",  "Red",    "High",    "Security"],
        ["ATL-CP-05",   "Green",  "Low",     "Logistics"],
    ]
    data = [[Paragraph(h, _CELL) for h in headers]] + [
        [Paragraph(c or "", _CELL) for c in r] for r in rows
    ]
    t = Table(data, colWidths=[1.4 * inch, 1.4 * inch, 1.4 * inch, 1.6 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#7fb069")),
        ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#e8a87c")),
        ("BACKGROUND", (1, 3), (1, 3), colors.HexColor("#d9534f")),
        ("BACKGROUND", (1, 4), (1, 4), colors.HexColor("#7fb069")),
    ]))
    story = [
        Paragraph("Site Status Dashboard (RAG color-coded)", _T),
        t,
    ]
    _build(p, story)
    return p


def ct_h(out: Path) -> Path:
    p = out / "ct_h_milestone_timeline_table.pdf"
    headers = ["Milestone", "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]
    rows = [
        ["Kickoff",       "X", "",  "",  ""],
        ["BOM finalized", "X", "",  "",  ""],
        ["Hardware on-site", "", "X", "",  ""],
        ["Cutover ATL-HQ-01", "", "X", "",  ""],
        ["Cutover ATL-WEST-02", "", "X", "",  ""],
        ["Cutover ATL-AIR-03", "", "",  "X", ""],
        ["ATP",           "", "",  "X", ""],
        ["Hypercare end", "", "",  "",  "X"],
    ]
    story = [
        Paragraph("Project Milestone Timeline (Gantt-style)", _T),
        _tbl(headers, rows, [2.5, 1.0, 1.0, 1.0, 1.0]),
    ]
    _build(p, story)
    return p


def ct_i(out: Path) -> Path:
    p = out / "ct_i_split_table_across_pages.pdf"
    headers = ["Row", "Site", "Part", "Qty"]
    rows = [[str(i+1), f"ATL-{['HQ','WEST','AIR','047','CP'][i%5]}-{(i%5)+1:02d}",
             f"PART-{2026:04d}-{i+1:03d}", str((i+1)*3)] for i in range(40)]
    story = [
        Paragraph("Long Table That Spans 2 Pages (40 rows)", _T),
        _tbl(headers, rows, [0.6, 1.4, 2.0, 0.8]),
    ]
    _build(p, story)
    return p


def ct_j(out: Path) -> Path:
    p = out / "ct_j_summary_then_detail.pdf"
    story = [
        Paragraph("Executive Summary", _T),
        Paragraph("Total contract value: $245,000 USD across 3 sites.", _BODY),
        Paragraph("Hardware refresh covers 50 access points, 5 switches, 12 SFP modules.", _BODY),
        Paragraph("Cutover scheduled for Q1 2026 with managed service kickoff Q2 2026.", _BODY),
        _tbl(
            ["Site", "AP Count", "Switch Count", "Spend"],
            [
                ["ATL-HQ-01",   "50", "5",  "$140,000"],
                ["ATL-WEST-02", "30", "3",  "$70,000"],
                ["ATL-AIR-03",  "0",  "0",  "$35,000"],
            ],
            [1.6, 1.0, 1.4, 1.4],
        ),
        PageBreak(),
        Paragraph("Detailed BOM", _T),
        _tbl(
            ["Site", "Part Number", "Description", "Qty", "Unit", "Total"],
            [
                ["ATL-HQ-01",   "WAP-9180AX-K9", "Wi-Fi 6E AP",    "50", "$1,200", "$60,000"],
                ["ATL-HQ-01",   "C9300-48P-A",   "48-port switch", "5",  "$3,500", "$17,500"],
                ["ATL-WEST-02", "WAP-9180AX-K9", "Wi-Fi 6E AP",    "30", "$1,200", "$36,000"],
                ["ATL-WEST-02", "C9300-48P-A",   "48-port switch", "3",  "$3,500", "$10,500"],
                ["ATL-AIR-03",  "SFP-10G-LR-S=", "10G LR module",  "12", "$420",   "$5,040"],
            ],
            [1.1, 1.3, 1.5, 0.6, 1.0, 1.0],
        ),
    ]
    _build(p, story)
    return p


def ct_k(out: Path) -> Path:
    p = out / "ct_k_charts_with_callouts.pdf"
    story = [
        Paragraph("FY26 Wireless AP Deployment Plan", _T),
        _bar_drawing("APs by Site", ["ATL-HQ-01", "ATL-WEST-02", "ATL-AIR-03", "ATL-CP-05"], [50, 30, 20, 10]),
        Paragraph(
            "Callouts: Tier-1 sites (ATL-HQ-01, ATL-WEST-02) deploy first in Q1. "
            "Tier-2 (ATL-AIR-03, ATL-CP-05) follow in Q2. Total 110 APs across all 4 sites.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def ct_l(out: Path) -> Path:
    p = out / "ct_l_pricing_with_subtotals.pdf"
    headers = ["Section", "Item", "Qty", "Unit", "Total"]
    rows = [
        ["Hardware",   "Wi-Fi APs",         "100", "$1,200",  "$120,000"],
        ["Hardware",   "Switches",          "11",  "$3,500",  "$38,500"],
        ["Hardware",   "SFP modules",       "12",  "$420",    "$5,040"],
        ["Subtotal Hardware", "",            "",    "",        "$163,540"],
        ["Services",   "Installation labor","1",   "$45,000", "$45,000"],
        ["Services",   "Project mgmt",      "1",   "$25,000", "$25,000"],
        ["Subtotal Services", "",            "",    "",        "$70,000"],
        ["Managed",    "12-month Silver",   "12",  "$5,000",  "$60,000"],
        ["Subtotal Managed",  "",            "",    "",        "$60,000"],
        ["GRAND TOTAL",       "",            "",    "",        "$293,540"],
    ]
    story = [
        Paragraph("Pricing with Subtotals + Grand Total", _T),
        _tbl(headers, rows, [1.5, 2.0, 0.6, 1.0, 1.2]),
    ]
    _build(p, story)
    return p


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_pdf_charts_tables.py <out_dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    builders = [
        ct_a, ct_b, ct_c, ct_d, ct_e, ct_f,
        ct_g, ct_h, ct_i, ct_j, ct_k, ct_l,
    ]
    for b in builders:
        p = b(out)
        print(f"  -> {p.name}")
    print(f"\n{len(builders)} chart/table PDFs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
