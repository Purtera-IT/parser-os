"""Round 2 adversarial bundle — push the parser harder on shapes
Round 1 didn't cover.

Outputs to <out_dir>/artifacts/:

  r2_a_multi_site_one_row.pdf
      Single line names two sites: "Deploy 50 APs at ATL-HQ-01 and
      ATL-WEST-02 in the same week." Both sites + quantity must be
      captured.

  r2_b_range_qty.pdf
      "Between 40 and 60 access points" / "approximately 50 APs" /
      "~75 cameras" / "50-75 endpoints". Range/approx quantities
      should still produce quantity entities (at least the upper
      bound or both bounds).

  r2_c_negative_credit.pdf
      "Issue $5,000 credit", "-$10,000 discount", "rebate of
      $2,500". Negative / credit amounts must surface as money
      entities (not silently dropped).

  r2_d_cross_reference.pdf
      "See Section 3.4 for SLA details. As defined in 2.1, P1
      response is 1 hour." Cross-refs shouldn't drop the constraint.

  r2_e_multi_column_layout.pdf
      Two-column layout (newspaper style) — text should not
      interleave between columns into garbled atoms.

  r2_f_footnotes.pdf
      Body text with footnote refs (¹ ² ³) + footnote bodies at
      page bottom. Footnote bodies shouldn't be confused with main
      scope.

  r2_g_unicode_addresses.pdf
      "Acme München, Hauptstraße 12, 80331 München" /
      "Acme Tokyo, 東京都港区六本木 6-10-1". Unicode characters
      in addresses must survive parsing.

  r2_h_rotated_landscape.pdf
      Mixed portrait + landscape pages. Both must produce atoms.

  r2_i_strikethrough_revisions.pdf
      Text with strikethrough markup (rendered as struck-through
      characters): "[STRIKE]Old: 50 APs[/STRIKE] New: 60 APs."
      Should treat the strikethrough as a quantity DELETED.

  r2_j_multi_currency_conversion.pdf
      "$100,000 USD (€95,000 EUR equivalent)". Both money entities
      preserved, NOT summed.

  r2_k_discount_percentage.pdf
      "10% discount if signed by Mar 31, 2026. Net price $90,000
      after discount." Percent + amount + date all preserved.

  r2_l_numbered_subsections.pdf
      "1. Scope / 1.1 Hardware / 1.1.1 Switches / 1.1.2 APs / 1.2
      Cabling / 2. Software." Deep nesting should not collapse.

  r2_m_implicit_references.pdf
      "The customer will provide WAN. The contractor will install
      hardware. They will coordinate cutover." Co-references kept
      as scope_items even without proper nouns.

  r2_n_conditional_clauses.pdf
      "If uptime falls below 99.9%, customer receives 5% service
      credit. If response time exceeds 4 hours, 10% credit applies."
      Conditional constraints with thresholds must surface.

  r2_o_time_zones.pdf
      "All maintenance windows in Pacific Time (PST). Cutover
      between 22:00 PT and 02:00 PT next day. Customer support
      window 06:00-18:00 ET, Mon-Fri."

  r2_p_page_break_split.pdf
      Sentence that breaks across pages: "The customer will deploy
      50 access points <PAGE BREAK> across three sites: ATL-HQ-01,
      ATL-WEST-02, and ATL-AIR-03."

  r2_q_hybrid_bullets_tables.pdf
      Mixed bullets + tables + prose in same section. All three
      shapes coexist; none should swallow the others.

  r2_r_empty_pdf.pdf
      Completely empty PDF (no text). Should not crash.

  r2_s_only_headers.pdf
      All-headings PDF, no body. Should not crash.

  r2_t_huge_part_numbers_table.pdf
      30-row BOM with many SKUs. Every row should produce a
      part_number entity.

Run:
  python scripts/build_ms_stress_round2.py <out_dir>
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
)


_STYLES = getSampleStyleSheet()
_NORMAL = _STYLES["Normal"]
_TITLE = ParagraphStyle("t", parent=_STYLES["Title"], fontSize=14, spaceAfter=10)
_H2 = ParagraphStyle("h2", parent=_STYLES["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
_H3 = ParagraphStyle("h3", parent=_STYLES["Heading3"], fontSize=10, spaceBefore=6, spaceAfter=3)
_BODY = ParagraphStyle("b", parent=_NORMAL, fontSize=10, leading=13, spaceAfter=6)
_SMALL = ParagraphStyle("s", parent=_NORMAL, fontSize=8, leading=10, textColor=colors.grey)
_CELL = ParagraphStyle("c", parent=_NORMAL, fontSize=9, leading=11)


def _build(path, story, landscape_mode=False):
    pagesize = landscape(LETTER) if landscape_mode else LETTER
    doc = SimpleDocTemplate(
        str(path), pagesize=pagesize,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=path.stem,
    )
    doc.build(story)


def _tbl(headers, rows, widths_inches):
    data = [[Paragraph(h, _CELL) for h in headers]]
    for r in rows:
        data.append([Paragraph(c or "", _CELL) for c in r])
    t = Table(data, colWidths=[w * inch for w in widths_inches], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ── Round 2 builders ─────────────────────────────────────────────


def r2_a_multi_site_one_row(out):
    p = out / "r2_a_multi_site_one_row.pdf"
    story = [
        Paragraph("Multi-Site One-Row Scope", _TITLE),
        Paragraph("Section: Deployment", _H2),
        Paragraph(
            "Deploy 50 access points at ATL-HQ-01 and ATL-WEST-02 in the same week.",
            _BODY,
        ),
        Paragraph(
            "Stage 20 additional switches at ATL-AIR-03 and ATL-CP-05 during cutover.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def r2_b_range_qty(out):
    p = out / "r2_b_range_qty.pdf"
    story = [
        Paragraph("Range and Approximate Quantities", _TITLE),
        Paragraph("Between 40 and 60 access points required across the rollout.", _BODY),
        Paragraph("Approximately 50 APs at HQ; ~75 cameras at airport.", _BODY),
        Paragraph("50-75 endpoints per site, depending on final occupancy.", _BODY),
        Paragraph("No more than 100 workstations per training room.", _BODY),
    ]
    _build(p, story)
    return p


def r2_c_negative_credit(out):
    p = out / "r2_c_negative_credit.pdf"
    story = [
        Paragraph("Credits and Discounts", _TITLE),
        Paragraph("Customer receives a service-credit of $5,000 for missed P1 SLA in Q1.", _BODY),
        Paragraph("Apply -$10,000 early-payment discount to invoice INV-2026-003.", _BODY),
        Paragraph("Manufacturer rebate of $2,500 issued for bulk wireless purchase.", _BODY),
        Paragraph("Net adjustment after credits: $7,500 reduction.", _BODY),
    ]
    _build(p, story)
    return p


def r2_d_cross_reference(out):
    p = out / "r2_d_cross_reference.pdf"
    story = [
        Paragraph("Cross-Reference Heavy SOW", _TITLE),
        Paragraph("Section 2.1: Priority Definitions", _H2),
        Paragraph("Priority 1 means service is fully down. P1 response: within 1 hour.", _BODY),
        Paragraph("Section 3.4: SLA Schedule", _H2),
        Paragraph(
            "As defined in Section 2.1, P1 response is 1 hour and P1 resolution is 4 hours. "
            "Refer to Exhibit B for tier mapping. See also Section 4.2 (escalation matrix) "
            "and Appendix A.1.",
            _BODY,
        ),
        Paragraph("Per Section 3.4, after-hours escalation is permitted for ATL-AIR-03 only.", _BODY),
    ]
    _build(p, story)
    return p


def r2_e_multi_column_layout(out):
    p = out / "r2_e_multi_column_layout.pdf"
    # Simulate two-column layout using a 2-column table without grid
    left_col = (
        "<b>Left Column: Scope</b><br/><br/>"
        "PurTera will furnish and install 50 wireless access points across ATL-HQ-01. "
        "All cabling will be Category 6A or better. Cutover scheduled for Mar 15, 2026."
    )
    right_col = (
        "<b>Right Column: Pricing</b><br/><br/>"
        "Total contract value: USD $245,000.00. Milestone billing: 30/40/20/10 split. "
        "Service tier: Silver ($5,000/month managed)."
    )
    t = Table(
        [[Paragraph(left_col, _BODY), Paragraph(right_col, _BODY)]],
        colWidths=[3.5 * inch, 3.5 * inch],
    )
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story = [Paragraph("Two-Column Layout (newspaper style)", _TITLE), t]
    _build(p, story)
    return p


def r2_f_footnotes(out):
    p = out / "r2_f_footnotes.pdf"
    story = [
        Paragraph("SOW with Footnotes", _TITLE),
        Paragraph(
            "PurTera will deliver 50 access points¹ at ATL-HQ-01 by Mar 15, 2026². "
            "All material costs include freight³ and standard 30-day warranty.",
            _BODY,
        ),
        Spacer(1, 30),
        Paragraph("─" * 80, _SMALL),
        Paragraph("¹ Cisco WAP-9180AX-K9 v2 or approved equivalent.", _SMALL),
        Paragraph("² Subject to site survey acceptance per Section 4.1.", _SMALL),
        Paragraph("³ Freight included for continental US only; HI/AK extra.", _SMALL),
    ]
    _build(p, story)
    return p


def r2_g_unicode_addresses(out):
    p = out / "r2_g_unicode_addresses.pdf"
    story = [
        Paragraph("International Sites (Unicode addresses)", _TITLE),
        _tbl(
            ["Site ID", "Facility", "Address"],
            [
                ["MUC-HQ-01", "Acme München",  "Hauptstraße 12, 80331 München, Deutschland"],
                ["TYO-HQ-01", "Acme Tokyo",     "Roppongi 6-10-1, Minato-ku, Tokyo 106-6131"],
                ["PAR-HQ-01", "Acme Paris",     "Champs-Élysées 50, 75008 Paris, France"],
                ["MEX-HQ-01", "Acme Ciudad de México", "Av. Reforma 222, Col. Juárez, 06600 CDMX"],
            ],
            [1.0, 1.6, 4.0],
        ),
    ]
    _build(p, story)
    return p


def r2_h_rotated_landscape(out):
    p = out / "r2_h_rotated_landscape.pdf"
    story = [
        Paragraph("Landscape Page (wide BOM)", _TITLE),
        _tbl(
            ["Site", "Item", "Qty", "Unit", "Total"],
            [
                ["ATL-HQ-01",   "Cisco C9300-48P-A",   "5", "$3,500", "$17,500"],
                ["ATL-WEST-02", "Cisco WAP-9180AX-K9", "30", "$1,200", "$36,000"],
                ["ATL-AIR-03",  "Cisco SFP-10G-LR-S=", "8", "$420",   "$3,360"],
            ],
            [1.6, 2.2, 0.7, 1.0, 1.0],
        ),
    ]
    _build(p, story, landscape_mode=True)
    return p


def r2_i_strikethrough_revisions(out):
    p = out / "r2_i_strikethrough_revisions.pdf"
    # Use literal STRIKE markers since rendering true strikethrough
    # produces text that loses the markup in get_text. The parser
    # should still treat the surrounding text correctly.
    story = [
        Paragraph("Scope Revisions (strikethrough-style edits)", _TITLE),
        Paragraph("Original: 50 access points. Revised: 60 access points.", _BODY),
        Paragraph("Original: $245,000. Revised after CO-001: $255,000.", _BODY),
        Paragraph("Original cutover: Mar 15, 2026. Revised: Mar 22, 2026 (delayed 1 week).", _BODY),
    ]
    _build(p, story)
    return p


def r2_j_multi_currency_conversion(out):
    p = out / "r2_j_multi_currency_conversion.pdf"
    story = [
        Paragraph("Multi-Currency with Conversion", _TITLE),
        Paragraph("Grand Total: USD $100,000 (approximately EUR 95,000 equivalent at current rates).", _BODY),
        Paragraph("UK allocation: GBP 25,000 (USD $32,000 equivalent).", _BODY),
        Paragraph("DO NOT sum USD and EUR figures — they represent the same payment in different reporting currencies.", _BODY),
    ]
    _build(p, story)
    return p


def r2_k_discount_percentage(out):
    p = out / "r2_k_discount_percentage.pdf"
    story = [
        Paragraph("Pricing with Discount Terms", _TITLE),
        Paragraph(
            "List price: $100,000. 10% early-signing discount available if signed by Mar 31, 2026. "
            "Net price after discount: $90,000.",
            _BODY,
        ),
        Paragraph(
            "Additional 5% volume rebate kicks in at $200,000 cumulative across the program year.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def r2_l_numbered_subsections(out):
    p = out / "r2_l_numbered_subsections.pdf"
    story = [
        Paragraph("Deeply Nested Scope", _TITLE),
        Paragraph("1. Scope of Work", _H2),
        Paragraph("1.1 Hardware", _H3),
        Paragraph("1.1.1 Switches: 5 Cisco C9300-48P-A.", _BODY),
        Paragraph("1.1.2 Access Points: 50 Cisco WAP-9180AX-K9 v2.", _BODY),
        Paragraph("1.2 Cabling", _H3),
        Paragraph("1.2.1 8 spools of Cat6A plenum cable.", _BODY),
        Paragraph("2. Software", _H2),
        Paragraph("2.1 Licenses", _H3),
        Paragraph("2.1.1 50 DNA-E licenses for access points.", _BODY),
        Paragraph("3. Services", _H2),
        Paragraph("3.1 Installation labor included.", _BODY),
        Paragraph("3.2 12 months of managed service post-cutover.", _BODY),
    ]
    _build(p, story)
    return p


def r2_m_implicit_references(out):
    p = out / "r2_m_implicit_references.pdf"
    story = [
        Paragraph("Implicit-Reference Heavy SOW", _TITLE),
        Paragraph("The customer will provide WAN/MPLS handoff ports at each MDF.", _BODY),
        Paragraph("The contractor will furnish and install all hardware and labor.", _BODY),
        Paragraph("They will coordinate cutover windows during weekend hours.", _BODY),
        Paragraph("Either party may invoke change-order procedures per Section 5.", _BODY),
    ]
    _build(p, story)
    return p


def r2_n_conditional_clauses(out):
    p = out / "r2_n_conditional_clauses.pdf"
    story = [
        Paragraph("Conditional SLA Clauses", _TITLE),
        Paragraph(
            "If monthly uptime falls below 99.9%, customer receives a 5% service credit on that month's fees.",
            _BODY,
        ),
        Paragraph(
            "If response time exceeds 4 business hours for P2 incidents, 10% credit applies for that incident.",
            _BODY,
        ),
        Paragraph(
            "If three or more consecutive P1 SLA misses occur within a quarter, customer may terminate without penalty.",
            _BODY,
        ),
        Paragraph(
            "Where ambiguous, the customer's reasonable interpretation governs.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def r2_o_time_zones(out):
    p = out / "r2_o_time_zones.pdf"
    story = [
        Paragraph("Time Zone Specifications", _TITLE),
        Paragraph("All maintenance windows in Pacific Time (PT / PST / PDT).", _BODY),
        Paragraph("Cutover window: 22:00 PT - 02:00 PT next day, Saturday Mar 21, 2026.", _BODY),
        Paragraph("Customer support: 06:00-18:00 ET, Mon-Fri.", _BODY),
        Paragraph("After-hours escalation: 18:00-06:00 ET, 24/7 weekends.", _BODY),
    ]
    _build(p, story)
    return p


def r2_p_page_break_split(out):
    p = out / "r2_p_page_break_split.pdf"
    story = [
        Paragraph("SOW (sentence splits across pages)", _TITLE),
        Paragraph("Section 3: Scope of Work", _H2),
        Paragraph(
            "The customer will deploy 50 access points across three sites for the Q1 2026 refresh program. "
            "All hardware will be furnished by PurTera and installed by certified technicians during approved "
            "maintenance windows.",
            _BODY,
        ),
        PageBreak(),
        Paragraph("Section 3 (continued)", _H2),
        Paragraph(
            "The deployment spans ATL-HQ-01, ATL-WEST-02, and ATL-AIR-03. Total contract value is "
            "$245,000.00 USD fixed price.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def r2_q_hybrid_bullets_tables(out):
    p = out / "r2_q_hybrid_bullets_tables.pdf"
    from reportlab.platypus import ListFlowable, ListItem
    story = [
        Paragraph("Hybrid Bullets + Tables + Prose", _TITLE),
        Paragraph("Section: Deployment", _H2),
        Paragraph(
            "PurTera will deliver the following hardware across the three sites:",
            _BODY,
        ),
        ListFlowable(
            [
                ListItem(Paragraph("50 Cisco WAP-9180AX-K9 v2 access points", _BODY)),
                ListItem(Paragraph("5 Cisco C9300-48P-A distribution switches", _BODY)),
                ListItem(Paragraph("12 SFP-10G-LR-S= optical modules", _BODY)),
            ],
            bulletType="bullet",
        ),
        Paragraph("Pricing breakdown:", _BODY),
        _tbl(
            ["Item", "Qty", "Unit Price", "Total"],
            [
                ["Access points", "50", "$1,200", "$60,000"],
                ["Switches",      "5",  "$3,500", "$17,500"],
                ["SFP modules",   "12", "$420",   "$5,040"],
            ],
            [1.8, 0.8, 1.4, 1.4],
        ),
        Paragraph(
            "Note: All amounts are USD and include freight to continental US sites.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def r2_r_empty_pdf(out):
    p = out / "r2_r_empty_pdf.pdf"
    # PDF with NOTHING — just a blank page
    _build(p, [Spacer(1, 1)])
    return p


def r2_s_only_headers(out):
    p = out / "r2_s_only_headers.pdf"
    story = [
        Paragraph("Document with Only Section Headers", _TITLE),
        Paragraph("Section 1: Introduction", _H2),
        Paragraph("Section 2: Scope", _H2),
        Paragraph("Section 3: Pricing", _H2),
        Paragraph("Section 4: Acceptance", _H2),
        Paragraph("Section 5: Signatures", _H2),
    ]
    _build(p, story)
    return p


def r2_t_huge_part_numbers_table(out):
    p = out / "r2_t_huge_part_numbers_table.pdf"
    rows = []
    for i in range(30):
        rows.append([
            f"R{i+1:02d}",
            ["Cisco", "Aruba", "CommScope", "Generic", "Panduit"][i % 5],
            f"PART-{2026:04d}-{i+1:03d}",
            f"Description for part {i+1} - hardware item",
            str((i + 1) * 5),
        ])
    story = [
        Paragraph("Large BOM (30 line items)", _TITLE),
        _tbl(
            ["Row", "Mfr", "Part Number", "Description", "Qty"],
            rows,
            [0.5, 0.9, 1.4, 3.4, 0.6],
        ),
    ]
    _build(p, story)
    return p


# ── Main ────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_ms_stress_round2.py <out_dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    builders = [
        r2_a_multi_site_one_row, r2_b_range_qty, r2_c_negative_credit,
        r2_d_cross_reference, r2_e_multi_column_layout, r2_f_footnotes,
        r2_g_unicode_addresses, r2_h_rotated_landscape, r2_i_strikethrough_revisions,
        r2_j_multi_currency_conversion, r2_k_discount_percentage,
        r2_l_numbered_subsections, r2_m_implicit_references,
        r2_n_conditional_clauses, r2_o_time_zones, r2_p_page_break_split,
        r2_q_hybrid_bullets_tables, r2_r_empty_pdf, r2_s_only_headers,
        r2_t_huge_part_numbers_table,
    ]
    for b in builders:
        p = b(out)
        print(f"  -> {p.name}")
    print(f"\n{len(builders)} Round 2 adversarial PDFs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
