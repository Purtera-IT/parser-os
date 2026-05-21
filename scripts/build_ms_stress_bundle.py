"""Build an adversarial managed-services artifact bundle.

Produces PDFs that exercise the edge cases that show up in real
managed-services deals (refresh, install, IMAC, decom, MSA add-on):

  ms_a_quantity_conflict_bom_vs_sow.pdf
      BOM says 50 APs, SOW says 60 wireless devices → must produce
      quantity_conflict warning.

  ms_b_pricing_conflict_quote_contract_co.pdf
      Quote $100K, contract $95K, change order +$10K → must surface
      all three with provenance.

  ms_c_date_format_chaos.pdf
      "Mar 15, 2026" / "2026-03-15" / "Q1 2026" / "FY26 Q2" → must
      normalize to date entities without dropping any.

  ms_d_stakeholder_roles.pdf
      "Director of Workplace Technology, OPTBOT" / "Program Manager,
      PurTera" → must produce stakeholder entities with role + org.

  ms_e_msa_boilerplate.pdf
      WHEREAS clauses + indemnity + force majeure → must NOT pollute
      scope_item atoms.

  ms_f_toc_and_footers.pdf
      Table of contents listing 8 sections + repeated page footers →
      TOC entries must NOT become individual scope items; footers
      must be suppressed.

  ms_g_watermark_draft.pdf
      "DRAFT - DO NOT DISTRIBUTE" stamp on every page + scope text →
      stamp must not flood scope atoms.

  ms_h_sla_constraints.pdf
      "Response time within 4 business hours" / "Resolution within
      24 hours" / "Uptime 99.9%" → must produce constraint atoms.

  ms_i_multi_currency.pdf
      Line items in USD + EUR + GBP → must NOT silently sum across
      currencies.

  ms_j_change_order_adds_removes.pdf
      Add 5 cameras at ATL-AIR-03 / Remove 2 cameras at ATL-WEST-02
      → must produce two quantity deltas.

  ms_k_service_tiers.pdf
      Bronze / Silver / Gold tier table → must capture tier → price
      mapping.

  ms_l_part_number_chaos.pdf
      "WAP-9180AX-K9 v2", "CAT6A/F-UTP", "SFP-10G-LR-S=", "Cisco
      C9300-48P-A" → must capture as part_number entities.

  ms_m_empty_placeholders.pdf
      Table cells with TBD / TBA / N/A / — / pending → must NOT
      produce data atoms.

  ms_n_long_paragraph.pdf
      One single 2000-char paragraph → must still extract entities.

  ms_o_compliance_sla_matrix.pdf
      Tier × metric matrix (Bronze/Silver/Gold × Response/Resolution/
      Uptime) → must extract each cell as a constraint with tier.

Run:
  python scripts/build_ms_stress_bundle.py <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


_STYLES = getSampleStyleSheet()
_NORMAL = _STYLES["Normal"]
_TITLE = ParagraphStyle("t", parent=_STYLES["Title"], fontSize=14, spaceAfter=10)
_H2 = ParagraphStyle("h2", parent=_STYLES["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
_BODY = ParagraphStyle("b", parent=_NORMAL, fontSize=10, leading=13, spaceAfter=6)
_CELL = ParagraphStyle("c", parent=_NORMAL, fontSize=9, leading=11)
_FOOTER = ParagraphStyle("f", parent=_NORMAL, fontSize=7, textColor=colors.grey, alignment=1)


def _tbl(headers, rows, widths):
    data = [[Paragraph(h, _CELL) for h in headers]] + [[Paragraph(c or "", _CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _build(path, story, page_footer=None):
    def _draw_footer(canvas, doc):
        if page_footer:
            canvas.saveState()
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.grey)
            canvas.drawCentredString(LETTER[0] / 2, 0.35 * inch, page_footer)
            canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=path.stem,
    )
    if page_footer:
        doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    else:
        doc.build(story)


# ── A. quantity conflict ─────────────────────────────────────────


def a_quantity_conflict(out):
    p = out / "ms_a_quantity_conflict_bom_vs_sow.pdf"
    story = [
        Paragraph("Project: Refresh ACME-2026", _TITLE),
        Paragraph("Section: BOM Allocation", _H2),
        Paragraph("Total wireless access points to ship: 50 units (Aruba AP-635).", _BODY),
        Paragraph("Section: SOW Scope Summary", _H2),
        Paragraph("PurTera shall furnish and install 60 wireless access points at the customer's three sites.", _BODY),
        Paragraph("Note: discrepancy between BOM (50) and SOW (60) is intentional; reviewer must catch.", _BODY),
    ]
    _build(p, story)
    return p


# ── B. pricing conflict ──────────────────────────────────────────


def b_pricing_conflict(out):
    p = out / "ms_b_pricing_conflict_quote_contract_co.pdf"
    story = [
        Paragraph("Project: Refresh ACME-2026", _TITLE),
        Paragraph("Section: Original Quote (rev 1)", _H2),
        Paragraph("Grand Total: USD $100,000.00 - valid through 2026-04-30.", _BODY),
        Paragraph("Section: Executed Contract (rev 3)", _H2),
        Paragraph("Final Contract Price: USD $95,000.00 (3 sites, fixed price).", _BODY),
        Paragraph("Section: Change Order CO-001", _H2),
        Paragraph("Add additional 8 access points: +USD $10,000.00. Total after CO-001: USD $105,000.00.", _BODY),
    ]
    _build(p, story)
    return p


# ── C. date format chaos ─────────────────────────────────────────


def c_date_format_chaos(out):
    p = out / "ms_c_date_format_chaos.pdf"
    story = [
        Paragraph("Project Schedule (multi-format)", _TITLE),
        Paragraph("Section 1: Milestones", _H2),
        Paragraph("Kickoff: Mar 15, 2026.", _BODY),
        Paragraph("Site survey complete: 2026-03-22.", _BODY),
        Paragraph("Hardware on-site: Q1 2026 (no later than end of March).", _BODY),
        Paragraph("Cutover: FY26 Q2 (April-June 2026).", _BODY),
        Paragraph("ATP: 6/30/26.", _BODY),
        Paragraph("Final acceptance: June 30, 2026.", _BODY),
    ]
    _build(p, story)
    return p


# ── D. stakeholder roles ─────────────────────────────────────────


def d_stakeholder_roles(out):
    p = out / "ms_d_stakeholder_roles.pdf"
    story = [
        Paragraph("Approvals (signature block)", _TITLE),
        Paragraph("OPTBOT - Director of Workplace Technology: Jane Roe", _BODY),
        Paragraph("OPTBOT - VP, IT Infrastructure: John Smith", _BODY),
        Paragraph("PurTera - Program Manager: Maria Lopez", _BODY),
        Paragraph("PurTera - Senior Solutions Architect: T. Nguyen", _BODY),
        Paragraph("Office of the CIO: Sarah Chen, CIO", _BODY),
        Paragraph("Approved by Owner-Architect: Alex Patel, AIA.", _BODY),
    ]
    _build(p, story)
    return p


# ── E. MSA boilerplate ───────────────────────────────────────────


def e_msa_boilerplate(out):
    p = out / "ms_e_msa_boilerplate.pdf"
    story = [
        Paragraph("Master Services Agreement - Boilerplate", _TITLE),
        Paragraph("Section 12: Indemnification", _H2),
        Paragraph("Each party (the \"Indemnifying Party\") shall indemnify, defend, and hold harmless the other party "
                  "and its affiliates, officers, directors, employees, and agents (collectively, the \"Indemnified Parties\") "
                  "from and against any and all third-party claims, damages, losses, costs, and expenses, including reasonable "
                  "attorneys' fees, arising out of or in connection with the Indemnifying Party's breach of this Agreement.",
                  _BODY),
        Paragraph("Section 13: Force Majeure", _H2),
        Paragraph("Neither party shall be liable for any failure or delay in performance under this Agreement "
                  "(other than payment obligations) resulting from acts of God, war, terrorism, government action, fire, "
                  "flood, earthquake, pandemic, or other event beyond the reasonable control of the affected party.",
                  _BODY),
        Paragraph("Section 14: Confidentiality", _H2),
        Paragraph("All non-public information disclosed by either party in connection with this Agreement shall be "
                  "treated as confidential and protected as such for a period of five (5) years.", _BODY),
    ]
    _build(p, story)
    return p


# ── F. TOC + footer ──────────────────────────────────────────────


def f_toc_and_footers(out):
    p = out / "ms_f_toc_and_footers.pdf"
    story = [
        Paragraph("Statement of Work - ACME 2026", _TITLE),
        Paragraph("Table of Contents", _H2),
        Paragraph("1. Project Overview ............................... 2", _BODY),
        Paragraph("2. Scope of Work ................................... 3", _BODY),
        Paragraph("3. Deliverables .................................... 5", _BODY),
        Paragraph("4. Schedule ........................................ 6", _BODY),
        Paragraph("5. Pricing ......................................... 7", _BODY),
        Paragraph("6. Acceptance Criteria ............................. 8", _BODY),
        Paragraph("7. Change Management ............................... 9", _BODY),
        Paragraph("8. Signatures ..................................... 10", _BODY),
        PageBreak(),
        Paragraph("1. Project Overview", _H2),
        Paragraph("This SOW covers the refresh of ACME's three-site campus to support hybrid work patterns. "
                  "PurTera will deliver hardware, installation labor, and 12 months of T2 managed service.", _BODY),
        Paragraph("Actual scope: 50 wireless access points, 5 distribution switches, 2 firewalls.", _BODY),
    ]
    _build(p, story, page_footer="Confidential | Page X of Y | (c) 2026 PurTera | DO NOT REDISTRIBUTE")
    return p


# ── G. watermark / DRAFT ─────────────────────────────────────────


def g_watermark_draft(out):
    p = out / "ms_g_watermark_draft.pdf"
    story = [
        Paragraph("DRAFT - DO NOT DISTRIBUTE", _TITLE),
        Paragraph("Project Scope (DRAFT - DO NOT DISTRIBUTE)", _H2),
        Paragraph("Install 50 access points at ATL-HQ-01. DRAFT.", _BODY),
        Paragraph("Install 30 access points at ATL-WEST-02. DRAFT.", _BODY),
        Paragraph("Install 20 access points at ATL-AIR-03. DRAFT.", _BODY),
        Paragraph("DRAFT - DO NOT DISTRIBUTE - DRAFT - DO NOT DISTRIBUTE", _BODY),
    ]
    _build(p, story)
    return p


# ── H. SLA constraints ──────────────────────────────────────────


def h_sla_constraints(out):
    p = out / "ms_h_sla_constraints.pdf"
    story = [
        Paragraph("Service Level Agreement - Managed Services", _TITLE),
        Paragraph("Incident Response Times", _H2),
        Paragraph("Priority 1 (Service down): Response within 1 hour, resolution within 4 hours.", _BODY),
        Paragraph("Priority 2 (Degraded): Response within 4 business hours, resolution within 24 hours.", _BODY),
        Paragraph("Priority 3 (Minor): Response within 1 business day, resolution within 5 business days.", _BODY),
        Paragraph("Uptime Commitment", _H2),
        Paragraph("Network uptime: 99.9% measured monthly, excluding planned maintenance.", _BODY),
        Paragraph("Wi-Fi uptime: 99.5% measured monthly.", _BODY),
        Paragraph("Service credits apply if monthly uptime falls below 99.5%.", _BODY),
    ]
    _build(p, story)
    return p


# ── I. multi-currency ────────────────────────────────────────────


def i_multi_currency(out):
    p = out / "ms_i_multi_currency.pdf"
    headers = ["Item", "Qty", "Unit Price", "Total"]
    rows = [
        ["Cisco WAP-635 (US allocation)",   "50",  "$1,000.00 USD",  "$50,000.00 USD"],
        ["Cisco WAP-635 (EMEA allocation)", "30",  "EUR 920.00",      "EUR 27,600.00"],
        ["Cisco WAP-635 (UK allocation)",   "20",  "GBP 780.00",      "GBP 15,600.00"],
    ]
    story = [
        Paragraph("Multi-currency BOM rollup", _TITLE),
        _tbl(headers, rows, [3.0, 0.7, 1.6, 1.7]),
        Spacer(1, 8),
        Paragraph("Totals must NOT be summed across currencies (do not produce a single dollar figure).", _BODY),
    ]
    _build(p, story)
    return p


# ── J. change order add/remove ───────────────────────────────────


def j_change_order(out):
    p = out / "ms_j_change_order_adds_removes.pdf"
    story = [
        Paragraph("Change Order CO-002", _TITLE),
        Paragraph("Additions", _H2),
        Paragraph("Add 5 mini-dome CCTV cameras at ATL-AIR-03 (loading dock coverage gap).", _BODY),
        Paragraph("Add 2 additional patch panels at ATL-HQ-01 telecom room MDF-3A.", _BODY),
        Paragraph("Removals", _H2),
        Paragraph("Remove 2 ceiling-mount cameras at ATL-WEST-02 conference rooms (canceled).", _BODY),
        Paragraph("Remove 1 access point at ATL-047-04 training kitchen (room reassigned).", _BODY),
        Paragraph("Net change: +$8,500 USD; net device count delta: +4 devices.", _BODY),
    ]
    _build(p, story)
    return p


# ── K. service tiers ─────────────────────────────────────────────


def k_service_tiers(out):
    p = out / "ms_k_service_tiers.pdf"
    headers = ["Tier", "Monthly fee", "Response P1", "Response P2", "Uptime", "After-hours"]
    rows = [
        ["Bronze",  "$2,500",  "4 hours",   "8 hours",   "99.0%", "Excluded"],
        ["Silver",  "$5,000",  "2 hours",   "4 hours",   "99.5%", "Included (1.5x)"],
        ["Gold",    "$8,500",  "1 hour",    "2 hours",   "99.9%", "Included (1.0x)"],
    ]
    story = [
        Paragraph("Managed Service Tier Pricing", _TITLE),
        _tbl(headers, rows, [0.8, 1.1, 1.0, 1.0, 0.9, 1.6]),
    ]
    _build(p, story)
    return p


# ── L. part number chaos ────────────────────────────────────────


def l_part_number_chaos(out):
    p = out / "ms_l_part_number_chaos.pdf"
    headers = ["Mfr", "Part Number", "Description", "Qty"]
    rows = [
        ["Cisco",   "C9300-48P-A",       "48-port PoE+ switch",      "5"],
        ["Cisco",   "WAP-9180AX-K9 v2",  "Wi-Fi 6E access point",    "50"],
        ["CommScope", "CAT6A/F-UTP-1000ft", "Plenum cable, 1000ft",  "8"],
        ["Cisco",   "SFP-10G-LR-S=",     "10GBASE-LR module",        "12"],
        ["Aruba",   "JL664A#ABA",         "Aruba 6300 stack module", "2"],
        ["Generic", "CAB-PWR-NA",         "Power cord, NA",          "75"],
    ]
    story = [
        Paragraph("Hardware Bill of Materials", _TITLE),
        _tbl(headers, rows, [1.0, 1.6, 2.8, 0.6]),
    ]
    _build(p, story)
    return p


# ── M. empty placeholders ───────────────────────────────────────


def m_empty_placeholders(out):
    p = out / "ms_m_empty_placeholders.pdf"
    headers = ["Site", "Decision", "Date", "Owner"]
    rows = [
        ["ATL-HQ-01",  "Approved", "2026-03-15", "OPTBOT"],
        ["ATL-WEST-02", "TBD",     "—",          "TBD"],
        ["ATL-AIR-03", "Pending",  "TBA",        "Pending"],
        ["ATL-047-04", "N/A",      "",           "—"],
        ["ATL-CP-05",  "Approved", "2026-03-22", "PurTera"],
    ]
    story = [
        Paragraph("Site Approval Matrix (with placeholders)", _TITLE),
        _tbl(headers, rows, [1.2, 1.3, 1.5, 1.5]),
    ]
    _build(p, story)
    return p


# ── N. long paragraph ────────────────────────────────────────────


def n_long_paragraph(out):
    p = out / "ms_n_long_paragraph.pdf"
    long = (
        "This managed services engagement covers the refresh of ACME's enterprise wireless network "
        "across three sites: ATL-HQ-01 (the Atlanta headquarters), ATL-WEST-02 (West Campus), and "
        "ATL-AIR-03 (Airport Logistics). The total scope includes 50 Wi-Fi 6E access points, 5 "
        "distribution switches with 48-port PoE+ density, 2 redundant firewall pairs in HA mode, "
        "12 SFP+ 10GBASE-LR optical modules for backbone links, and 8 spools of Category 6A plenum "
        "cable totaling approximately 8,000 linear feet. PurTera shall furnish and install all "
        "hardware, perform site surveys at each of the three sites prior to installation, complete "
        "the cutover during the maintenance windows defined in the project schedule (kickoff Mar 15, "
        "2026; ATP no later than June 30, 2026), and provide 12 months of T2 managed service "
        "(Silver tier: 2-hour P1 response, 4-hour P2 response, 99.5% uptime SLA) after final "
        "acceptance. The contract value is $345,000.00 USD fixed-price with a 5% holdback released "
        "upon completion of the 30-day post-cutover stabilization window. Change orders are billed "
        "T&M at $165/hr standard, $248/hr after-hours, materials at cost plus 15%."
    )
    story = [
        Paragraph("Full Scope (Single Paragraph)", _TITLE),
        Paragraph(long, _BODY),
    ]
    _build(p, story)
    return p


# ── O. compliance matrix ────────────────────────────────────────


def o_compliance_matrix(out):
    p = out / "ms_o_compliance_sla_matrix.pdf"
    headers = ["Metric", "Bronze", "Silver", "Gold"]
    rows = [
        ["P1 Response time",    "4 hours",  "2 hours",  "1 hour"],
        ["P2 Response time",    "8 hours",  "4 hours",  "2 hours"],
        ["Resolution P1",       "24 hours", "8 hours",  "4 hours"],
        ["Network uptime",      "99.0%",    "99.5%",    "99.9%"],
        ["Wi-Fi uptime",        "98.5%",    "99.0%",    "99.5%"],
        ["After-hours support", "Excluded", "1.5x",     "Included"],
        ["Quarterly review",    "No",       "Yes",      "Yes (with exec)"],
    ]
    story = [
        Paragraph("SLA / Compliance Matrix by Tier", _TITLE),
        _tbl(headers, rows, [2.4, 1.2, 1.2, 1.4]),
    ]
    _build(p, story)
    return p


# ── Main ────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_ms_stress_bundle.py <out_dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    builders = [
        a_quantity_conflict, b_pricing_conflict, c_date_format_chaos,
        d_stakeholder_roles, e_msa_boilerplate, f_toc_and_footers,
        g_watermark_draft, h_sla_constraints, i_multi_currency,
        j_change_order, k_service_tiers, l_part_number_chaos,
        m_empty_placeholders, n_long_paragraph, o_compliance_matrix,
    ]
    for b in builders:
        p = b(out)
        print(f"  -> {p.name}")
    print(f"\n{len(builders)} adversarial MS PDFs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
