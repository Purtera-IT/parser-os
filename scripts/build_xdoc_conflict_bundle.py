"""Build a cross-doc test bundle that exercises the new device-only
quantity-conflict detector.

Produces three PDFs in <out_dir>/artifacts/:

  bom.pdf             BOM/quote-style line items — "50 access points"
  sow.pdf             SOW/scope-style prose       — "60 access points"
  no_conflict.pdf     Same device but at a DIFFERENT site (no conflict)

The expected behavior:
  - Compile detects ONE cross-artifact contradiction between bom.pdf
    and sow.pdf (same device:access_point, qty 50 vs 60).
  - no_conflict.pdf says "20 access points at ATL-AIR-03" — should
    NOT pair with bom.pdf (which is at ATL-HQ-01).
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


_S = getSampleStyleSheet()
_TITLE = ParagraphStyle("t", parent=_S["Title"], fontSize=14, spaceAfter=10)
_H2 = ParagraphStyle("h2", parent=_S["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
_BODY = ParagraphStyle("b", parent=_S["Normal"], fontSize=10, leading=13, spaceAfter=6)
_CELL = ParagraphStyle("c", parent=_S["Normal"], fontSize=9)


def _build(path, story):
    doc = SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    doc.build(story)


def _tbl(headers, rows, widths):
    data = [[Paragraph(h, _CELL) for h in headers]] + [
        [Paragraph(c or "", _CELL) for c in r] for r in rows
    ]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
    ]))
    return t


def build_bom(out):
    p = out / "bom.pdf"
    story = [
        Paragraph("Acme Refresh - Hardware BOM (v1)", _TITLE),
        Paragraph(
            "PurTera shall ship 50 wireless access points to ATL-HQ-01 for the Q1 2026 refresh.",
            _BODY,
        ),
        Spacer(1, 12),
        Paragraph(
            "BOM total quantities are authoritative for shipping.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def build_sow(out):
    p = out / "sow.pdf"
    story = [
        Paragraph("Acme Refresh - Statement of Work (v3)", _TITLE),
        Paragraph(
            "PurTera will install 60 access points at ATL-HQ-01 across the renovation footprint.",
            _BODY,
        ),
        Spacer(1, 12),
        Paragraph(
            "SOW scope governs deployed quantities.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def build_no_conflict(out):
    p = out / "no_conflict.pdf"
    story = [
        Paragraph("Acme Refresh - Airport Annex Scope", _TITLE),
        Paragraph("Section: ATL-AIR-03", _H2),
        Paragraph(
            "PurTera will install 20 access points at ATL-AIR-03 during phase 2.",
            _BODY,
        ),
    ]
    _build(p, story)
    return p


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_xdoc_conflict_bundle.py <out_dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  -> {build_bom(out).name}")
    print(f"  -> {build_sow(out).name}")
    print(f"  -> {build_no_conflict(out).name}")
    print(f"\n3 cross-doc PDFs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
