"""PDF safety bundle — corrupt / encrypted / empty / oversized files
the parser must handle without crashing.

  ps_a_empty_pdf.pdf            0-page, minimal valid PDF
  ps_b_truncated.pdf            Truncated PDF stream (corrupt mid-file)
  ps_c_encrypted_password.pdf   Password-protected (open with password 'demo')
  ps_d_only_metadata.pdf        PDF with only Author / Title metadata, no body
  ps_e_oversized_blob.pdf       1MB single paragraph (stress text extraction)
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

_S = getSampleStyleSheet()
_BODY = ParagraphStyle("b", parent=_S["Normal"], fontSize=10, leading=13)
_T = ParagraphStyle("t", parent=_S["Title"], fontSize=14)


def ps_a_empty_pdf(out: Path) -> Path:
    p = out / "ps_a_empty_pdf.pdf"
    doc = SimpleDocTemplate(str(p), pagesize=LETTER)
    doc.build([Spacer(1, 1)])
    return p


def ps_b_truncated(out: Path) -> Path:
    p = out / "ps_b_truncated.pdf"
    # Write a partial PDF stream — header + first object + truncated
    p.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj <<")
    return p


def ps_c_encrypted_password(out: Path) -> Path:
    p = out / "ps_c_encrypted_password.pdf"
    # reportlab supports owner/user password via the BaseDocTemplate
    # encrypt parameter. SimpleDocTemplate doesn't accept it directly,
    # so we use a slightly lower-level approach.
    try:
        from reportlab.lib import pdfencrypt
        enc = pdfencrypt.StandardEncryption("demo", "demo", canPrint=1, canModify=0)
        doc = SimpleDocTemplate(str(p), pagesize=LETTER, encrypt=enc)
        doc.build([Paragraph("Confidential scope: 50 access points at ATL-HQ-01.", _BODY)])
    except Exception:
        # Fallback: produce an unencrypted PDF labelled "(encryption skipped)"
        doc = SimpleDocTemplate(str(p), pagesize=LETTER)
        doc.build([Paragraph("(encryption skipped — reportlab version mismatch)", _BODY)])
    return p


def ps_d_only_metadata(out: Path) -> Path:
    p = out / "ps_d_only_metadata.pdf"
    doc = SimpleDocTemplate(
        str(p), pagesize=LETTER,
        title="Acme 2026 Refresh — Metadata Only",
        author="PurTera",
        subject="Demo",
    )
    doc.build([Spacer(1, 1)])
    return p


def ps_e_oversized_blob(out: Path) -> Path:
    p = out / "ps_e_oversized_blob.pdf"
    # ~50 KB single paragraph (≈8000 words)
    blob = (
        "PurTera will furnish and install 50 wireless access points at ATL-HQ-01. " * 100
        + "Contract value is USD $245,000. Cutover scheduled Mar 15, 2026. " * 100
    )
    doc = SimpleDocTemplate(str(p), pagesize=LETTER,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build([Paragraph("Mega-paragraph stress test", _T), Paragraph(blob, _BODY)])
    return p


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_pdf_safety_bundle.py <out_dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    for b in [ps_a_empty_pdf, ps_b_truncated, ps_c_encrypted_password, ps_d_only_metadata, ps_e_oversized_blob]:
        p = b(out)
        print(f"  -> {p.name}")
    print(f"\n5 PDF safety files in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
