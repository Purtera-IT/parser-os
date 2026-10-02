"""An image in an email's signature is the sender's, not the equipment list's (010087).

Stephanie's email introduced an equipment screenshot ("Please see the full
equipment list below.") and signed off with her company's images: a logo, an
address banner, a certification badge. The equipment handler OCRs every
inline image of such an email and gave each whole-image reading the
"Equipment list" heading, so the banner and the badge (too long, and with
digits, for the one-word logo rule) were typed scope_item under the
equipment list. An image set after the sign-off is signature chatter on its
own message, with no borrowed heading; the screenshot keeps it.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

import app.parsers.email_parser as ep

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
) + b"\x00" * 4096
OCR = {
    b"EQ": "Rack: 42U four post rack in the main MDF with two 20A circuits and ladder tray",
    b"L2": "Amtivo Certification Services 100 Park Ave Suite 1600 New York NY 10017 www.amtivo.com",
    b"L3": "ISO 9001 Certified Quality Management System UKAS 0012",
}


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _eml(path: Path) -> None:
    plain = ("Hi Trent,\n\nPlease see the full equipment list below.\n[cid:eq@x]\n\n"
             "Let me know if you have questions.\n\nThanks,\nStephanie Hechsel\nAccount Manager\n"
             "[cid:l2@x]\n[cid:l3@x]\n")
    html = plain.replace("\n", "<br>")
    parts: list[str] = []
    for cid in ("EQ", "L2", "L3"):
        html = html.replace(f"[cid:{cid.lower()}@x]", f'<img src="cid:{cid.lower()}@x">')
        parts += ["--rel", "Content-Type: image/png", "Content-Transfer-Encoding: base64",
                  f"Content-ID: <{cid.lower()}@x>", "", base64.b64encode(PNG + cid.encode()).decode()]
    lines = ["From: Stephanie Hechsel <stephanie.hechsel@amtivo.com>", "To: Trent <t@purtera-it.com>",
             "Subject: Equipment list", "Date: Tue, 7 Jul 2026 15:12:00 -0400", "MIME-Version: 1.0",
             'Content-Type: multipart/related; boundary="rel"', "",
             "--rel", 'Content-Type: multipart/alternative; boundary="alt"', "",
             "--alt", "Content-Type: text/plain; charset=utf-8", "", plain,
             "--alt", "Content-Type: text/html; charset=utf-8", "", f"<html><body>{html}</body></html>",
             "--alt--"] + parts + ["--rel--", ""]
    path.write_text("\r\n".join(lines), encoding="utf-8")


def test_signature_images_are_chatter_on_their_email_not_equipment(tmp_path: Path, monkeypatch) -> None:
    def fake(payload, content_type=""):
        return next((v for k, v in OCR.items() if bytes(payload).endswith(k)), "")

    monkeypatch.setattr(ep, "_ocr_text_from_cid_inline", fake)
    path = tmp_path / "m.eml"
    _eml(path)
    atoms = ep.EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=path).atoms
    by = {a.raw_text: a for a in atoms if (a.value or {}).get("kind") == "email_cid_inline_body"}
    eq = by[OCR[b"EQ"]]
    assert "Equipment list" in eq.source_refs[0].locator["section_path"]
    assert "chatter" not in eq.review_flags
    for key in (b"L2", b"L3"):
        a = by[OCR[key]]
        assert "chatter" in a.review_flags, a.review_flags
        assert "section_path" not in a.source_refs[0].locator, a.source_refs[0].locator
        assert "lead_in" not in a.value
        assert a.source_refs[0].locator["message_index"] == eq.source_refs[0].locator["message_index"]
