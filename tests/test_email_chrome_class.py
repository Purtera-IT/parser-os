"""Mail chrome is its own class, not a suppressed line (deal 000132).

Every email on 000132 listed 33-64 "suppressed" lines, nearly all of them
signature rows ("PurTera-IT.com<https://urldefense.com/...>", "Email:
x@y.com<mailto:x@y.com>"), the header rows of a quoted message ("From:",
"Sent:", "To:", "Cc:", "Subject:") and quoted greetings ("Hi John,"). The
labeler's Missed tab reads suppressed lines as possible misses, so all of
that was noise. Those lines are now ``chrome`` with a reason; a real dropped
fact stays ``suppressed``.
"""
from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import orbitbrief_envelope as env
from app.core.email_chrome import atom_chrome_reason, chrome_reason
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.text_coverage import coverage_for_artifact

BODY = """Hi John,

We need 40 Cat6 drops on the second floor before the 14th.

Thanks,
Jane Roe
PurTera-IT.com<https://urldefense.com/v3/__https://purtera-it.com__;!!abc$>
➔ Email: jane@purtera-it.com<mailto:jane@purtera-it.com>
Mobile: 555-123-4567

From: John Smith <john@client.example>
Sent: Monday, September 14, 2026 9:12 AM
To: Jane Roe <jane@purtera-it.com>
Cc: Ops Team <ops@client.example>
Subject: RE: Cabling

Hi Jane,

The riser room is locked after 6pm and the super has the only key.
"""

CHROME_LINES = [
    "Hi John,",
    "Thanks,",
    "PurTera-IT.com<https://urldefense.com/v3/__https://purtera-it.com__;!!abc$>",
    "➔ Email: jane@purtera-it.com<mailto:jane@purtera-it.com>",
    "Mobile: 555-123-4567",
    "From: John Smith <john@client.example>",
    "Sent: Monday, September 14, 2026 9:12 AM",
    "To: Jane Roe <jane@purtera-it.com>",
    "Cc: Ops Team <ops@client.example>",
    "Subject: RE: Cabling",
    "Hi Jane,",
]


def _atom(text: str, *, artifact_type=ArtifactType.email, value=None, flags=None, artifact="art_m",
          filename="m.eml"):
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**10}", project_id="p", artifact_id=artifact,
        atom_type=AtomType.scope_item, raw_text=text, normalized_text=text.lower(),
        value=value or {"kind": "email_body_line"}, entity_keys=[],
        source_refs=[SourceRef(id="s1", artifact_id=artifact, artifact_type=artifact_type,
                               filename=filename, locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted,
        review_flags=list(flags or ["suppressed:quoted_history_dedup"]), parser_version="t",
    )


def _eml(tmp_path: Path) -> Path:
    m = EmailMessage()
    m["From"] = "jane@purtera-it.com"
    m["To"] = "john@client.example"
    m["Subject"] = "RE: Cabling"
    m.set_content(BODY)
    p = tmp_path / "m.eml"
    p.write_bytes(bytes(m))
    return p


@pytest.mark.parametrize("line", CHROME_LINES)
def test_mail_chrome_shapes_are_recognised(line):
    assert chrome_reason(line), line


@pytest.mark.parametrize("line", [
    "We need 40 Cat6 drops on the second floor before the 14th.",
    "The riser room is locked after 6pm and the super has the only key.",
    "Diagram: <https://example.com/floorplan.pdf>",
    "Please send the updated quote to john@client.example by Friday.",
    "Hi John, the lift is booked for Tuesday and the riser key is with the super.",
])
def test_content_is_not_chrome(line):
    assert chrome_reason(line) is None


def test_coverage_lists_chrome_with_reason_not_as_suppressed(tmp_path):
    kept = [_atom("We need 40 Cat6 drops on the second floor before the 14th.")]
    # A stage dropped the chrome AND a real fact.
    dropped = [_atom(t) for t in CHROME_LINES] + [
        _atom("The riser room is locked after 6pm and the super has the only key."),
    ]
    cov = coverage_for_artifact(_eml(tmp_path), "art_m", kept, dropped)
    by_text = {x["text"]: x for x in cov["unclaimed"]}
    for line in CHROME_LINES:
        row = by_text[line.strip()]
        assert row["state"] == "chrome", (line, row)
        assert row.get("reason"), row
    # the name under the sign-off is the signature, not a dropped fact
    assert by_text["Jane Roe"]["state"] == "chrome"
    riser = [x for x in cov["unclaimed"] if "riser room" in x["text"]]
    assert riser and riser[0]["state"] == "suppressed"
    assert cov["suppressed_count"] == 1
    assert cov["chrome_count"] >= len(CHROME_LINES)
    assert cov["unread_count"] == 0


@pytest.fixture
def carrying_the_ledger(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")


def test_envelope_splits_chrome_out_of_the_suppressed_ledger(carrying_the_ledger):
    chrome = [_atom(t) for t in CHROME_LINES]
    fact = _atom("The riser room is locked after 6pm and the super has the only key.")
    # A survey PDF's "Phone:" row is a site contact, not mail chrome.
    pdf_contact = _atom("Phone: 555-987-6543", artifact_type=ArtifactType.pdf, artifact="art_pdf",
                         filename="survey.pdf")
    # An admission-regex reject carries its own reason.
    regex_reject = _atom("Best regards,", value={"admission_regex": "signoff"},
                         flags=["admission_regex", "suppressed:quoted_chatter_dedup"])
    result = SimpleNamespace(suppressed_atoms=chrome + [fact, pdf_contact, regex_reject], project_id="d1")

    shown = env._suppressed_for_review(result, [])
    assert {r["text"] for r in shown} == {fact.raw_text, pdf_contact.raw_text}
    assert env._suppressed_total(result) == 2

    side = env._suppressed_chrome_for_review(result)
    assert len(side) == len(CHROME_LINES) + 1
    assert all(r["reason"] and r["stage"] for r in side)
    assert {r["reason"] for r in side} >= {"greeting", "signoff", "quoted_header",
                                           "wrapped_link", "signature_contact"}
    assert env._suppressed_chrome_total(result) == len(CHROME_LINES) + 1
    assert atom_chrome_reason(pdf_contact) is None


def test_chrome_side_list_is_off_with_the_ledger(monkeypatch):
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", raising=False)
    result = SimpleNamespace(suppressed_atoms=[_atom("Hi John,")], project_id="d1")
    assert env._suppressed_chrome_for_review(result) == []
    assert env._suppressed_chrome_total(result) == 0


# ---------------------------------------------------------------------------
# Signature images and e-sign page stamps never stay atoms (010087).

GUID = "3F2A9C1E-1B2C-4D5E-9F00-ABCDEF123456"


def test_signature_image_without_a_signoff_is_chrome(tmp_path, monkeypatch):
    """A logo under a contact block with no "Thanks," above it was typed scope
    under "Equipment list"; the screenshot the body introduces keeps it."""
    import base64

    import app.parsers.email_parser as ep

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ) + b"\x00" * 4096
    ocr = {
        b"EQ": "Rack: 42U four post rack in the main MDF with two 20A circuits and ladder tray",
        b"LG": "Amtivo Certification Services 100 Park Ave Suite 1600 New York NY 10017",
    }
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    monkeypatch.setattr(ep, "_ocr_text_from_cid_inline",
                        lambda payload, content_type="": next(
                            (v for k, v in ocr.items() if bytes(payload).endswith(k)), ""))
    plain = ("Hi Trent,\n\nPlease see the full equipment list below.\n[cid:eq@x]\n\n"
             "Stephanie Hechsel | Account Manager\nM: 555-201-3344\n"
             "stephanie.hechsel@amtivo.com\n[cid:image001.png@x]\n")
    html = plain.replace("\n", "<br>")
    parts: list[str] = []
    for cid, key in (("eq@x", "EQ"), ("image001.png@x", "LG")):
        html = html.replace(f"[cid:{cid}]", f'<img src="cid:{cid}">')
        parts += ["--rel", "Content-Type: image/png", "Content-Transfer-Encoding: base64",
                  f"Content-ID: <{cid}>", "", base64.b64encode(png + key.encode()).decode()]
    lines = ["From: Stephanie Hechsel <stephanie.hechsel@amtivo.com>", "To: Trent <t@purtera-it.com>",
             "Subject: Equipment list", "Date: Tue, 7 Jul 2026 15:12:00 -0400", "MIME-Version: 1.0",
             'Content-Type: multipart/related; boundary="rel"', "",
             "--rel", 'Content-Type: multipart/alternative; boundary="alt"', "",
             "--alt", "Content-Type: text/plain; charset=utf-8", "", plain,
             "--alt", "Content-Type: text/html; charset=utf-8", "", f"<html><body>{html}</body></html>",
             "--alt--"] + parts + ["--rel--", ""]
    path = tmp_path / "m.eml"
    path.write_text("\r\n".join(lines), encoding="utf-8")
    atoms = ep.EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=path).atoms
    logo = [a for a in atoms if (a.value or {}).get("content_id") == "image001.png"]
    assert logo, [a.raw_text for a in atoms]
    assert all((a.value or {}).get("admission_regex") == "signature_image" for a in logo)
    eq = [a for a in atoms if (a.value or {}).get("content_id") == "eq"]
    assert eq and not any("chatter" in (a.review_flags or []) for a in eq)

    # The compiler's diversion takes the logo out of the atom set into the
    # ledger, and the envelope lists it as chrome with its reason.
    from app.core.email_chrome import divert_chrome

    ledger: list = []
    kept = divert_chrome(list(atoms), ledger)
    assert not any((a.value or {}).get("content_id") == "image001.png" for a in kept)
    assert eq[0] in kept
    assert {a.id for a in ledger} == {a.id for a in logo}
    assert all("suppressed:chrome" in a.review_flags for a in ledger)
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    result = SimpleNamespace(suppressed_atoms=ledger, project_id="p")
    assert env._suppressed_for_review(result, kept) == []
    rows = env._suppressed_chrome_for_review(result)
    assert rows and {r["reason"] for r in rows} == {"signature_image"}
    assert {r["stage"] for r in rows} == {"chrome"}


def test_esign_stamp_is_diverted_to_chrome(monkeypatch):
    """#280 kept the DocuSign stamp as a deal_metadata chatter atom; it reached
    every list as an atom. It is chrome: out of the atoms, in the chrome list."""
    from app.core.email_chrome import divert_chrome
    from app.parsers.orbitbrief_pdf import _flag_doc_stamps

    stamp = _atom(f"Docusign Envelope ID: {GUID}", artifact_type=ArtifactType.pdf,
                  artifact="art_sow", filename="Signed SOW.pdf", flags=[])
    fact = _atom("PurTera will install 24 cameras across the warehouse.",
                 artifact_type=ArtifactType.pdf, artifact="art_sow", filename="Signed SOW.pdf", flags=[])
    _flag_doc_stamps([stamp, fact])
    ledger: list = []
    kept = divert_chrome([stamp, fact], ledger)
    assert kept == [fact] and ledger == [stamp]
    assert stamp.value["_suppression"] == {"stage": "chrome", "reason": "esign_stamp"}
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    result = SimpleNamespace(suppressed_atoms=ledger, project_id="p")
    assert env._suppressed_for_review(result, kept) == []
    assert [r["reason"] for r in env._suppressed_chrome_for_review(result)] == ["esign_stamp"]


def test_esign_stamp_line_is_chrome_in_pdf_coverage(tmp_path):
    fitz = pytest.importorskip("fitz")
    from app.core.email_chrome import divert_chrome
    from app.parsers.orbitbrief_pdf import _flag_doc_stamps

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 40), f"Docusign Envelope ID: {GUID}", fontsize=8)
    page.insert_text((36, 104), "PurTera will install 24 cameras across the warehouse.", fontsize=10)
    path = tmp_path / "Signed SOW.pdf"
    doc.save(str(path))
    stamp = _atom(f"Docusign Envelope ID: {GUID}", artifact_type=ArtifactType.pdf,
                  artifact="art_sow", filename="Signed SOW.pdf", flags=[])
    fact = _atom("PurTera will install 24 cameras across the warehouse.",
                 artifact_type=ArtifactType.pdf, artifact="art_sow", filename="Signed SOW.pdf", flags=[])
    _flag_doc_stamps([stamp, fact])
    ledger: list = []
    kept = divert_chrome([stamp, fact], ledger)
    cov = coverage_for_artifact(path, "art_sow", kept, ledger)
    rows = [x for x in cov["unclaimed"] if "Envelope ID" in x["text"]]
    assert rows and rows[0]["state"] == "chrome" and rows[0]["reason"] == "esign_stamp"
    assert cov["suppressed_count"] == 0 and cov["unread_count"] == 0
